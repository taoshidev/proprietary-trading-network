import statistics
import bittensor as bt
import threading
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional

from flask import Flask, jsonify, request, Response, g
import os
import time
import json
import gzip
import traceback
from setproctitle import setproctitle
from waitress import serve
from flask_compress import Compress
from bittensor_wallet import Keypair

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig
from multiprocessing import current_process
from ptn_api.api_key_refresh import APIKeyMixin
from ptn_api.nonce_manager import NonceManager


class APIMetricsTracker:
    """
    Tracks API usage metrics and logs them periodically.
    Uses a rolling time window approach to track:
    1. API key usage counts
    2. Endpoint performance metrics (request count, avg processing time)
    3. Failed request tracking
    """

    def __init__(self, log_interval_minutes: int = 5, api_key_mapping: Dict = None):
        """
        Initialize the metrics tracker with the given log interval.

        Args:
            log_interval_minutes: How often to log metrics (in minutes)
        """
        self.log_interval_minutes = log_interval_minutes
        self.log_interval_seconds = log_interval_minutes * 60

        # Maps API key name to deque of timestamps
        self.api_key_hits: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10000))

        # Maps endpoint to deque of (timestamp, latency) tuples
        self.endpoint_hits: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=10000))

        # Track failed requests: maps (user_id, endpoint, status_code) to deque of timestamps
        self.failed_requests: Dict[Tuple[str, str, int], Deque[float]] = defaultdict(lambda: deque(maxlen=1000))

        # Lock for thread safety
        self.metrics_lock = threading.Lock()

        # Reference to API key to user ID mapping
        self.api_key_to_alias = api_key_mapping or {}  # Use provided mapping or empty dict

        # Start logging thread
        self.start_logging_thread()

    def track_request(self, api_key: str, endpoint: str, duration: float, status_code: int = 200):
        """
        Track a request with its associated API key, endpoint, and duration.

        Args:
            api_key: The API key used for the request
            endpoint: The endpoint that was accessed
            duration: Request processing time in seconds
            status_code: HTTP status code of the response
        """
        # Get user_id from api_key if available
        user_id = self.api_key_to_alias.get(api_key, f"unknown_key_{api_key[:8] if api_key else 'none'}")

        now = time.time()

        with self.metrics_lock:
            self.api_key_hits[user_id].append(now)
            self.endpoint_hits[endpoint].append((now, duration))

            # Track failed requests
            if status_code >= 400:
                self.failed_requests[(user_id, endpoint, status_code)].append(now)

    def log_metrics(self):
        """Log the current metrics based on the rolling time window."""
        current_time = time.time()
        cutoff_time = current_time - self.log_interval_seconds

        # Process metrics with lock to ensure thread safety
        api_counts = {}
        endpoint_stats = {}
        failed_stats = {}

        with self.metrics_lock:
            # Process API key hits
            empty_keys = []
            for key, timestamps in self.api_key_hits.items():
                # Remove outdated entries
                while timestamps and timestamps[0] < cutoff_time:
                    timestamps.popleft()

                # Count remaining hits in the window
                count = len(timestamps)
                if count > 0:
                    api_counts[key] = count
                else:
                    # Mark empty keys for removal
                    empty_keys.append(key)

            # Remove empty keys
            for key in empty_keys:
                del self.api_key_hits[key]

            # Process endpoint hits
            empty_endpoints = []
            for endpoint, entries in self.endpoint_hits.items():
                # Remove outdated entries
                while entries and entries[0][0] < cutoff_time:
                    entries.popleft()

                # Calculate stats for remaining hits
                count = len(entries)
                if count > 0:
                    # Extract just the durations
                    durations = [duration for _, duration in entries]
                    # Calculate multiple statistics
                    stats = {
                        "count": count,
                        "mean": statistics.mean(durations),
                        "median": statistics.median(durations),
                        "min": min(durations),
                        "max": max(durations)
                    }
                    # Remove None values for percentiles that couldn't be calculated
                    stats = {k: v for k, v in stats.items() if v is not None}
                    endpoint_stats[endpoint] = stats
                else:
                    # Mark empty endpoints for removal
                    empty_endpoints.append(endpoint)

            # Remove empty endpoints
            for endpoint in empty_endpoints:
                del self.endpoint_hits[endpoint]

            # Process failed requests
            empty_failed = []
            for key, timestamps in self.failed_requests.items():
                # Remove outdated entries
                while timestamps and timestamps[0] < cutoff_time:
                    timestamps.popleft()

                count = len(timestamps)
                if count > 0:
                    failed_stats[key] = count
                else:
                    empty_failed.append(key)

            # Remove empty failed request entries
            for key in empty_failed:
                del self.failed_requests[key]

        # Skip logging if there's no activity
        if not api_counts and not endpoint_stats and not failed_stats:
            bt.logging.info(f"No API activity in the last {self.log_interval_minutes} minutes")
            return

        # Format and log the metrics report
        log_lines = [f"\n===== API Metrics (Last {self.log_interval_minutes} minutes) ====="]

        # Log API key usage
        log_lines.append("\nAPI Key Usage:")
        if api_counts:
            for key, count in sorted(api_counts.items(), key=lambda x: x[1], reverse=True):
                log_lines.append(f"  {key}: {count} requests")
        else:
            log_lines.append("  No API requests in this period")

        # Log endpoint metrics
        log_lines.append("\nEndpoint Performance:")
        if endpoint_stats:
            for endpoint, stats in sorted(endpoint_stats.items(),
                                          key=lambda x: x[1]["count"], reverse=True):
                log_lines.append(f"  {endpoint}: {stats['count']} requests")
                log_lines.append(f"    mean: {stats['mean'] * 1000:.2f}ms")
                log_lines.append(f"    median: {stats['median'] * 1000:.2f}ms")
                log_lines.append(f"    min/max: {stats['min'] * 1000:.2f}ms / {stats['max'] * 1000:.2f}ms")
        else:
            log_lines.append("  No endpoint activity in this period")

        # Log failed requests
        if failed_stats:
            log_lines.append("\nFailed Requests:")
            for (user_id, endpoint, status_code), count in sorted(failed_stats.items(),
                                                                  key=lambda x: x[1], reverse=True):
                log_lines.append(f"  {user_id} -> {endpoint} [{status_code}]: {count} failures")

        # Log the complete report
        final_str = "\n".join(log_lines)
        bt.logging.info(final_str)

    def periodic_logging(self):
        """Periodically log metrics based on the configured interval."""
        while True:
            # Sleep for the log interval
            time.sleep(self.log_interval_seconds)

            # Log metrics with exception handling
            try:
                self.log_metrics()
            except Exception as e:
                print(f"Error in metrics logging: {e}")
                traceback.print_exc()

    def start_logging_thread(self):
        """Start the periodic logging thread."""
        logging_thread = threading.Thread(target=self.periodic_logging, daemon=True)
        logging_thread.start()
        bt.logging.info(f"API metrics logging started (interval: {self.log_interval_minutes} minutes)")


class PTNRestServer(APIKeyMixin):
    """Handles REST API requests with Flask and Waitress."""

    def __init__(self, api_keys_file, shared_queue=None, host="127.0.0.1",
                 port=48888, refresh_interval=15, metrics_interval_minutes=5, position_manager=None, contract_manager=None):
        """Initialize the REST server with API key handling and routing.

        Args:
            api_keys_file: Path to the JSON file containing API keys
            shared_queue: Optional shared queue for communication with WebSocket server
            host: Hostname or IP to bind the server to
            port: Port to bind the server to
            refresh_interval: How often to check for API key changes (seconds)
            metrics_interval_minutes: How often to log API metrics (minutes)
            position_manager: Optional position manager for handling miner positions
            contract_manager: Optional contract manager for handling collateral operations
        """
        # Initialize API key handling
        APIKeyMixin.__init__(self, api_keys_file, refresh_interval)

        # REST server configuration
        self.shared_queue = shared_queue
        self.position_manager: PositionManager = position_manager
        self.contract_manager = contract_manager
        self.nonce_manager = NonceManager()
        self.data_path = ValiConfig.BASE_DIR
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        self.contract_manager.load_contract_owner_credentials()

        # Initialize Flask-Compress for GZIP compression
        Compress(self.app)

        # Initialize metrics tracking
        self._setup_metrics(metrics_interval_minutes)

        # Register routes
        self._register_routes()

        # Register error handlers
        self._register_error_handlers()

        # Start API key refresh thread
        self.start_refresh_thread()
        print(f"[{current_process().name}] RestServer initialized with {len(self.accessible_api_keys)} API keys")

    def _setup_metrics(self, metrics_interval_minutes):
        """Set up API metrics tracking."""
        # Initialize the metrics tracker as instance variable
        self.metrics = APIMetricsTracker(metrics_interval_minutes, self.api_key_to_alias)

        # Set up Flask request hooks for automatic metrics tracking
        @self.app.before_request
        def start_timer():
            # Store start time in Flask's g object for this request
            g.start_time = time.time()

        @self.app.after_request
        def record_metrics(response):
            # Calculate request duration
            end_time = time.time()
            duration = end_time - getattr(g, 'start_time', end_time)

            # Get API key - handle errors gracefully
            try:
                api_key = self._get_api_key_safe()
            except Exception:
                api_key = None

            # Get endpoint (use rule if available, otherwise path)
            url = request.url_rule.rule if request.url_rule else request.path

            # Track the request using the instance metrics tracker
            self.metrics.track_request(api_key, url, duration, response.status_code)

            return response

    def _register_error_handlers(self):
        """Register custom error handlers for common exceptions."""

        @self.app.errorhandler(400)
        def handle_bad_request(e):
            # Log the error with user context
            api_key = self._get_api_key_safe()
            user_id = self.api_key_to_alias.get(api_key, f"unknown_key_{api_key[:8] if api_key else 'none'}")

            bt.logging.warning(
                f"Bad Request: user={user_id} endpoint={request.path} method={request.method} "
                f"error={str(e).split(':')[0] if ':' in str(e) else str(e)[:50]}"
            )

            return jsonify({'error': 'Bad request'}), 400

        @self.app.errorhandler(401)
        def handle_unauthorized(e):
            return jsonify({'error': 'Unauthorized access'}), 401

        @self.app.errorhandler(403)
        def handle_forbidden(e):
            return jsonify({'error': 'Forbidden'}), 403

        @self.app.errorhandler(404)
        def handle_not_found(e):
            return jsonify({'error': 'Resource not found'}), 404

        @self.app.errorhandler(500)
        def handle_internal_error(e):
            # Log the error with user context
            api_key = self._get_api_key_safe()
            user_id = self.api_key_to_alias.get(api_key, f"unknown_key_{api_key[:8] if api_key else 'none'}")

            bt.logging.error(
                f"Internal Error: user={user_id} endpoint={request.path} method={request.method} "
                f"error={str(e)[:100]}"
            )

            return jsonify({'error': 'Internal server error'}), 500

        @self.app.errorhandler(Exception)
        def handle_exception(e):
            # Log unexpected errors
            api_key = self._get_api_key_safe()
            user_id = self.api_key_to_alias.get(api_key, f"unknown_key_{api_key[:8] if api_key else 'none'}")

            bt.logging.error(
                f"Unhandled Exception: user={user_id} endpoint={request.path} method={request.method} "
                f"error_type={type(e).__name__} error={str(e)[:100]}"
            )

            # Only log full traceback for truly unexpected errors
            if not isinstance(e, (json.JSONDecodeError, KeyError, ValueError)):
                bt.logging.debug(f"Full traceback:\n{traceback.format_exc()}")

            return jsonify({'error': 'An error occurred processing your request'}), 500

    def _register_routes(self):
        """Register all API routes."""

        @self.app.route("/miner-positions", methods=["GET"])
        def get_miner_positions():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Get the 'tier' query parameter from the request
            requested_tier = str(request.args.get('tier', 100))
            is_gz_data = True

            # Validate the 'tier' parameter
            if requested_tier not in ['0', '30', '50', '100']:
                return jsonify({'error': 'Invalid tier value. Allowed values are 0, 30, 50, or 100'}), 400

            # Check if API key has sufficient tier access
            if not self.can_access_tier(api_key, int(requested_tier)):
                return jsonify({'error': f'Your API key does not have access to tier {requested_tier} data'}), 403

            f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)

            # Attempt to retrieve the file
            data = self._get_file(f, binary=is_gz_data)

            if data is None:
                return jsonify({'error': 'Data not found'}), 404
            return Response(data, content_type='application/json', headers={
                'Content-Encoding': 'gzip'
            })

        @self.app.route("/miner-positions/<minerid>", methods=["GET"])
        def get_miner_positions_unique(minerid):
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Use the API key's tier for access
            api_key_tier = self.get_api_key_tier(api_key)
            if api_key_tier == 100 and self.position_manager:
                existing_positions: list[Position] = self.position_manager.get_positions_for_one_hotkey(minerid,
                                                                                                        sort_positions=True)
                if not existing_positions:
                    return jsonify({'error': f'Miner ID {minerid} not found'}), 404
                filtered_data = self.position_manager.positions_to_dashboard_dict(existing_positions,
                                                                                  TimeUtil.now_in_millis())
            else:
                requested_tier = str(api_key_tier)
                f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)
                data = self._get_file(f)

                if data is None:
                    return jsonify({'error': 'Data not found'}), 404
                # Filter the data for the specified miner ID
                filtered_data = data.get(minerid, None)

            if not filtered_data:
                return jsonify({'error': f'Miner ID {minerid} not found'}), 404

            return jsonify(filtered_data)

        @self.app.route("/miner-hotkeys", methods=["GET"])
        def get_miner_hotkeys():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            if self.position_manager:
                # Use the position manager to get miner hotkeys
                miner_hotkeys = list(self.position_manager.get_miner_hotkeys_with_at_least_one_position())
            else:
                f = ValiBkpUtils.get_miner_positions_output_path()
                data = self._get_file(f)

                if data is None:
                    return jsonify({'error': 'Data not found'}), 404

                miner_hotkeys = list(data.keys())

            if len(miner_hotkeys) == 0:
                return jsonify({'error': 'No miner hotkeys found'}), 404
            else:
                return jsonify(miner_hotkeys)

        @self.app.route("/validator-checkpoint", methods=["GET"])
        def get_validator_checkpoint():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Validator checkpoint data is only available for tier 100
            if not self.can_access_tier(api_key, 100):
                return jsonify({'error': 'Validator checkpoint data requires tier 100 access'}), 403

            f = ValiBkpUtils.get_vcp_output_path()
            data = self._get_file(f)

            if data is None:
                return jsonify({'error': 'Checkpoint data not found'}), 404
            else:
                return jsonify(data)

        @self.app.route("/statistics", methods=["GET"])
        def get_validator_checkpoint_statistics():
            api_key = self._get_api_key_safe()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = ValiBkpUtils.get_miner_stats_dir()
            data = self._get_file(f)
            if data is None:
                return jsonify({'error': 'Statistics data not found'}), 404

            # Grab the optional "checkpoints" query param; default it to "true"
            show_checkpoints = request.args.get("checkpoints", "true").lower()

            # If checkpoints=false, remove the "checkpoints" key from each element in data
            if show_checkpoints == "false":
                for element in data.get("data", []):
                    element.pop("checkpoints", None)

            return jsonify(data)

        @self.app.route("/statistics/<minerid>/", methods=["GET"])
        def get_validator_checkpoint_statistics_unique(minerid):
            api_key = self._get_api_key_safe()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = ValiBkpUtils.get_miner_stats_dir()
            data = self._get_file(f)
            if data is None:
                return jsonify({'error': 'Statistics data not found'}), 404

            data_summary = data.get("data", [])
            if not data_summary:
                return jsonify({'error': 'No data found'}), 404

            # Grab the optional "checkpoints" query param; default it to "true"
            show_checkpoints = request.args.get("checkpoints", "true").lower()

            for element in data_summary:
                if element.get("hotkey", None) == minerid:
                    # If the user set checkpoints=false, remove them from this element
                    if show_checkpoints == "false":
                        element.pop("checkpoints", None)
                    return jsonify(element)

            return jsonify({'error': f'Miner ID {minerid} not found'}), 404

        @self.app.route("/eliminations", methods=["GET"])
        def get_eliminations():
            api_key = self._get_api_key_safe()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = ValiBkpUtils.get_eliminations_dir()
            data = self._get_file(f)

            if data is None:
                return jsonify({'error': 'Eliminations data not found'}), 404
            else:
                return jsonify(data)

        @self.app.route("/collateral/deposit", methods=["POST"])
        def deposit_collateral():
            """Process collateral deposit with encoded extrinsic."""
            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503
                
            try:
                # Parse JSON request
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400
                    
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Invalid JSON body'}), 400
                    
                # Validate required fields
                required_fields = ['extrinsic']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                        
                # Process the deposit using raw data
                result = self.contract_manager.process_deposit_request(
                    extrinsic_hex=data['extrinsic']
                )
                
                # Return response
                return jsonify(result)
                
            except Exception as e:
                bt.logging.error(f"Error processing collateral deposit: {e}")
                return jsonify({'error': 'Internal server error processing deposit'}), 500
                
        @self.app.route("/collateral/withdraw", methods=["POST"])
        def withdraw_collateral():
            """Process collateral withdrawal request."""
            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503
                
            try:
                # Parse JSON request
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400
                    
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Invalid JSON body'}), 400
                    
                # Validate required fields for signed withdrawal
                required_fields = ['amount', 'miner_coldkey', 'miner_hotkey', 'nonce', 'timestamp', 'signature']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                # Verify nonce
                is_valid, error_msg = self.nonce_manager.is_valid_request(
                    address=data['miner_hotkey'],
                    nonce=data['nonce'],
                    timestamp=data['timestamp']
                )
                if not is_valid:
                    return jsonify({'error': f'{error_msg}'}), 401

                # Verify the withdrawal signature
                keypair = Keypair(ss58_address=data['miner_coldkey'])
                message = json.dumps({
                    "amount": data['amount'],
                    "miner_coldkey": data['miner_coldkey'],
                    "miner_hotkey": data['miner_hotkey'],
                    "nonce": data['nonce'],
                    "timestamp": data['timestamp']
                }, sort_keys=True).encode('utf-8')
                is_valid = keypair.verify(message, bytes.fromhex(data['signature']))
                if not is_valid:
                    return jsonify({'error': 'Invalid signature. Withdrawal request unauthorized'}), 401

                # Verify coldkey-hotkey ownership using subtensor
                owns_hotkey = self._verify_coldkey_owns_hotkey(data['miner_coldkey'], data['miner_hotkey'])
                if not owns_hotkey:
                    return jsonify({'error': 'Coldkey does not own the specified hotkey'}), 403
                        
                # Process the withdrawal using verified data
                result = self.contract_manager.process_withdrawal_request(
                    amount=data['amount'],
                    miner_coldkey=data['miner_coldkey'],
                    miner_hotkey=data['miner_hotkey']
                )
                
                # Return response
                return jsonify(result)
                
            except Exception as e:
                bt.logging.error(f"Error processing collateral withdrawal: {e}")
                return jsonify({'error': 'Internal server error processing withdrawal'}), 500
                
        @self.app.route("/collateral/balance/<miner_address>", methods=["GET"])
        def get_collateral_balance(miner_address):
            """Get a miner's collateral balance."""
            # Check if contract manager is available
            if not self.contract_manager:
                return jsonify({'error': 'Collateral operations not available'}), 503
                
            try:
                # Get the balance
                balance = self.contract_manager.get_miner_collateral_balance(miner_address)
                
                if balance is None:
                    return jsonify({'error': 'Failed to retrieve collateral balance'}), 500
                    
                return jsonify({
                    'miner_address': miner_address,
                    'balance_theta': balance
                })
                
            except Exception as e:
                bt.logging.error(f"Error getting collateral balance for {miner_address}: {e}")
                return jsonify({'error': 'Internal server error retrieving balance'}), 500
                
                return jsonify(response_data)
                
            except Exception as e:
                bt.logging.error(f"Error getting collateral balance for {miner_address}: {e}")
                return jsonify({'error': 'Internal server error retrieving balance'}), 500

    def _verify_coldkey_owns_hotkey(self, coldkey_ss58: str, hotkey_ss58: str) -> bool:
        """
        Verify that a coldkey owns the specified hotkey using subtensor.

        Args:
            coldkey_ss58: The coldkey SS58 address
            hotkey_ss58: The hotkey SS58 address to verify ownership of

        Returns:
            bool: True if coldkey owns the hotkey, False otherwise
        """
        try:
            subtensor_api = self.contract_manager.collateral_manager.subtensor_api
            coldkey_owner = subtensor_api.queries.query_subtensor("Owner", None, [hotkey_ss58])

            if getattr(coldkey_owner, "value", None) is not None:
                return coldkey_owner.value == coldkey_ss58
            return False
        except Exception as e:
            bt.logging.error(f"Error verifying coldkey-hotkey ownership: {e}")
            return False

    def _get_api_key_safe(self) -> Optional[str]:
        """
        Safely get the API key from the query parameters or request headers.
        Returns None if there's any error accessing the API key.
        """
        try:
            # First try Authorization header
            auth_header = request.headers.get('Authorization')
            if auth_header:
                # Handle both "Bearer <token>" and plain token formats
                parts = auth_header.split(' ', 1)
                if len(parts) == 2 and parts[0].lower() == 'bearer':
                    return parts[1]
                else:
                    return auth_header

            # Then try query parameter
            api_key = request.args.get('api_key')
            if api_key:
                return api_key

            # Finally try JSON body if it's a JSON request
            if request.is_json:
                try:
                    data = request.get_json(force=True, silent=True)
                    if data and isinstance(data, dict) and "api_key" in data:
                        return data["api_key"]
                except Exception:
                    pass

            return None
        except Exception as e:
            bt.logging.debug(f"Error extracting API key: {e}")
            return None

    def _get_api_key(self):
        """
        Get the API key from the query parameters or request headers.
        This is the original method kept for backward compatibility.
        """
        api_key = self._get_api_key_safe()
        if api_key is None:
            # Log when no API key is found
            bt.logging.debug(f"No API key found in request to {request.path}")
        return api_key

    def _get_file(self, f, attempts=3, binary=False):
        """Read file with multiple attempts and return its contents."""
        file_path = os.path.abspath(os.path.join(self.data_path, f))
        if not os.path.exists(file_path):
            return None

        for attempt_number in range(attempts):
            try:
                if binary:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                else:
                    if file_path.endswith('.gz'):
                        with gzip.open(file_path, 'rt', encoding='utf-8') as fh:
                            data = json.load(fh)
                    else:
                        with open(file_path, "r") as file:
                            data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                if attempt_number == attempts - 1:
                    bt.logging.error(f"Failed to decode JSON after {attempts} attempts: {file_path}")
                    raise
                else:
                    bt.logging.debug(
                        f"Attempt {attempt_number + 1} failed with JSONDecodeError {e}, retrying..."
                    )
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                bt.logging.error(f"Unexpected error reading file {file_path}: {type(e).__name__}: {str(e)}")
                raise

    def run(self):
        """Start the REST server using Waitress."""
        print(f"[{current_process().name}] Starting REST server at http://{self.host}:{self.port}")
        setproctitle(f"vali_{self.__class__.__name__}")
        serve(self.app, host=self.host, port=self.port)


# This allows the module to be run directly for testing
if __name__ == "__main__":
    import argparse

    bt.logging.enable_info()

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the REST API server with API key authentication")
    parser.add_argument("--api-keys", type=str, default="api_keys.json", help="Path to API keys JSON file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=48888, help="Port to bind the server to")

    args = parser.parse_args()

    # Create test API keys file if it doesn't exist
    if not os.path.exists(args.api_keys):
        with open(args.api_keys, "w") as f:
            json.dump({"test_user": "test_key", "client": "abc"}, f)
        print(f"Created test API keys file at {args.api_keys}")

    # Create and run the server
    server = PTNRestServer(
        api_keys_file=args.api_keys,
        host=args.host,
        port=args.port,
        metrics_interval_minutes=1
    )
    server.run()
