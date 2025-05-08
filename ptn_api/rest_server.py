import statistics
import bittensor as bt
import threading
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple

from flask import Flask, jsonify, request, Response, g
import os
import time
import json
import gzip
import traceback
from setproctitle import setproctitle
from waitress import serve
from flask_compress import Compress

from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig
from multiprocessing import current_process
from ptn_api.api_key_refresh import APIKeyMixin


class APIMetricsTracker:
    """
    Tracks API usage metrics and logs them periodically.
    Uses a rolling time window approach to track:
    1. API key usage counts
    2. Endpoint performance metrics (request count, avg processing time)
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

        # Lock for thread safety
        self.metrics_lock = threading.Lock()

        # Reference to API key to user ID mapping
        self.api_key_to_alias = api_key_mapping or {}  # Use provided mapping or empty dict

        # Start logging thread
        self.start_logging_thread()

    def track_request(self, api_key: str, endpoint: str, duration: float):
        """
        Track a request with its associated API key, endpoint, and duration.

        Args:
            api_key: The API key used for the request
            endpoint: The endpoint that was accessed
            duration: Request processing time in seconds
        """
        # Get user_id from api_key if available
        user_id = self.api_key_to_alias.get(api_key, "unknown")

        now = time.time()

        with self.metrics_lock:
            self.api_key_hits[user_id].append(now)
            self.endpoint_hits[endpoint].append((now, duration))

    def log_metrics(self):
        """Log the current metrics based on the rolling time window."""
        current_time = time.time()
        cutoff_time = current_time - self.log_interval_seconds

        # Process metrics with lock to ensure thread safety
        api_counts = {}
        endpoint_stats = {}
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

        # Skip logging if there's no activity
        if not api_counts and not endpoint_stats:
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
                 port=48888, refresh_interval=15, metrics_interval_minutes=5, position_manager=None):
        """Initialize the REST server with API key handling and routing.

        Args:
            api_keys_file: Path to the JSON file containing API keys
            shared_queue: Optional shared queue for communication with WebSocket server
            host: Hostname or IP to bind the server to
            port: Port to bind the server to
            refresh_interval: How often to check for API key changes (seconds)
            metrics_interval_minutes: How often to log API metrics (minutes)
            position_manager: Optional position manager for handling miner positions
        """
        # Initialize API key handling
        APIKeyMixin.__init__(self, api_keys_file, refresh_interval)

        # REST server configuration
        self.shared_queue = shared_queue
        self.position_manager: PositionManager = position_manager
        self.data_path = ValiConfig.BASE_DIR
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        # Initialize Flask-Compress for GZIP compression
        Compress(self.app)

        # Initialize metrics tracking
        self._setup_metrics(metrics_interval_minutes)

        # Register routes
        self._register_routes()

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

            # Get API key
            api_key = self._get_api_key()

            # Get endpoint (use rule if available, otherwise path)
            url = request.url_rule.rule if request.url_rule else request.path

            # Track the request using the instance metrics tracker
            self.metrics.track_request(api_key, url, duration)

            return response

    def _register_routes(self):
        """Register all API routes."""

        @self.app.route("/miner-positions", methods=["GET"])
        def get_miner_positions():
            api_key = self._get_api_key()

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
                return f"{f} not found", 404
            return Response(data, content_type='application/json', headers={
                'Content-Encoding': 'gzip'
            })

        @self.app.route("/miner-positions/<minerid>", methods=["GET"])
        def get_miner_positions_unique(minerid):
            api_key = self._get_api_key()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Use the API key's tier for access
            api_key_tier = self.get_api_key_tier(api_key)
            if api_key_tier == 100 and self.position_manager:
                existing_positions: list[Position] = self.position_manager.get_positions_for_one_hotkey(minerid, sort_positions=True)
                if not existing_positions:
                    return jsonify({'error': f'Miner ID {minerid} not found in position manager'}), 404
                filtered_data = self.position_manager.positions_to_dashboard_dict(existing_positions, TimeUtil.now_in_millis())
            else:
                requested_tier = str(api_key_tier)
                f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)
                data = self._get_file(f)

                if data is None:
                    return f"{f} not found", 404
                # Filter the data for the specified miner ID
                filtered_data = data.get(minerid, None)

            if not filtered_data:
                return jsonify({'error': f'Miner ID {minerid} not found'}), 404

            return jsonify(filtered_data)

        @self.app.route("/miner-hotkeys", methods=["GET"])
        def get_miner_hotkeys():
            api_key = self._get_api_key()

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
                    return f"{f} not found", 404

                miner_hotkeys = list(data.keys())

            if len(miner_hotkeys) == 0:
                return f"{f} not found", 404
            else:
                return jsonify(miner_hotkeys)

        @self.app.route("/validator-checkpoint", methods=["GET"])
        def get_validator_checkpoint():
            api_key = self._get_api_key()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Validator checkpoint data is only available for tier 100
            if not self.can_access_tier(api_key, 100):
                return jsonify({'error': 'Validator checkpoint data requires tier 100 access'}), 403

            f = ValiBkpUtils.get_vcp_output_path()
            data = self._get_file(f)

            if data is None:
                return f"{f} not found", 404
            else:
                return jsonify(data)

        @self.app.route("/statistics", methods=["GET"])
        def get_validator_checkpoint_statistics():
            api_key = self._get_api_key()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = ValiBkpUtils.get_miner_stats_dir()
            data = self._get_file(f)
            if data is None:
                return f"{f} not found", 404

            # Grab the optional "checkpoints" query param; default it to "true"
            show_checkpoints = request.args.get("checkpoints", "true").lower()

            # If checkpoints=false, remove the "checkpoints" key from each element in data
            if show_checkpoints == "false":
                for element in data.get("data", []):
                    element.pop("checkpoints", None)

            return jsonify(data)

        @self.app.route("/statistics/<minerid>/", methods=["GET"])
        def get_validator_checkpoint_statistics_unique(minerid):
            api_key = self._get_api_key()
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = ValiBkpUtils.get_miner_stats_dir()
            data = self._get_file(f)
            if data is None:
                return f"{f} not found", 404

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

            return jsonify({'error': 'Miner ID not found'}), 404

        @self.app.route("/eliminations", methods=["GET"])
        def get_eliminations():
            api_key = self._get_api_key()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = ValiBkpUtils.get_eliminations_dir()
            data = self._get_file(f)

            if data is None:
                return f"{f} not found", 404
            else:
                return jsonify(data)

    def _get_api_key(self):
        """Get the API key from the query parameters or request headers."""
        if request.is_json and "api_key" in request.json:
            api_key = request.json["api_key"]
        else:
            api_key = request.headers.get('Authorization')
            if api_key:
                api_key = api_key.split(' ')[1]  # Remove 'Bearer ' prefix
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
                    print(f"[{current_process().name}] Failed to decode JSON after multiple attempts: {e}")
                    raise
                else:
                    print(
                        f"[{current_process().name}] Attempt {attempt_number + 1} failed with JSONDecodeError, retrying...")
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                print(f"[{current_process().name}] Unexpected error reading file {file_path}: {e}")
                traceback.print_exc()
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