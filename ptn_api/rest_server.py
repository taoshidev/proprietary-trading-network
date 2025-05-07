from flask import Flask, jsonify, request, Response
import os
import time
import json
import gzip
import traceback
from setproctitle import setproctitle
from waitress import serve
from flask_compress import Compress

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_config import ValiConfig
from multiprocessing import current_process
from ptn_api.api_key_refresh import APIKeyMixin


class PTNRestServer(APIKeyMixin):
    """Handles REST API requests with Flask and Waitress."""

    def __init__(self, api_keys_file, shared_queue=None, host="127.0.0.1",
                 port=48888, refresh_interval=15):
        """Initialize the REST server with API key handling and routing.

        Args:
            api_keys_file: Path to the JSON file containing API keys
            shared_queue: Optional shared queue for communication with WebSocket server
            host: Hostname or IP to bind the server to
            port: Port to bind the server to
            refresh_interval: How often to check for API key changes (seconds)
        """
        # Initialize API key handling
        APIKeyMixin.__init__(self, api_keys_file, refresh_interval)

        # REST server configuration
        self.shared_queue = shared_queue
        self.data_path = ValiConfig.BASE_DIR
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        # Initialize Flask-Compress for GZIP compression
        Compress(self.app)

        # Register routes
        self._register_routes()

        # Start API key refresh thread
        self.start_refresh_thread()

        print(f"[{current_process().name}] RestServer initialized with {len(self.accessible_api_keys)} API keys")

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
            requested_tier = str(api_key_tier)

            f = ValiBkpUtils.get_miner_positions_output_path(suffix_dir=requested_tier)
            data = self._get_file(f)

            if data is None:
                return f"{f} not found", 404

            # Filter the data for the specified miner ID
            filtered_data = data.get(minerid, None)

            if filtered_data is None:
                return jsonify({'error': 'Miner ID not found'}), 404

            return jsonify(filtered_data)

        @self.app.route("/miner-hotkeys", methods=["GET"])
        def get_miner_hotkeys():
            api_key = self._get_api_key()

            # Check if the API key is valid
            if not self.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

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
        port=args.port
    )
    server.run()