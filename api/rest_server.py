from flask import Flask, jsonify, request, Response
import os
import time
import json
from waitress import serve
from flask_compress import Compress


class RestServer:
    """Handles REST API requests with Flask and Waitress."""

    def __init__(self, api_manager, data_path="../proprietary-trading-network/validation/", host="127.0.0.1",
                 port=48888):
        """Initialize the REST server with API manager for authentication."""
        self.api_manager = api_manager
        self.data_path = data_path
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        # Initialize Flask-Compress for GZIP compression
        Compress(self.app)

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register all API routes."""

        @self.app.route("/miner-positions", methods=["GET"])
        def get_miner_positions():
            api_key = self._get_api_key()

            # Check if the API key is valid
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            # Get the 'tier' query parameter from the request
            tier = request.args.get('tier')
            is_gz_data = tier is not None

            if is_gz_data:
                # Validate the 'tier' parameter
                if tier not in ['0', '30', '50', '100']:
                    return jsonify({'error': 'Invalid tier value. Allowed values are 0, 30, 50, or 100'}), 400

                # Construct the relative path based on the specified tier
                f = f"outputs/tiered_positions/{tier}/output.json.gz"
            else:
                # If 'tier' parameter is not provided, return the default output.json
                f = "outputs/output.json"

            # Attempt to retrieve the file
            data = self._get_file(f, binary=is_gz_data)

            if data is None:
                return f"{f} not found", 404
            if is_gz_data:
                return Response(data, content_type='application/json', headers={
                    'Content-Encoding': 'gzip'
                })
            return jsonify(data)

        @self.app.route("/miner-positions/<minerid>", methods=["GET"])
        def get_miner_positions_unique(minerid):
            api_key = self._get_api_key()

            # Check if the API key is valid
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = "outputs/output.json"
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
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = "outputs/output.json"
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
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = "../runnable/validator_checkpoint.json"
            data = self._get_file(f)

            if data is None:
                return f"{f} not found", 404
            else:
                return jsonify(data)

        @self.app.route("/statistics", methods=["GET"])
        def get_validator_checkpoint_statistics():
            api_key = self._get_api_key()
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = "../runnable/minerstatistics.json"
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
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = "../runnable/minerstatistics.json"
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
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = "eliminations.json"
            data = self._get_file(f)

            if data is None:
                return f"{f} not found", 404
            else:
                return jsonify(data)

        @self.app.route("/miner-copying", methods=["GET"])
        def get_miner_copying():
            api_key = self._get_api_key()

            # Check if the API key is valid
            if not self.api_manager.is_valid_api_key(api_key):
                return jsonify({'error': 'Unauthorized access'}), 401

            f = "miner_copying.json"
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
                    with open(file_path, "r") as file:
                        data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                if attempt_number == attempts - 1:
                    print(f"rest_server.py Failed to decode JSON after multiple attempts: {e}")
                    raise
                else:
                    print(f"rest_server.py Attempt {attempt_number + 1} failed with JSONDecodeError, retrying...")
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                print(f"rest_server.py Unexpected error reading file: {e}")
                raise

    def run(self):
        """Start the REST server using Waitress."""
        print(f"Starting REST server at http://{self.host}:{self.port}")
        serve(self.app, host=self.host, port=self.port)


# This allows the module to be run directly for testing
if __name__ == "__main__":
    # Simple APIManager mock for testing
    class MockAPIManager:
        def is_valid_api_key(self, api_key):
            return api_key == "test_key"


    # Create and run a test server
    test_server = RestServer(MockAPIManager())
    test_server.run()