import json
import os
import time
import threading
from multiprocessing import Process
from rest_server import RestServer
from websocket_server import WebSocketServer


class APIManager:
    """Manages API key authentication, refreshing, and initializing services."""

    def __init__(self, api_keys_file="api_keys.json", refresh_interval=15,
                 data_path="../proprietary-trading-network/validation/"):
        self.api_keys_file = api_keys_file
        self.refresh_interval = refresh_interval
        self.data_path = data_path
        self.accessible_api_keys = []
        self.load_api_keys()
        self.api_key_refresh_thread = threading.Thread(target=self.refresh_api_keys, daemon=True)

    def load_api_keys(self):
        """Loads API keys from a file and logs changes."""
        if not os.path.exists(self.api_keys_file):
            raise FileNotFoundError(f"API keys file '{self.api_keys_file}' not found!")
        try:
            with open(self.api_keys_file, "r") as f:
                new_keys = json.load(f)
            if not isinstance(new_keys, list):
                raise ValueError("API keys file must contain a list of keys.")
            if len(new_keys) != len(self.accessible_api_keys):
                print(f"API key list size changed: {len(self.accessible_api_keys)} -> {len(new_keys)}")
            last_modified = time.ctime(os.path.getmtime(self.api_keys_file))
            print(f"API keys file last modified at: {last_modified}")
            self.accessible_api_keys = new_keys
        except Exception as e:
            print(f"Error loading API keys: {e}")

    def refresh_api_keys(self):
        """Continuously refresh API keys every specified interval."""
        while True:
            try:
                self.load_api_keys()
            except Exception as e:
                print(f"Failed to refresh API keys: {e}")
            time.sleep(self.refresh_interval)

    def is_valid_api_key(self, api_key):
        """Checks if an API key is valid."""
        return api_key in self.accessible_api_keys

    def start_rest_server(self):
        """Starts the REST API as a separate process."""
        rest_server = RestServer(self, self.data_path)
        rest_server.run()

    def start_websocket_server(self):
        """Starts the WebSocket server as a separate process."""
        websocket_server = WebSocketServer(self)
        websocket_server.run()

    def run(self):
        """Main entry point to run API key management, REST API, and WebSocket server."""
        print("Starting API Manager...")
        # Start API key refresh thread
        self.api_key_refresh_thread.start()

        # Start REST server process
        rest_process = Process(target=self.start_rest_server)
        rest_process.start()
        print("REST API server started in a separate process")

        # Start WebSocket server process
        ws_process = Process(target=self.start_websocket_server)
        ws_process.start()
        print("WebSocket server started in a separate process")

        # Keep the main process running to manage the child processes
        try:
            # Wait for processes to complete (which they won't unless interrupted)
            rest_process.join()
            ws_process.join()
        except KeyboardInterrupt:
            print("Shutting down API Manager...")
            # Clean up processes on keyboard interrupt
            rest_process.terminate()
            ws_process.terminate()
            rest_process.join()
            ws_process.join()


if __name__ == "__main__":
    api_manager = APIManager()
    api_manager.run()