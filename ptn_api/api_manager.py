import json
import os
import time
import traceback
from multiprocessing import Process, Manager
from ptn_api.rest_server import PTNRestServer
from ptn_api.websocket_server import WebSocketServer

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


def start_rest_server(shared_queue, host="127.0.0.1", port=48888, refresh_interval=15, position_manager=None):
    """Starts the REST API server in a separate process."""
    try:

        # Get default API keys file path
        api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        print(f"Starting REST server process with host={host}, port={port}")

        # Create and run the REST server
        rest_server = PTNRestServer(
            api_keys_file=api_keys_file,
            shared_queue=shared_queue,
            host=host,
            port=port,
            refresh_interval=refresh_interval,
            position_manager=position_manager
        )
        rest_server.run()
    except Exception as e:
        print(f"Error in REST server process: {e}")
        print(traceback.format_exc())
        raise


def start_websocket_server(shared_queue, host="localhost", port=8765, refresh_interval=15):
    """Starts the WebSocket server in a separate process."""
    try:
        # Get default API keys file path
        api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        print(f"Starting WebSocket server process with host={host}, port={port}")

        # Create and run the WebSocket server with the shared queue
        websocket_server = WebSocketServer(
            api_keys_file=api_keys_file,
            shared_queue=shared_queue,
            host=host,
            port=port,
            refresh_interval=refresh_interval
        )
        websocket_server.run()
    except Exception as e:
        print(f"Error in WebSocket server process: {e}")
        print(traceback.format_exc())
        raise


class APIManager:
    """Manages API services and processes."""

    def __init__(self, shared_queue, refresh_interval=15,
                 rest_host="127.0.0.1", rest_port=48888,
                 ws_host="localhost", ws_port=8765, position_manager=None):
        """Initialize API management with shared queue and server configurations.

        Args:
            shared_queue: Multiprocessing.Queue for WebSocket messaging (required)
            refresh_interval: How often to check for API key changes (seconds)
            rest_host: Host address for the REST API server
            rest_port: Port for the REST API server
            ws_host: Host address for the WebSocket server
            ws_port: Port for the WebSocket server
            position_manager: PositionManager instance (optional) for fast miner positions
        """
        if shared_queue is None:
            raise ValueError("shared_queue cannot be None - a valid queue is required")

        self.shared_queue = shared_queue
        self.refresh_interval = refresh_interval

        # Server configurations
        self.rest_host = rest_host
        self.rest_port = rest_port
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.position_manager = position_manager

        # Get default API keys file path
        self.api_keys_file = ValiBkpUtils.get_api_keys_file_path()

        # Verify API keys file exists
        if not os.path.exists(self.api_keys_file):
            print(f"WARNING: API keys file '{self.api_keys_file}' not found!")
        else:
            print(f"API keys file found at: {self.api_keys_file}")
            # Check if it's a valid JSON file
            try:
                with open(self.api_keys_file, "r") as f:
                    keys = json.load(f)
                print(f"API keys file contains {len(keys)} keys")
            except Exception as e:
                print(f"ERROR reading API keys file: {e}")

    def run(self):
        """Main entry point to run REST API and WebSocket server."""
        print("Starting API services...")

        # Start REST server process with host/port configuration
        rest_process = Process(
            target=start_rest_server,
            args=(self.shared_queue, self.rest_host, self.rest_port, self.refresh_interval, self.position_manager),
            name="RestServer"
        )
        rest_process.start()
        print(f"REST API server started at http://{self.rest_host}:{self.rest_port}")

        # Start WebSocket server process with host/port configuration
        ws_process = Process(
            target=start_websocket_server,
            args=(self.shared_queue, self.ws_host, self.ws_port, self.refresh_interval),
            name="WebSocketServer"
        )
        ws_process.start()
        print(f"WebSocket server started at ws://{self.ws_host}:{self.ws_port}")

        # Keep the main process running to manage the child processes
        try:
            # Wait for processes to complete (which they won't unless interrupted)
            rest_process.join()
            ws_process.join()
        except KeyboardInterrupt:
            print("Shutting down API services...")
            # Clean up processes on keyboard interrupt
            rest_process.join()
            ws_process.join()


if __name__ == "__main__":
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the API services")
    parser.add_argument("--rest-host", type=str, default="127.0.0.1", help="Host for the REST server")
    parser.add_argument("--rest-port", type=int, default=48888, help="Port for the REST server")
    parser.add_argument("--ws-host", type=str, default="localhost", help="Host for the WebSocket server")
    parser.add_argument("--ws-port", type=int, default=8765, help="Port for the WebSocket server")

    args = parser.parse_args()

    # Create a manager for the shared queue
    mp_manager = Manager()
    shared_queue = mp_manager.Queue()

    # Create test message
    shared_queue.put({"type": "test", "message": "This is a test message", "timestamp": int(time.time() * 1000)})

    # Create and run the API manager with command-line arguments
    api_manager = APIManager(
        shared_queue=shared_queue,
        rest_host=args.rest_host,
        rest_port=args.rest_port,
        ws_host=args.ws_host,
        ws_port=args.ws_port
    )
    api_manager.run()