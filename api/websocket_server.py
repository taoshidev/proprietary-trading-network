import asyncio
import websockets
import json
import time
from multiprocessing import current_process


class WebSocketServer:
    """Handles WebSocket connections with authentication and message broadcasting."""

    def __init__(self, api_manager, host="localhost", port=8765,
                 reconnect_interval=3, max_reconnect_attempts=10):
        """Initialize the WebSocket server with API manager for authentication."""
        self.api_manager = api_manager
        self.host = host
        self.port = port
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.server = None

    async def handle_client(self, websocket, path):
        """Handle client connection with authentication."""
        client_id = id(websocket)
        print(f"[{current_process().name}] New client connected (ID: {client_id}), waiting for authentication...")

        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)

            if 'api_key' not in auth_data:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Authentication required. Please provide an API key."
                }))
                return

            api_key = auth_data['api_key']

            # Validate API key with API Manager
            if not self.api_manager.is_valid_api_key(api_key):
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid API key. Authentication failed."
                }))
                return

            # Send authentication success message
            await websocket.send(json.dumps({
                "status": "success",
                "message": "Authentication successful."
            }))

            print(f"[{current_process().name}] Client {client_id} authenticated successfully")

            # Send "Hello World" messages every second
            while True:
                # Check if API key is still valid (in case it was revoked)
                if not self.api_manager.is_valid_api_key(api_key):
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "API key no longer valid. Disconnecting."
                    }))
                    break

                # Send hello world message
                await websocket.send(json.dumps({
                    "message": "Hello World"
                }))

                # Wait 1 second
                await asyncio.sleep(1)

        except websockets.exceptions.ConnectionClosed:
            print(f"[{current_process().name}] Client {client_id} disconnected")
        except json.JSONDecodeError:
            print(f"[{current_process().name}] Received invalid JSON data from client {client_id}")
        except Exception as e:
            print(f"[{current_process().name}] Error handling client {client_id}: {e}")

    async def start(self):
        """Start the WebSocket server with retry logic."""
        attempts = 0
        while attempts < self.max_reconnect_attempts or self.max_reconnect_attempts <= 0:
            try:
                self.server = await websockets.serve(
                    self.handle_client,
                    self.host,
                    self.port
                )

                print(f"[{current_process().name}] WebSocket server started at ws://{self.host}:{self.port}")

                # Keep the server running indefinitely
                await asyncio.Future()

            except OSError as e:
                attempts += 1
                print(
                    f"[{current_process().name}] Failed to start WebSocket server (attempt {attempts}/{self.max_reconnect_attempts}): {e}")

                if attempts < self.max_reconnect_attempts or self.max_reconnect_attempts <= 0:
                    print(f"[{current_process().name}] Retrying in {self.reconnect_interval} seconds...")
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    print(f"[{current_process().name}] Maximum retry attempts reached. Giving up.")
                    raise
            except Exception as e:
                print(f"[{current_process().name}] Unexpected error starting WebSocket server: {e}")
                raise

    def run(self):
        """Start the server in the current process."""
        print(f"[{current_process().name}] Starting WebSocket server...")
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            print(f"[{current_process().name}] WebSocket server shutting down...")
        except Exception as e:
            print(f"[{current_process().name}] Error in WebSocket server: {e}")


def run_websocket_server(api_manager):
    """Entry point for the WebSocket server process."""
    server = WebSocketServer(api_manager)
    server.run()


# This allows the module to be run directly for testing
if __name__ == "__main__":
    # Simple APIManager mock for testing
    class MockAPIManager:
        def is_valid_api_key(self, api_key):
            return api_key == "test_key"


    # Create and run a test server
    server = WebSocketServer(MockAPIManager())
    server.run()