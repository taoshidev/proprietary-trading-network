import asyncio
import websockets
import json
import random
import time
import sys


class WebSocketClient:
    """Client for connecting to WebSocket server with authentication and retry logic."""

    def __init__(self, uri="ws://localhost:8765", api_key=None,
                 initial_backoff=1, max_backoff=60, jitter=0.1):
        """Initialize the WebSocket client with connection parameters and retry settings."""
        self.uri = uri
        self.api_key = api_key
        self.connected = False
        self.websocket = None

        # Exponential backoff parameters
        self.initial_backoff = initial_backoff  # Initial backoff in seconds
        self.max_backoff = max_backoff  # Maximum backoff in seconds
        self.jitter = jitter  # Random jitter factor (0-1)

        # Message handlers
        self.message_handlers = []

        # Statistics
        self.connect_attempts = 0
        self.messages_received = 0

    def add_message_handler(self, handler):
        """Add a message handler function that will be called for each received message."""
        self.message_handlers.append(handler)

    async def connect(self):
        """Connect to the WebSocket server with retry logic."""
        self.connect_attempts = 0
        backoff = self.initial_backoff

        while True:
            try:
                self.connect_attempts += 1
                print(f"Connecting to {self.uri} (attempt {self.connect_attempts})...")

                async with websockets.connect(self.uri) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    print(f"Connected to {self.uri}")

                    # Authenticate
                    await self._authenticate()

                    # Process messages
                    await self._process_messages()

            except (websockets.exceptions.ConnectionClosed,
                    websockets.exceptions.InvalidStatusCode,
                    ConnectionRefusedError, OSError) as e:
                self.connected = False
                self.websocket = None

                # Apply exponential backoff with jitter
                jitter_amount = random.uniform(-self.jitter, self.jitter)
                adjusted_backoff = backoff * (1 + jitter_amount)

                print(f"Connection error: {e}")
                print(f"Reconnecting in {adjusted_backoff:.2f} seconds...")

                await asyncio.sleep(adjusted_backoff)

                # Increase backoff for next attempt (exponential)
                backoff = min(backoff * 2, self.max_backoff)

            except Exception as e:
                print(f"Unexpected error: {e}")
                self.connected = False
                self.websocket = None
                raise

    async def _authenticate(self):
        """Send authentication message with API key."""
        if not self.api_key:
            raise ValueError("API key is required for authentication")

        auth_message = json.dumps({"api_key": self.api_key})
        await self.websocket.send(auth_message)

        # Wait for auth response
        response = await self.websocket.recv()
        response_data = json.loads(response)

        if response_data.get("status") != "success":
            error_msg = response_data.get("message", "Authentication failed")
            raise Exception(f"Authentication error: {error_msg}")

        print("Authentication successful")

    async def _process_messages(self):
        """Process incoming messages and dispatch to handlers."""
        while True:
            try:
                message = await self.websocket.recv()
                self.messages_received += 1

                # Parse JSON message
                data = json.loads(message)

                # Check for error messages
                if data.get("status") == "error":
                    print(f"Server error: {data.get('message', 'Unknown error')}")
                    if "API key no longer valid" in data.get('message', ''):
                        raise Exception("API key revoked")
                    continue

                # Log message
                print(f"Received: {message}")

                # Dispatch to handlers
                for handler in self.message_handlers:
                    await handler(data)

            except (websockets.exceptions.ConnectionClosed, ConnectionResetError):
                print("Connection closed by server")
                raise
            except json.JSONDecodeError:
                print("Received invalid JSON data")
                continue
            except Exception as e:
                print(f"Error processing message: {e}")
                raise

    async def send(self, message):
        """Send a message to the server."""
        if not self.connected or not self.websocket:
            raise Exception("Not connected to server")

        if isinstance(message, dict):
            message = json.dumps(message)

        await self.websocket.send(message)

    def run(self):
        """Run the client in the current process."""
        try:
            asyncio.run(self.connect())
        except KeyboardInterrupt:
            print("Client shutting down...")
        except Exception as e:
            print(f"Fatal error in client: {e}")
            return 1
        return 0


# Example message handler
async def print_message(message):
    """Simple message handler that prints the received message."""
    # Just print, but you could do more complex processing here
    print(f"Handler received: {message}")


# This allows the module to be run directly for testing
if __name__ == "__main__":
    # Get API key from command line argument or use default
    api_key = sys.argv[1] if len(sys.argv) > 1 else "test_key"

    # Create client
    client = WebSocketClient(api_key=api_key)

    # Add message handler
    client.add_message_handler(print_message)

    # Run client
    sys.exit(client.run())