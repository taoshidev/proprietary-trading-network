import json
import os
import time
import traceback
import threading
from multiprocessing import Process, Manager
from ptn_api.rest_server import PTNRestServer
from ptn_api.websocket_server import WebSocketServer
from ptn_api.slack_notifier import SlackNotifier

from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


def start_rest_server(shared_queue, host="127.0.0.1", port=48888, refresh_interval=15, position_manager=None,
                      contract_manager=None, miner_statistics_manager=None, request_core_manager=None,
                      asset_selection_manager=None, debt_ledger_manager=None):
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
            position_manager=position_manager,
            contract_manager=contract_manager,
            miner_statistics_manager=miner_statistics_manager,
            request_core_manager=request_core_manager,
            asset_selection_manager=asset_selection_manager,
            debt_ledger_manager=debt_ledger_manager
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
        print(f"Creating WebSocketServer instance...")
        websocket_server = WebSocketServer(
            api_keys_file=api_keys_file,
            shared_queue=shared_queue,
            host=host,
            port=port,
            refresh_interval=refresh_interval
        )
        print(f"WebSocketServer instance created, calling run()...")
        websocket_server.run()
        print(f"WebSocketServer.run() returned (this shouldn't happen unless shutting down)")
    except Exception as e:
        print(f"FATAL: Exception in WebSocket server process: {type(e).__name__}: {e}")
        print(f"Full traceback:")
        print(traceback.format_exc())
        raise


class APIManager:
    """Manages API services and processes."""

    def __init__(self, shared_queue, refresh_interval=15,
                 rest_host="127.0.0.1", rest_port=48888,
                 ws_host="localhost", ws_port=8765,
                 position_manager=None, contract_manager=None,
                 miner_statistics_manager=None, request_core_manager=None,
                 asset_selection_manager=None, slack_webhook_url=None, debt_ledger_manager=None,
                 validator_hotkey=None):
        """Initialize API management with shared queue and server configurations.

        Args:
            shared_queue: Multiprocessing.Queue for WebSocket messaging (required)
            refresh_interval: How often to check for API key changes (seconds)
            rest_host: Host address for the REST API server
            rest_port: Port for the REST API server
            ws_host: Host address for the WebSocket server
            ws_port: Port for the WebSocket server
            position_manager: PositionManager instance (optional) for fast miner positions
            contract_manager: ValidatorContractManager instance (optional) for collateral operations
            slack_webhook_url: Slack webhook URL for health alerts (optional)
            validator_hotkey: Validator hotkey for identification in alerts (optional)
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
        self.contract_manager = contract_manager
        self.miner_statistics_manager = miner_statistics_manager
        self.request_core_manager = request_core_manager
        self.asset_selection_manager = asset_selection_manager
        self.debt_ledger_manager = debt_ledger_manager

        # Initialize Slack notifier
        self.slack_notifier = SlackNotifier(webhook_url=slack_webhook_url, hotkey=validator_hotkey)
        self.health_monitor_thread = None
        self.shutdown_event = threading.Event()

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

    def _health_monitor_daemon(self, rest_process, ws_process):
        """
        Daemon thread that monitors process health and sends Slack alerts.
        Runs independently of the main monitoring loop.
        """
        print("[HealthMonitor] Daemon thread started")
        ws_was_down = False
        rest_was_down = False

        while not self.shutdown_event.is_set():
            try:
                # Check WebSocket server health
                if not ws_process.is_alive():
                    if not ws_was_down:
                        print(f"[HealthMonitor] WebSocket server down! PID: {ws_process.pid}, Exit: {ws_process.exitcode}")
                        self.slack_notifier.send_websocket_down_alert(
                            pid=ws_process.pid,
                            exit_code=ws_process.exitcode,
                            host=self.ws_host,
                            port=self.ws_port
                        )
                        ws_was_down = True
                else:
                    if ws_was_down:
                        print("[HealthMonitor] WebSocket server recovered!")
                        self.slack_notifier.send_recovery_alert("WebSocket Server")
                        ws_was_down = False

                # Check REST server health
                if not rest_process.is_alive():
                    if not rest_was_down:
                        print(f"[HealthMonitor] REST server down! PID: {rest_process.pid}, Exit: {rest_process.exitcode}")
                        self.slack_notifier.send_rest_down_alert(
                            pid=rest_process.pid,
                            exit_code=rest_process.exitcode,
                            host=self.rest_host,
                            port=self.rest_port
                        )
                        rest_was_down = True
                else:
                    if rest_was_down:
                        print("[HealthMonitor] REST server recovered!")
                        self.slack_notifier.send_recovery_alert("REST Server")
                        rest_was_down = False

                # Sleep before next check
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                print(f"[HealthMonitor] Error in health check: {e}")
                time.sleep(10)

        print("[HealthMonitor] Daemon thread stopped")

    def run(self):
        """Main entry point to run REST API and WebSocket server."""
        print("Starting API services...")

        # Start REST server process with host/port configuration
        rest_process = Process(
            target=start_rest_server,
            args=(self.shared_queue, self.rest_host, self.rest_port, self.refresh_interval, self.position_manager,
                  self.contract_manager, self.miner_statistics_manager, self.request_core_manager,
                  self.asset_selection_manager, self.debt_ledger_manager),
            name="RestServer"
        )
        rest_process.start()
        print(f"REST API server process started (PID: {rest_process.pid}) at http://{self.rest_host}:{self.rest_port}")

        # Start WebSocket server process with host/port configuration
        ws_process = Process(
            target=start_websocket_server,
            args=(self.shared_queue, self.ws_host, self.ws_port, self.refresh_interval),
            name="WebSocketServer"
        )
        ws_process.start()
        print(f"WebSocket server process started (PID: {ws_process.pid}) at ws://{self.ws_host}:{self.ws_port}")

        # Start health monitor daemon thread
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_daemon,
            args=(rest_process, ws_process),
            daemon=True,
            name="HealthMonitor"
        )
        self.health_monitor_thread.start()
        print("Health monitor daemon thread started")

        # Keep main thread alive - health monitoring happens in daemon thread
        try:
            while True:
                time.sleep(60)  # Just keep alive, daemon handles all monitoring

        except KeyboardInterrupt:
            print("\nShutting down API services due to keyboard interrupt...")

            # Signal health monitor to stop
            self.shutdown_event.set()

            # Terminate processes
            if rest_process.is_alive():
                print(f"Terminating REST server process (PID: {rest_process.pid})...")
                rest_process.terminate()
            if ws_process.is_alive():
                print(f"Terminating WebSocket server process (PID: {ws_process.pid})...")
                ws_process.terminate()

            # Wait for clean shutdown
            rest_process.join(timeout=10)
            ws_process.join(timeout=10)
            print("API services shutdown complete.")


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
