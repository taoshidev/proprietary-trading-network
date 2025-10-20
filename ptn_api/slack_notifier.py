import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime
import bittensor as bt


class SlackNotifier:
    """Utility for sending Slack notifications with rate limiting."""

    def __init__(self, webhook_url=None, min_interval_seconds=300):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL (can also be set via SLACK_WEBHOOK_URL env var)
            min_interval_seconds: Minimum seconds between same alert type (default 5 minutes)
        """
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self.min_interval = min_interval_seconds
        self.last_alert_time = {}  # Track last alert time per alert_key

        if not self.webhook_url:
            bt.logging.warning("No Slack webhook URL configured. Notifications disabled.")

    def send_alert(self, message, alert_key=None, force=False):
        """
        Send alert to Slack with rate limiting.

        Args:
            message: Message text to send
            alert_key: Unique key for this alert type (for rate limiting)
            force: If True, bypass rate limiting

        Returns:
            bool: True if sent, False if skipped or failed
        """
        if not self.webhook_url:
            bt.logging.info(f"[Slack] Would send (no webhook configured): {message}")
            return False

        # Rate limiting
        if not force and alert_key:
            now = time.time()
            last_time = self.last_alert_time.get(alert_key, 0)
            if now - last_time < self.min_interval:
                bt.logging.debug(f"[Slack] Skipping alert '{alert_key}' (rate limited)")
                return False
            self.last_alert_time[alert_key] = now

        try:
            # Format payload
            payload = {
                "text": message,
                "username": "PTN Validator Monitor",
                "icon_emoji": ":rotating_light:"
            }

            # Send request
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    bt.logging.info(f"[Slack] Alert sent: {message[:50]}...")
                    return True
                else:
                    bt.logging.error(f"[Slack] Failed to send alert: HTTP {response.status}")
                    return False

        except urllib.error.URLError as e:
            bt.logging.error(f"[Slack] Network error sending alert: {e}")
            return False
        except Exception as e:
            bt.logging.error(f"[Slack] Error sending alert: {e}")
            return False

    def send_websocket_down_alert(self, pid, exit_code, host, port):
        """Send formatted alert for websocket server failure."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = (
            f":rotating_light: *WebSocket Server Down!*\n"
            f"*Time:* {timestamp}\n"
            f"*PID:* {pid}\n"
            f"*Exit Code:* {exit_code}\n"
            f"*Endpoint:* ws://{host}:{port}\n"
            f"*Action:* Check validator logs immediately"
        )
        return self.send_alert(message, alert_key="websocket_down")

    def send_rest_down_alert(self, pid, exit_code, host, port):
        """Send formatted alert for REST server failure."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = (
            f":rotating_light: *REST API Server Down!*\n"
            f"*Time:* {timestamp}\n"
            f"*PID:* {pid}\n"
            f"*Exit Code:* {exit_code}\n"
            f"*Endpoint:* http://{host}:{port}\n"
            f"*Action:* Check validator logs immediately"
        )
        return self.send_alert(message, alert_key="rest_down")

    def send_recovery_alert(self, service_name):
        """Send alert when service recovers."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = (
            f":white_check_mark: *{service_name} Recovered*\n"
            f"*Time:* {timestamp}\n"
            f"Service is back online"
        )
        return self.send_alert(message, alert_key=f"{service_name}_recovery", force=True)
