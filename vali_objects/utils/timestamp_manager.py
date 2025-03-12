import threading

from shared_objects.cache_controller import CacheController
from shared_objects.rate_limiter import RateLimiter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

class TimestampManager(CacheController):
    def __init__(self, metagraph=None, hotkey=None, running_unit_tests=False):
        super().__init__(metagraph=metagraph, running_unit_tests=running_unit_tests)
        self.hotkey = hotkey
        self.last_received_order_time_ms = 0
        self.timestamp_write_rate_limiter = RateLimiter(max_requests_per_window=1,
                                                        rate_limit_window_duration_seconds=60 * 60)
        self.timestamp_lock = threading.Lock()

    def update_timestamp(self, t_ms: int):
        """
        keep track of most recent order timestamp
        write timestamp to file periodically so that timestamp is preserved on a reboot
        """
        with self.timestamp_lock:
            self.last_received_order_time_ms = max(self.last_received_order_time_ms, t_ms)
            allowed, wait_time = self.timestamp_write_rate_limiter.is_allowed(self.hotkey)
            if allowed:
                self.write_last_order_timestamp_from_memory_to_disk(self.last_received_order_time_ms)

    def get_last_order_timestamp(self) -> int:
        """
        get the timestamp of the last received order
        if we haven't received any signals, read our timestamp file to get the last order received
        """
        if self.last_received_order_time_ms == 0:
            self.last_received_order_time_ms = self.read_last_order_timestamp()
        return self.last_received_order_time_ms

    def write_last_order_timestamp_from_memory_to_disk(self, timestamp: int):
        timestamp_data = {
            "timestamp": timestamp
        }
        ValiBkpUtils.write_file(
            ValiBkpUtils.get_last_order_timestamp_file_location(
                running_unit_tests=self.running_unit_tests
            ),
            timestamp_data
        )

    def read_last_order_timestamp(self) -> int:
        return ValiUtils.get_vali_json_file_dict(
            ValiBkpUtils.get_last_order_timestamp_file_location(running_unit_tests=self.running_unit_tests)
        ).get("timestamp", -1)
