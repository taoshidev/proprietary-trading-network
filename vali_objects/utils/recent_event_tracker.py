import threading
import time

from sortedcontainers import SortedList

from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.price_source import PriceSource

class RecentEventTracker:
    OLDEST_ALLOWED_RECORD_MS = 300000  # 5 minutes
    def __init__(self):
        self.events = SortedList(key=lambda x: x[0])  # Assuming each event is a tuple (timestamp, event_data)
        self.lock = threading.Lock()

    def add_event(self, event):
        with self.lock:
            event_time_ms = event.start_ms
            self.events.add((event_time_ms, event))
            #print(f"Added event at {TimeUtil.millis_to_formatted_date_str(event_time_ms)}")
            self._cleanup_old_events()

    def _cleanup_old_events(self):
        # Don't lock here, as this method is called from within a lock
        current_time_ms = TimeUtil.now_in_millis()
        # Remove events older than 5 minutes
        while self.events and current_time_ms - self.events[0][0] > self.OLDEST_ALLOWED_RECORD_MS:
            self.events.pop(0)

    def get_events_in_range(self, start_time_ms, end_time_ms):
        """
            Get all events that have timestamps between start_time_ms and end_time_ms, inclusive.

            Args:
            start_time_ms (int): The start timestamp in milliseconds.
            end_time_ms (int): The end timestamp in milliseconds.

            Returns:
            list: A list of events (event_data) within the specified time range.
            """
        with self.lock:
            if self.count_events() == 0:
                return []
            # Find the index of the first event greater than or equal to start_time_ms
            start_idx = self.events.bisect_left((start_time_ms,))
            # Find the index of the first event strictly greater than end_time_ms
            end_idx = self.events.bisect_right((end_time_ms + 1,))  # to include events at end_time_ms
            # Retrieve all events within the range [start_idx, end_idx)
            return [event[1] for event in self.events[start_idx:end_idx]]

    def get_closest_event(self, timestamp_ms) -> PriceSource or None:
        with self.lock:
            #print(f"Looking for event at {TimeUtil.millis_to_formatted_date_str(timestamp_ms)}")
            if self.count_events() == 0:
                return None
            # Find the event closest to the given timestamp
            idx = self.events.bisect_left((timestamp_ms,))
            if idx == 0:
                return self.events[0][1]
            elif idx == len(self.events):
                return self.events[-1][1]
            else:
                before = self.events[idx - 1]
                after = self.events[idx]
                return after[1] if (after[0] - timestamp_ms) < (timestamp_ms - before[0]) else before[1]

    def count_events(self):
        # Return the number of events currently stored
        return len(self.events)