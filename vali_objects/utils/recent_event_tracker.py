import time

from sortedcontainers import SortedList

from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.price_source import PriceSource

class RecentEventTracker:
    OLDEST_ALLOWED_RECORD_MS = 300000  # 5 minutes
    def __init__(self):
        self.events = SortedList(key=lambda x: x[0])  # Assuming each event is a tuple (timestamp, event_data)

    def add_event(self, event):
        event_time_ms = event.start_ms
        self.events.add((event_time_ms, event))
        #print(f"Added event at {TimeUtil.millis_to_formatted_date_str(event_time_ms)}")
        self.cleanup_old_events()

    def cleanup_old_events(self):
        current_time_ms = TimeUtil.now_in_millis()
        # Remove events older than 5 minutes
        while self.events and current_time_ms - self.events[0][0] > self.OLDEST_ALLOWED_RECORD_MS:
            self.events.pop(0)

    def get_closest_event(self, timestamp_ms) -> PriceSource or None:
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