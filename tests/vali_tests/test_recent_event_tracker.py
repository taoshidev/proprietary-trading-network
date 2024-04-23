import unittest
from unittest.mock import patch

from sortedcontainers import SortedList

from vali_objects.utils.recent_event_tracker import RecentEventTracker
from vali_objects.vali_dataclasses.price_source import PriceSource
from time_util.time_util import TimeUtil

class TestRecentEventTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = RecentEventTracker()

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_add_event(self, mock_time):
        # Mock current time
        mock_time.return_value = 1000000
        event = PriceSource(start_ms=1000000, open=100.0, close=105.0)
        self.tracker.add_event(event)

        # Test if event is added correctly
        self.assertEqual(len(self.tracker.events), 1)
        self.assertEqual(self.tracker.events[0][1], event)

        # Add another event and test cleanup of old events
        event2 = PriceSource(start_ms=1003000, open=102.0, close=106.0)
        mock_time.return_value = 1004000
        self.tracker.add_event(event2)
        self.tracker.add_event(PriceSource(start_ms=1004000, open=103.0, close=107.0))

        # Check if the first event is still there as it's within 5 minutes
        self.assertEqual(len(self.tracker.events), 3)
        self.tracker._cleanup_old_events()
        self.assertEqual(len(self.tracker.events), 2)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_cleanup_old_events(self, mock_time):
        mock_time.return_value = 1000000
        self.tracker.add_event(PriceSource(start_ms=1000000, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=1003000, open=102.0, close=106.0))

        # Forward time to trigger cleanup
        mock_time.return_value = 1400000
        self.tracker._cleanup_old_events()

        # First event should be removed
        self.assertEqual(len(self.tracker.events), 1)

    def test_get_events_in_range(self):
        self.tracker.events = SortedList([(1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),
                                          (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0))],
                                         key=lambda x: x[0])
        events = self.tracker.get_events_in_range(1000000, 1002000)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].open, 100.0)

    def test_get_closest_event(self):
        self.tracker.events = SortedList([(1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),
                                          (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0))],
                                         key=lambda x: x[0])
        closest_event = self.tracker.get_closest_event(1001500)
        self.assertEqual(closest_event.start_ms, 1000000)


if __name__ == '__main__':
    unittest.main()