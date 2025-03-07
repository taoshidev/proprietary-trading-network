import unittest
from threading import Thread
from unittest.mock import patch

from sortedcontainers import SortedList

from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker
from vali_objects.vali_dataclasses.price_source import PriceSource

class TestRecentEventTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = RecentEventTracker()

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_add_event(self, mock_time):
        # First event added
        mock_time.return_value = 10000000  # Mock time set for the first event
        event = PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0, bid=95, ask=96)
        self.tracker.add_event(event, is_forex_quote=True)
        existing_event = self.tracker.get_event_by_timestamp(mock_time.return_value)
        self.assertEqual(existing_event[0], event)
        self.assertEqual(existing_event[1], [([event.bid], [event.ask])])

        # Assert the first event is added correctly
        self.assertEqual(len(self.tracker.events), 1)
        self.assertEqual(self.tracker.events[0][1], event)

        # Add second event, update mock time first
        mock_time.return_value = 10000000 + 1000 * 60 * 4  # Update mock time for second event. 4 minutes later
        event2 = PriceSource(start_ms=mock_time.return_value, open=102.0, close=106.0)
        self.tracker.add_event(event2)

        # Assert the first two events are still there as it's within 5 minutes
        self.assertEqual(len(self.tracker.events), 2)

        mock_time.return_value = 10000000 + 1000 * 60 * 6  # Update mock time for third event. 2 minutes later
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=103.0, close=107.0))

        # Now only two events should remain after cleanup
        self.assertEqual(len(self.tracker.events), 2)

        # Get most event
        most_recent_event = self.tracker.get_closest_event(mock_time.return_value)
        self.assertEqual(most_recent_event.open, 103.0)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_cleanup_old_events(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + 3000, open=102.0, close=106.0))

        # Forward time to trigger cleanup
        mock_time.return_value += 1000 * 60 * 5  # 5 minutes later
        self.tracker._cleanup_old_events()

        # Both events still there (edge case test)
        self.assertEqual(len(self.tracker.events), 2)

        mock_time.return_value += 1
        self.tracker._cleanup_old_events()
        #First event should be removed
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

    def test_get_events_in_range2(self):
        self.tracker.events = SortedList([
            (995000, PriceSource(start_ms=995000, open=99.0, close=104.0)),  # Before range
            (1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),  # Start of range
            (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0)),  # After range start
            (1005000, PriceSource(start_ms=1005000, open=103.0, close=107.0))  # After range
        ], key=lambda x: x[0])

        # Target range partially includes some events
        events = self.tracker.get_events_in_range(1000000, 1004000)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].open, 100.0)
        self.assertEqual(events[1].open, 102.0)

        # Test for no events in range
        events_empty = self.tracker.get_events_in_range(990000, 994998)
        self.assertEqual(len(events_empty), 0, events_empty)

    def test_get_closest_event2(self):
        self.tracker.events = SortedList([
            (995000, PriceSource(start_ms=995000, open=99.0, close=104.0)),
            (1000000, PriceSource(start_ms=1000000, open=100.0, close=105.0)),
            (1001500, PriceSource(start_ms=1001500, open=101.0, close=106.0)),  # Exact match
            (1003000, PriceSource(start_ms=1003000, open=102.0, close=106.0)),
            (1004500, PriceSource(start_ms=1004500, open=103.0, close=107.0))
        ], key=lambda x: x[0])

        # Target timestamp exactly matches one event
        closest_event = self.tracker.get_closest_event(1001500)
        self.assertEqual(closest_event.start_ms, 1001500)

        closest_event_equidistant = self.tracker.get_closest_event(1002250 + 1)
        self.assertEqual(closest_event_equidistant.start_ms, 1003000, closest_event)
        closest_event_equidistant = self.tracker.get_closest_event(1002250 - 1)
        self.assertEqual(closest_event_equidistant.start_ms, 1001500, closest_event)

        # No events, should return None
        self.tracker.events = SortedList()
        no_event = self.tracker.get_closest_event(1001500)
        self.assertIsNone(no_event)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_concurrent_add_events(self, mock_time):
        def add_events(tracker, events):
            for event in events:
                tracker.add_event(event)

        mock_time.return_value = 10000000
        events1 = [PriceSource(start_ms=mock_time.return_value + i * 1000, open=100.0 + i, close=105.0 + i) for i in range(50)]
        events2 = [PriceSource(start_ms=mock_time.return_value + i * 1000, open=200.0 + i, close=205.0 + i) for i in range(50, 100)]

        thread1 = Thread(target=add_events, args=(self.tracker, events1))
        thread2 = Thread(target=add_events, args=(self.tracker, events2))

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Check for the correct number of events
        self.assertEqual(len(self.tracker.events), 100)
        # Check that events are sorted correctly
        all_events = list(self.tracker.events)
        self.assertTrue(all(e1[0] <= e2[0] for e1, e2 in zip(all_events[:-1], all_events[1:])))

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_event_overlap(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=101.0, close=106.0))  # Exact same start time
        self.assertEqual(len(self.tracker.events), 1)
        # self.assertNotEqual(self.tracker.events[0][1], self.tracker.events[1][1])
        # self.assertEqual(self.tracker.events[0][0], self.tracker.events[1][0])

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_precise_timing(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + 1, open=101.0, close=106.0))  # 1 millisecond later
        self.assertEqual(len(self.tracker.events), 2)
        self.assertNotEqual(self.tracker.events[0][1], self.tracker.events[1][1])
        self.assertTrue(self.tracker.events[1][0] - self.tracker.events[0][0] == 1)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_boundary_event_removal(self, mock_time):
        mock_time.return_value = 10000000
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value, open=100.0, close=105.0))
        self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + 299999, open=101.0, close=106.0))  # Just under 5 minutes

        mock_time.return_value = mock_time.return_value + 300000  # Exactly 5 minutes later
        self.tracker._cleanup_old_events()

        self.assertEqual(len(self.tracker.events), 2)

        mock_time.return_value = mock_time.return_value + 1
        self.tracker._cleanup_old_events()

        self.assertEqual(len(self.tracker.events), 1)

    @patch('time_util.time_util.TimeUtil.now_in_millis')
    def test_efficiency_of_cleanup(self, mock_time):
        mock_time.return_value = 10000000
        for i in range(100):
            self.tracker.add_event(PriceSource(start_ms=mock_time.return_value + i * 1000, open=100.0 + i, close=105.0 + i))

        mock_time.return_value += + 1000 * 60 * 6  # Forward time to ensure some events are old
        self.tracker._cleanup_old_events()

        # Verify that the method has cleaned up exactly the right amount of events
        self.assertTrue(len(self.tracker.events) < 100)
        self.assertTrue(all(event[0] >= 10000000 + 1000 * 60 for event in self.tracker.events))


if __name__ == '__main__':
    unittest.main()