# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from datetime import datetime, timedelta, timezone
from typing import List, Tuple


class TimeUtil:

    @staticmethod
    def generate_range_timestamps(start_date: datetime, end_date_days: int, print_timestamps=False) -> List[Tuple[datetime, datetime]]:
        end_date = start_date + timedelta(days=end_date_days)

        timestamps = []

        current_date = start_date
        while current_date <= end_date:
            start_timestamp = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_timestamp = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            if end_timestamp > end_date:
                end_timestamp = end_date
            timestamps.append((start_timestamp.replace(tzinfo=timezone.utc), end_timestamp.replace(tzinfo=timezone.utc)))
            current_date += timedelta(days=1)

        if print_timestamps:
            print(timestamps)

        return timestamps

    @staticmethod
    def generate_start_timestamp(days: int) -> datetime:
        return datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=days)

    @staticmethod
    def convert_range_timestamps_to_millis(timestamps: List[Tuple[datetime, datetime]]) -> List[Tuple[int, int]]:
        return [(int(row[0].timestamp() * 1000), int(row[1].timestamp() * 1000)) for row in timestamps]

    @staticmethod
    def now_in_millis() -> int:
        return int(datetime.utcnow().replace(tzinfo=timezone.utc).timestamp() * 1000)

    @staticmethod
    def timestamp_to_millis(dt) -> int:
        return int(dt.timestamp() * 1000)

    @staticmethod
    def seconds_to_timestamp(seconds: int) -> datetime:
        return datetime.utcfromtimestamp(seconds).replace(tzinfo=timezone.utc)

    @staticmethod
    def millis_to_timestamp(millis: int) -> datetime:
        return datetime.utcfromtimestamp(millis / 1000).replace(tzinfo=timezone.utc)

    @staticmethod
    def minute_in_millis(minutes: int) -> int:
        return minutes * 60000

    @staticmethod
    def hours_in_millis(hours: int = 24) -> int:
        # standard is 1 day
        return 60000 * 60 * hours * 1 * 1