# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from functools import lru_cache
from zoneinfo import ZoneInfo  # Make sure to use Python 3.9 or later

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from pandas.tseries.holiday import USFederalHolidayCalendar  # noqa: E402

import pandas_market_calendars as mcal  # noqa: E402
from pandas.tseries.holiday import Holiday, nearest_workday, EasterMonday, GoodFriday  # noqa: E402
MS_IN_8_HOURS =  28800000
MS_IN_24_HOURS = 86400000


class ForexHolidayCalendar(USFederalHolidayCalendar):
    """
    Calendar for global Forex trading holidays.
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("Boxing Day", month=12, day=26, observance=nearest_workday)
    ]

    def __init__(self):
        # Call to the superclass constructor
        self.rules = ForexHolidayCalendar.rules
        super().__init__()

        # Cache to store holiday lists by year
        self.holidays_cache = {}

        # Initialize cache attributes to zero
        self.cache_valid_min_ms = 0
        self.cache_valid_max_ms = 0
        self.cache_valid_ans = False

    def get_holidays(self, timestamp):
        # Check if the holidays for the given year are already cached
        if timestamp.year not in self.holidays_cache:
            # If not cached, calculate and cache them
            start_date = pd.Timestamp(f"{timestamp.year}-01-01")
            end_date = pd.Timestamp(f"{timestamp.year}-12-31")

            self.holidays_cache[timestamp.year] = self.holidays(start=start_date, end=end_date)
        return self.holidays_cache[timestamp.year]

    def is_forex_market_open(self, ms_timestamp):
        # Check if our answer is cached.
        if self.cache_valid_min_ms <= ms_timestamp <= self.cache_valid_max_ms:
            return self.cache_valid_ans
        # Convert millisecond timestamp to pandas Timestamp in UTC
        timestamp = pd.Timestamp(ms_timestamp, unit='ms', tz='UTC')

        # Convert timestamp to New York time using zoneinfo
        ny_timezone = ZoneInfo('America/New_York')
        ny_timestamp = timestamp.astimezone(ny_timezone)

        ans = True
        # Check if the day is a weekend in New York time
        if ny_timestamp.weekday() < 4:  # Monday to Thursday
            self.cache_valid_max_ms = ny_timestamp.replace(hour=23, minute=59, second=59).timestamp() * 1000
        elif ny_timestamp.weekday() == 5:  # Saturday all day
            self.cache_valid_max_ms = ny_timestamp.replace(hour=23, minute=59, second=59).timestamp() * 1000
            ans = False
        elif ny_timestamp.weekday() == 4:
            if ny_timestamp.hour >= 17:  # Market closes at 5 PM Friday NY time
                ans = False
                self.cache_valid_max_ms = ny_timestamp.replace(hour=23, minute=59, second=59).timestamp() * 1000
            else:
                ans = True
                self.cache_valid_max_ms = ny_timestamp.replace(hour=16, minute=59, second=59).timestamp() * 1000
        elif ny_timestamp.weekday() == 6:
            if ny_timestamp.hour < 17:  # Market opens at 5 PM Sunday NY time
                ans = False
                self.cache_valid_max_ms = ny_timestamp.replace(hour=16, minute=59, second=59).timestamp() * 100
            else:
                ans = True
                self.cache_valid_max_ms = ny_timestamp.replace(hour=23, minute=59, second=59).timestamp() * 1000
        else:
            raise Exception(f"Unexpected weekday: {ny_timestamp.weekday()}")

        if ans:
            # Check if the day is a holiday (assuming holiday impacts the full day)
            # Ensure get_holidays function is aware of local dates
            holidays = self.get_holidays(ny_timestamp)
            if ny_timestamp.strftime('%Y-%m-%d') in holidays:
                ans = False
                self.cache_valid_max_ms = ny_timestamp.replace(hour=23, minute=59, second=59).timestamp() * 1000

        self.cache_valid_min_ms = ms_timestamp
        self.cache_valid_ans = ans
        return ans


class IndicesMarketCalendar:
    def __init__(self):
        # Create market calendars for NYSE, NASDAQ, and CBOE
        self.nyse_calendar = mcal.get_calendar('NYSE')
        self.nasdaq_calendar = mcal.get_calendar('NASDAQ')
        self.cboe_calendar = mcal.get_calendar('CBOE_Index_Options')  # For VIX

        # Initialize cache attributes to zero
        self.cache_valid_min_ms = 0
        self.cache_valid_max_ms = 0
        self.cache_valid_ans = False


    def get_market_calendar(self, ticker):
        # Return the appropriate calendar based on the ticker
        if ticker.upper() in ['SPX', 'DJI']:  # S&P 500 and Dow Jones are on the NYSE
            return self.nyse_calendar
        elif ticker.upper() == 'NDX':  # NASDAQ 100 is on the NASDAQ
            return self.nasdaq_calendar
        elif ticker.upper() == 'VIX':  # Volatility Index is derived from CBOE
            return self.cboe_calendar
        else:
            raise ValueError(f"Ticker not supported {ticker}. Supported tickers are: SPX, NDX, DJI, VIX")

    @lru_cache(maxsize=3000)
    def schedule_from_cache(self, tsn, market_name):
        # Normalize the timestamp to ensure cache consistency
        if market_name == 'CBOE_Index_Options':
            market_calendar = self.cboe_calendar
        elif market_name == 'NYSE':
            market_calendar = self.nyse_calendar
        elif market_name == 'NASDAQ':
            market_calendar = self.nasdaq_calendar
        else:
            raise ValueError(f"Market calendar not supported {market_name}")
        #start_date = tsn - timedelta(days=5)
        #end_date = tsn + timedelta(days=5)
        schedule = market_calendar.schedule(start_date=tsn, end_date=tsn)
        return schedule


    def is_market_open(self, ticker, timestamp_ms):
        if self.cache_valid_min_ms <= timestamp_ms <= self.cache_valid_max_ms:
            return self.cache_valid_ans
        # Convert millisecond timestamp to pandas Timestamp in UTC
        timestamp = pd.Timestamp(timestamp_ms, unit='ms', tz='UTC')

        if ticker in ['GDAXI', 'FTSE']:
            return False

        # Get the market calendar for the given ticker
        market_calendar = self.get_market_calendar(ticker)

        # Calculate the start and end dates for the schedule
        tsn = timestamp.normalize()
        schedule = self.schedule_from_cache(tsn, market_calendar.name)

        if schedule.empty:
            self.cache_valid_ans = False
            self.cache_valid_min_ms = timestamp_ms
            self.cache_valid_max_ms = self.cache_valid_min_ms + MS_IN_24_HOURS
            return self.cache_valid_ans
        #print('schedule', schedule, 'ts', timestamp, 'ts_m', timestamp_ms)

        start_time_ms = TimeUtil.timestamp_to_millis(schedule.iloc[0]['market_open'])
        end_time_ms = TimeUtil.timestamp_to_millis(schedule.iloc[0]['market_close'])
        if timestamp_ms < start_time_ms:
            self.cache_valid_ans = False
            self.cache_valid_min_ms = timestamp_ms
            self.cache_valid_max_ms = start_time_ms - 1
        elif timestamp_ms < end_time_ms:
            self.cache_valid_ans = True
            self.cache_valid_min_ms = timestamp_ms
            self.cache_valid_max_ms = end_time_ms - 1
        else:
            self.cache_valid_ans = False
            self.cache_valid_min_ms = timestamp_ms
            self.cache_valid_max_ms = TimeUtil.timestamp_to_millis(tsn) + MS_IN_24_HOURS

        # Check if the timestamp is within trading hours
        #market_open = market_calendar.open_at_time(schedule, timestamp, include_close=False)
        return self.cache_valid_ans



class UnifiedMarketCalendar:
    def __init__(self):
        # Initialize both market calendars
        self.indices_calendar = IndicesMarketCalendar()
        self.forex_calendar = ForexHolidayCalendar()

    def is_market_open(self, trade_pair, timestamp_ms:int):
        #t0 = time.time()
        if not trade_pair:
            raise ValueError("Trade pair is required")
        if trade_pair.is_crypto:
            # Crypto markets are assumed to be always open
            return True
        elif trade_pair.is_forex:
            ans = self.forex_calendar.is_forex_market_open(timestamp_ms)
            #tf = time.time()
            #print(f"found forex {trade_pair.trade_pair_id} in {tf - t0}")
            # Check if the Forex market is open using the Forex calendar
            return ans
        elif trade_pair.is_indices:
            ticker = trade_pair.trade_pair_id  # Use the trade_pair_id as the ticker
            ans = self.indices_calendar.is_market_open(ticker, timestamp_ms)
            #tf = time.time()
            #print(f"found index {ticker} in {tf - t0}")
            # Check if the index market is open using the indices calendar
            return ans
        else:
            raise ValueError("Unsupported trade pair category")

class TimeUtil:

    @staticmethod
    def generate_range_timestamps(start_date: datetime, end_date_days: int, print_timestamps=False) -> List[
        Tuple[datetime, datetime]]:
        end_date = start_date + timedelta(days=end_date_days)

        timestamps = []

        current_date = start_date
        while current_date <= end_date:
            start_timestamp = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_timestamp = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            if end_timestamp > end_date:
                end_timestamp = end_date
            timestamps.append(
                (start_timestamp.replace(tzinfo=timezone.utc), end_timestamp.replace(tzinfo=timezone.utc)))
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
    def millis_to_datetime(millis: int) -> datetime:
        """
        Convert a timestamp in milliseconds to a datetime object in UTC.

        Parameters:
        - millis: An integer representing a timestamp in milliseconds.

        Returns:
        - A datetime object representing the timestamp in UTC.
        """
        # Convert milliseconds to seconds
        seconds = millis / 1000.0
        # Convert seconds to a datetime object, and make it timezone-aware in UTC
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    @staticmethod
    def millis_to_formatted_date_str(millis: int) -> str:
        temp = TimeUtil.millis_to_datetime(millis)
        return temp.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def millis_to_verbose_formatted_date_str(millis: int) -> str:
        temp = TimeUtil.millis_to_datetime(millis)
        # Include milliseconds in the format. The `[:-3]` trims the microseconds to milliseconds.
        return temp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    @staticmethod
    def formatted_date_str_to_millis(date_string: str) -> int:
        date_format = '%Y-%m-%d %H:%M:%S'
        # Parse the string as a UTC datetime object
        datetime_obj = datetime.strptime(date_string, date_format)
        # Assume the datetime object is in UTC
        datetime_obj = datetime_obj.replace(tzinfo=timezone.utc)
        # Convert the UTC datetime object to a timestamp in seconds
        timestamp_seconds = datetime_obj.timestamp()
        # Convert the timestamp to milliseconds
        timestamp_milliseconds = int(timestamp_seconds * 1000)
        return timestamp_milliseconds

    @staticmethod
    def timestamp_ms_to_eastern_time_str(timestamp_ms):
        # Convert milliseconds to seconds
        timestamp_s = timestamp_ms / 1000.0

        # Create a datetime object in UTC
        utc_datetime = datetime.utcfromtimestamp(timestamp_s)

        # Manually define the Eastern Standard Time offset (-5 hours from UTC)
        EST_OFFSET = timedelta(hours=-5)

        # Adjust from UTC to Eastern Time
        eastern_datetime = utc_datetime + EST_OFFSET

        # Format the datetime object to include the day of the week
        formatted_date_string = eastern_datetime.strftime('%A, %Y-%m-%d %H:%M:%S EST')
        return formatted_date_string

    @staticmethod
    def timestamp_to_millis(dt) -> int:
        return int(dt.timestamp() * 1000)

    @staticmethod
    def seconds_to_timestamp(seconds: int) -> datetime:
        return datetime.utcfromtimestamp(seconds).replace(tzinfo=timezone.utc)

    @staticmethod
    def millis_to_timestamp(millis: int, tzone=timezone.utc, change_timezone=True) -> datetime:
        if change_timezone:
            return datetime.utcfromtimestamp(millis / 1000).replace(tzinfo=tzone)
        else:
            return datetime.utcfromtimestamp(millis / 1000)

    @staticmethod
    def minute_in_millis(minutes: int) -> int:
        return minutes * 60000

    @staticmethod
    def hours_in_millis(hours: int = 24) -> int:
        # standard is 1 day
        return 60000 * 60 * hours * 1 * 1


    @staticmethod
    def ms_at_start_of_day(dt: datetime) -> int:
        return int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)

    @staticmethod
    def delta_ms_to_next_crypto_interval(t_ms: int):
        # Convert the timestamp to a datetime object in UTC
        dt = TimeUtil.millis_to_timestamp(t_ms, change_timezone=False)

        # Get the current hour
        hour = dt.hour

        # Calculate the start of the next day (UTC)
        if hour < 4:
            next_interval = dt.replace(hour=4, minute=0, second=0, microsecond=0)
        elif hour < 12:
            next_interval = dt.replace(hour=12, minute=0, second=0, microsecond=0)
        elif hour < 20:
            next_interval = dt.replace(hour=20, minute=0, second=0, microsecond=0)
        elif hour < 24:
            temp = dt + timedelta(days=1)
            next_interval = temp.replace(hour=4, minute=0, second=0, microsecond=0)
        else:
            raise Exception(f'Unexpected hour: {hour}')

        # Calculate the difference in milliseconds
        delta_ms = (next_interval - dt).total_seconds() * 1000
        return int(delta_ms)

    @staticmethod
    def n_intervals_elapsed_crypto(start_ms:int, current_time_ms:int) -> Tuple[int, int]:
        elapsed_ms = current_time_ms - start_ms
        #print(f'Start time {TimeUtil.millis_to_formatted_date_str(start_ms)} end time {TimeUtil.millis_to_formatted_date_str(current_time_ms)}')

        n_intervals = elapsed_ms // MS_IN_8_HOURS
        remainder_ms = elapsed_ms % MS_IN_8_HOURS
        #print(f'n_intervals {n_intervals} remainder_ms {remainder_ms}')
        delta_ms_to_first_interval = TimeUtil.delta_ms_to_next_crypto_interval(start_ms)
        if remainder_ms >= delta_ms_to_first_interval:
            n_intervals += 1

        return n_intervals, TimeUtil.delta_ms_to_next_crypto_interval(current_time_ms)

    @staticmethod
    def delta_ms_to_next_forex_indices_interval(t_ms: int):
        # Convert the timestamp to a datetime object in UTC
        dt = TimeUtil.millis_to_timestamp(t_ms, change_timezone=False)

        # Get the current hour
        hour = dt.hour

        # Calculate the start of the next day (UTC)
        if hour < 21:
            next_interval = dt.replace(hour=21, minute=0, second=0, microsecond=0)
        else:
            temp = dt + timedelta(days=1)
            next_interval = temp.replace(hour=21, minute=0, second=0, microsecond=0)

        # Calculate the difference in milliseconds
        delta_ms = (next_interval - dt).total_seconds() * 1000
        return int(delta_ms)

    @staticmethod
    def n_intervals_elapsed_forex_indices(start_ms: int, current_time_ms: int) -> Tuple[int, int]:
        elapsed_ms = current_time_ms - start_ms

        n_intervals = elapsed_ms // MS_IN_24_HOURS
        remainder_ms = elapsed_ms % MS_IN_24_HOURS

        delta_ms_to_first_interval = TimeUtil.delta_ms_to_next_forex_indices_interval(start_ms)
        if remainder_ms >= delta_ms_to_first_interval:
            n_intervals += 1

        return n_intervals, TimeUtil.delta_ms_to_next_forex_indices_interval(current_time_ms)

    @staticmethod
    def get_day_of_week_from_timestamp(ms_timestamp: int) -> int:
        # Convert the milliseconds timestamp to a datetime object in UTC
        dt = datetime.fromtimestamp(ms_timestamp / 1000, tz=timezone.utc)

        # Get the day of the week (0 = Monday, ..., 6 = Sunday)
        day_of_week = dt.weekday()

        return day_of_week