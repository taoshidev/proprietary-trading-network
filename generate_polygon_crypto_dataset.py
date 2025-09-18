#!/usr/bin/env python3
"""
Generate historical crypto dataset from Polygon API
- Minutely resolution OHLC data
- Only includes data when crypto market is open (24/7 for crypto)
- Backfills missing minutes with last known data
- Covers BTC/USD for last 18 months
- Output format: CSV (same structure as Tiingo version)
- Uses Polygon's 50,000 candle limit for optimal chunking
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vali_objects.utils.vali_utils import ValiUtils
from data_generator.polygon_data_service import PolygonDataService
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_objects.vali_config import TradePair


class PolygonCryptoDataGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.market_calendar = UnifiedMarketCalendar()

        # Initialize Polygon service
        self.polygon_service = PolygonDataService(api_key=api_key, disable_ws=True)

        # Trade pairs to fetch - focusing on BTC
        self.trade_pairs = {
            'BTCUSD': TradePair.BTCUSD
        }

        # Polygon candle limit
        self.candle_limit = 50000  # From polygon_data_service.py: N_CANDLES_LIMIT

        # Rate limiting
        self.requests_per_minute = 200  # Polygon has higher limits than Tiingo
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = 0

    def rate_limit(self):
        """Rate limiting for Polygon API"""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.minute_start > 60:
            self.request_count = 0
            self.minute_start = current_time

        # If we've made too many requests, wait
        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                print(f"Rate limiting: sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()

        self.request_count += 1

        # Small delay between requests
        time.sleep(0.05)

    def calculate_optimal_chunk_days(self, crypto_minutes_per_day: int = 1440) -> int:
        """
        Calculate optimal chunk size based on Polygon's 50,000 candle limit.

        Args:
            crypto_minutes_per_day: Crypto markets trade 24/7 = 1440 minutes per day

        Returns:
            Optimal chunk size in days
        """
        # Crypto markets trade 24/7: 1440 minutes per day
        # With 50,000 candle limit: 50,000 / 1440 = ~34 days max
        # Use 80% safety margin: 34 * 0.8 = ~27 days

        max_days = int((self.candle_limit * 0.8) / crypto_minutes_per_day)
        return min(27, max(1, max_days))  # Cap at 27 days, minimum 1 day

    def fetch_crypto_data(self, trade_pair: TradePair, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch crypto data from Polygon API for given date range"""

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        print(f"    Fetching {trade_pair.trade_pair_id} from {start_date.date()} to {end_date.date()}...")

        self.rate_limit()

        try:
            # Use Polygon's unified_candle_fetcher for minute data
            raw_aggs = self.polygon_service.unified_candle_fetcher(
                trade_pair=trade_pair,
                start_timestamp_ms=start_ms,
                end_timestamp_ms=end_ms,
                timespan="minute"
            )

            # Convert Polygon Agg objects to our format (matching Tiingo structure)
            data_points = []
            for agg in raw_aggs:
                data_point = {
                    'date': datetime.fromtimestamp(agg.timestamp / 1000).isoformat() + 'Z',
                    'ticker': trade_pair.trade_pair_id.lower(),
                    'open': float(agg.open),
                    'high': float(agg.high),
                    'low': float(agg.low),
                    'close': float(agg.close),
                    'volume': float(agg.volume) if hasattr(agg, 'volume') and agg.volume else 0.0
                }
                data_points.append(data_point)

            print(f"      Retrieved {len(data_points)} data points")
            return data_points

        except Exception as e:
            print(f"      Error fetching {trade_pair.trade_pair_id}: {e}")
            return []

    def is_market_open_at_time(self, trade_pair: TradePair, timestamp_ms: int) -> bool:
        """Check if crypto market is open at given timestamp - crypto markets are always open"""
        # Crypto markets trade 24/7, so they're always open
        return True

    def process_raw_data(self, raw_data: List[Dict], trade_pair: TradePair) -> pd.DataFrame:
        """Process raw Polygon data into a clean DataFrame with backfilling for crypto"""

        if not raw_data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(raw_data)

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['date'])
        df['timestamp_ms'] = (df['timestamp'].astype('int64') / 1e6).astype('int64')

        # Since crypto markets are always open, we don't need market hours filtering
        # but we still want to create a complete minute timeline for consistency

        if df.empty:
            return pd.DataFrame()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Create complete minute timeline
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()

        complete_timeline = []
        current_time = start_time.replace(second=0, microsecond=0)

        while current_time <= end_time:
            complete_timeline.append(current_time)
            current_time += timedelta(minutes=1)

        # Create complete DataFrame
        complete_df = pd.DataFrame({'timestamp': complete_timeline})
        complete_df['timestamp_ms'] = (complete_df['timestamp'].astype('int64') / 1e6).astype('int64')

        # Merge with actual data
        result_df = complete_df.merge(df[['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']],
                                     on='timestamp', how='left')

        # Forward fill missing values (backfill with last known data)
        result_df['is_backfilled'] = result_df['close'].isna()

        # Fill forward
        result_df['ticker'] = result_df['ticker'].ffill().fillna(trade_pair.trade_pair_id.lower())
        result_df['open'] = result_df['open'].ffill()
        result_df['high'] = result_df['high'].ffill()
        result_df['low'] = result_df['low'].ffill()
        result_df['close'] = result_df['close'].ffill()
        result_df['volume'] = result_df['volume'].ffill().fillna(0.0)

        # Add trade pair column
        result_df['trade_pair'] = trade_pair.trade_pair_id

        # Reorder columns
        result_df = result_df[['timestamp', 'timestamp_ms', 'trade_pair', 'open', 'high', 'low', 'close', 'volume', 'is_backfilled']]

        return result_df

    def generate_dataset(self, months_back: int = 18, output_file: str = 'polygon_crypto_dataset.csv'):
        """Generate complete crypto dataset for all trade pairs"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)  # Approximate months

        print(f"Generating Polygon crypto dataset from {start_date.date()} to {end_date.date()}")
        print(f"Trade pairs: {list(self.trade_pairs.keys())}")

        # Calculate optimal chunk size and estimates
        chunk_days = self.calculate_optimal_chunk_days()
        total_days = (end_date - start_date).days
        estimated_chunks = (total_days // chunk_days + 1) * len(self.trade_pairs)

        print(f"Optimal chunk size: {chunk_days} days (based on {self.candle_limit:,} candle limit)")
        print(f"Estimated API calls: {estimated_chunks}")
        print(f"Estimated time: ~{estimated_chunks * 0.5 / 60:.1f} minutes with rate limiting")

        all_data = []

        # Process each trade pair
        for ticker, trade_pair in self.trade_pairs.items():
            print(f"\nProcessing {ticker}...")

            current_date = start_date
            ticker_data = []

            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days), end_date)

                # Fetch raw data
                raw_data = self.fetch_crypto_data(trade_pair, current_date, chunk_end)

                if raw_data:
                    # Process and filter data
                    processed_df = self.process_raw_data(raw_data, trade_pair)

                    if not processed_df.empty:
                        ticker_data.append(processed_df)
                        backfilled = processed_df['is_backfilled'].sum()
                        print(f"      Added {len(processed_df)} data points ({backfilled} backfilled)")

                current_date = chunk_end

            # Combine all chunks for this ticker
            if ticker_data:
                ticker_combined = pd.concat(ticker_data, ignore_index=True)
                ticker_combined = ticker_combined.drop_duplicates(subset=['timestamp', 'trade_pair']).sort_values('timestamp')
                all_data.append(ticker_combined)
                print(f"  Total for {ticker}: {len(ticker_combined)} data points")

        # Combine all trade pairs
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df = final_df.sort_values(['timestamp', 'trade_pair']).reset_index(drop=True)

            # Summary statistics
            print(f"\nDataset Summary:")
            print(f"Total records: {len(final_df)}")
            print(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
            print(f"Backfilled records: {final_df['is_backfilled'].sum()} ({final_df['is_backfilled'].mean():.2%})")

            print("\nRecords per trade pair:")
            for tp in final_df['trade_pair'].unique():
                tp_data = final_df[final_df['trade_pair'] == tp]
                backfilled = tp_data['is_backfilled'].sum()
                print(f"  {tp}: {len(tp_data)} records ({backfilled} backfilled, {backfilled/len(tp_data):.2%})")

            # Save to CSV
            print(f"\nSaving to {output_file}...")
            final_df.to_csv(output_file, index=False)

            print("Dataset generation complete!")
            return final_df
        else:
            print("No data retrieved!")
            return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Generate Polygon crypto dataset')
    parser.add_argument('--months', type=int, default=18, help='Number of months back to fetch (default: 18)')
    parser.add_argument('--output', type=str, default='polygon_crypto_dataset.csv', help='Output CSV filename')
    parser.add_argument('--test', action='store_true', help='Run in test mode (only fetch last 3 days)')

    args = parser.parse_args()

    # Get API key
    try:
        secrets = ValiUtils.get_secrets()
        api_key = secrets.get("polygon_apikey")

        if not api_key:
            print("Error: polygon_apikey not found in secrets.json")
            return
    except Exception as e:
        print(f"Error loading secrets: {e}")
        return

    # Create generator
    generator = PolygonCryptoDataGenerator(api_key)

    # Generate dataset
    if args.test:
        print("Running in test mode (last 3 days only)")
        # Test with 3 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        print(f"Testing with BTC/USD from {start_date.date()} to {end_date.date()}")
        raw_data = generator.fetch_crypto_data(TradePair.BTCUSD, start_date, end_date)

        if raw_data:
            df = generator.process_raw_data(raw_data, TradePair.BTCUSD)
            print(f"Processed {len(df)} records")
            print(f"Backfilled: {df['is_backfilled'].sum()}")

            # Save test output
            test_output = args.output.replace('.csv', '_test.csv')
            df.to_csv(test_output, index=False)
            print(f"Test data saved to {test_output}")

            # Show sample
            print("\nSample data:")
            print(df.head(10))
        else:
            print("No test data retrieved")
    else:
        # Full generation
        generator.generate_dataset(args.months, args.output)


if __name__ == "__main__":
    main()