#!/usr/bin/env python3
"""
Generate historical crypto dataset from Tiingo API
- Minutely resolution
- Crypto markets trade 24/7, so no market hours filtering needed
- Backfills missing minutes with last known data
- Covers BTC/USD for last 18 months
- Output format: CSV (same structure as forex version)
- Uses Coinbase (GDAX) exchange data
"""

import json
import requests
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vali_objects.utils.vali_utils import ValiUtils
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_objects.vali_config import TradePair
import time


class TiingoCryptoDataGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.market_calendar = UnifiedMarketCalendar()

        # Trade pairs to fetch - focusing on BTC
        self.trade_pairs = {
            'BTCUSD': TradePair.BTCUSD
        }

        # Tiingo uses GDAX as the preferred exchange for crypto
        self.exchange = 'GDAX'  # Coinbase exchange identifier

        # Rate limiting - Tiingo has lower limits than Polygon
        self.requests_per_minute = 50
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = 0

    def rate_limit(self):
        """Simple rate limiting to respect API limits"""
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
        time.sleep(0.1)

    def fetch_crypto_data(self, ticker: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch crypto data from Tiingo API for given date range"""

        start_formatted = start_date.strftime('%Y-%m-%d')
        end_formatted = end_date.strftime('%Y-%m-%d')

        # Use Tiingo crypto prices endpoint with Coinbase exchange
        url = f"https://api.tiingo.com/tiingo/crypto/prices?tickers={ticker.lower()}&startDate={start_formatted}&endDate={end_formatted}&resampleFreq=1min&token={self.api_key}&exchanges={self.exchange}"

        self.rate_limit()

        try:
            response = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                # Tiingo crypto API returns a list with one element containing the ticker and priceData
                if data and len(data) > 0 and 'priceData' in data[0]:
                    return data[0]['priceData']  # Extract the actual price data
                return []
            else:
                print(f"Error fetching {ticker} for {start_formatted} to {end_formatted}: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Exception fetching {ticker}: {e}")
            return []

    def is_market_open_at_time(self, trade_pair: TradePair, timestamp_ms: int) -> bool:
        """Check if crypto market is open at given timestamp - crypto markets are always open"""
        # Crypto markets trade 24/7, so they're always open
        return True

    def process_raw_data(self, raw_data: List[Dict], trade_pair: TradePair) -> pd.DataFrame:
        """Process raw Tiingo data into a clean DataFrame with backfilling for crypto"""

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

        # Merge with actual data - only use available columns
        columns_to_merge = ['timestamp', 'open', 'high', 'low', 'close']
        # Include volume if it exists in the data
        if 'volume' in df.columns:
            columns_to_merge.append('volume')

        result_df = complete_df.merge(df[columns_to_merge], on='timestamp', how='left')

        # Forward fill missing values (backfill with last known data)
        result_df['is_backfilled'] = result_df['close'].isna()

        # Fill forward
        result_df['open'] = result_df['open'].ffill()
        result_df['high'] = result_df['high'].ffill()
        result_df['low'] = result_df['low'].ffill()
        result_df['close'] = result_df['close'].ffill()

        # Handle volume if present
        if 'volume' in result_df.columns:
            result_df['volume'] = result_df['volume'].ffill().fillna(0.0)

        # Add ticker and trade pair columns
        result_df['ticker'] = trade_pair.trade_pair_id.lower()
        result_df['trade_pair'] = trade_pair.trade_pair_id

        # Reorder columns
        base_columns = ['timestamp', 'timestamp_ms', 'trade_pair', 'open', 'high', 'low', 'close']
        if 'volume' in result_df.columns:
            base_columns.append('volume')
        base_columns.append('is_backfilled')

        result_df = result_df[base_columns]

        return result_df

    def generate_dataset(self, months_back: int = 18, output_file: str = 'tiingo_crypto_dataset.csv', chunk_days: int = 2):
        """Generate complete crypto dataset for all trade pairs"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)  # Approximate months

        print(f"Generating Tiingo crypto dataset from {start_date.date()} to {end_date.date()}")
        print(f"Trade pairs: {list(self.trade_pairs.keys())}")
        print(f"Exchange: {self.exchange}")

        # Calculate estimated performance (crypto is 24/7 so more data than forex)
        total_days = (end_date - start_date).days
        # Use configurable chunk size to avoid Tiingo truncating data
        # chunk_days * 1440 minutes/day = data points per request

        estimated_chunks = (total_days // chunk_days + 1) * len(self.trade_pairs)
        print(f"Estimated API calls: {estimated_chunks} ({chunk_days}-day chunks)")
        print(f"Data points per request: ~{chunk_days * 1440} (24/7 trading)")
        print(f"Estimated time: ~{estimated_chunks * 2 / 60:.1f} minutes with rate limiting")

        all_data = []

        # Process each trade pair
        for ticker, trade_pair in self.trade_pairs.items():
            print(f"\nProcessing {ticker}...")

            # Split into smaller date ranges to avoid early termination issues
            current_date = start_date

            ticker_data = []

            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=chunk_days), end_date)

                print(f"  Fetching {current_date.date()} to {chunk_end.date()}...")

                # Fetch raw data
                raw_data = self.fetch_crypto_data(ticker, current_date, chunk_end)

                if raw_data:
                    # Process and filter data
                    processed_df = self.process_raw_data(raw_data, trade_pair)

                    if not processed_df.empty:
                        ticker_data.append(processed_df)
                        backfilled = processed_df['is_backfilled'].sum()
                        print(f"    Added {len(processed_df)} data points ({backfilled} backfilled)")

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
    parser = argparse.ArgumentParser(description='Generate Tiingo crypto dataset')
    parser.add_argument('--months', type=int, default=18, help='Number of months back to fetch (default: 18)')
    parser.add_argument('--output', type=str, default='tiingo_crypto_dataset.csv', help='Output CSV filename')
    parser.add_argument('--test', action='store_true', help='Run in test mode (only fetch last 3 days)')
    parser.add_argument('--chunk-days', type=int, default=2, help='Days per API request chunk (default: 2)')

    args = parser.parse_args()

    # Get API key
    try:
        secrets = ValiUtils.get_secrets()
        api_key = secrets.get("tiingo_apikey")

        if not api_key:
            print("Error: tiingo_apikey not found in secrets.json")
            return
    except Exception as e:
        print(f"Error loading secrets: {e}")
        return

    # Create generator
    generator = TiingoCryptoDataGenerator(api_key)

    # Generate dataset
    if args.test:
        print("Running in test mode (last 3 days only)")
        # Test with 3 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        print(f"Testing with BTC/USD from {start_date.date()} to {end_date.date()}")
        raw_data = generator.fetch_crypto_data('BTCUSD', start_date, end_date)

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
        generator.generate_dataset(args.months, args.output, args.chunk_days)


if __name__ == "__main__":
    main()