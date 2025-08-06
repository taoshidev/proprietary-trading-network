#!/usr/bin/env python3
"""
Daily Portfolio Returns Calculator

This script calculates daily portfolio returns for each miner by:
1. Loading positions from database using PositionSourceManager API
2. Loading eliminations for filtering out eliminated miners
3. Stepping through time day by day (UTC)
4. Filtering out eliminated miners at each date
5. Calculating total portfolio return at the beginning of each UTC day
6. Using live_price_fetcher.get_close_at_date for pricing

Usage:
    python daily_portfolio_returns.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--hotkeys HOTKEY1,HOTKEY2,...] [--elimination-source DATABASE]
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

import bittensor as bt
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

from time_util.time_util import TimeUtil
from time_util.time_util import MS_IN_24_HOURS
from vali_objects.position import Position
from vali_objects.utils.position_source import PositionSourceManager, PositionSource
from vali_objects.utils.elimination_source import EliminationSourceManager, EliminationSource
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.vali_config import TradePair, TradePairCategory, CryptoSubcategory, ForexSubcategory
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.utils.vali_utils import ValiUtils


# Database setup
Base = declarative_base()

class MinerPortValuesModel(Base):
    """Database model for storing miner port values matching taoshi-ts-ptn.miner_port_values schema."""
    __tablename__ = 'miner_port_values'
    
    miner_hotkey = Column(String(255), primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    date = Column(String(10))  # YYYY-MM-DD format
    # Aggregate port values
    all_port_value = Column(Float)
    crypto_port_value = Column(Float)
    forex_port_value = Column(Float)
    # Crypto subcategory port values
    crypto_majors_port_value = Column(Float)
    crypto_alts_port_value = Column(Float)
    # Forex subcategory port values
    forex_group1_port_value = Column(Float)
    forex_group2_port_value = Column(Float)
    forex_group3_port_value = Column(Float)
    forex_group4_port_value = Column(Float)
    forex_group5_port_value = Column(Float)
    # Count columns
    all_count = Column(Integer)
    crypto_count = Column(Integer)
    forex_count = Column(Integer)
    crypto_majors_count = Column(Integer)
    crypto_alts_count = Column(Integer)
    forex_group1_count = Column(Integer)
    forex_group2_count = Column(Integer)
    forex_group3_count = Column(Integer)
    forex_group4_count = Column(Integer)
    forex_group5_count = Column(Integer)


class DatabaseManager:
    """Handles database operations for daily portfolio returns."""
    
    def __init__(self, connection_string: str):
        """Initialize database connection."""
        self.engine = create_engine(connection_string)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.ensure_table_exists()
    
    def ensure_table_exists(self):
        """Check that miner_port_values table exists - don't create new tables."""
        try:
            inspector = inspect(self.engine)
            if not inspector.has_table('miner_port_values'):
                raise RuntimeError("miner_port_values table does not exist in database. This script requires an existing miner_port_values table.")
            
            bt.logging.info("Database table 'miner_port_values' found and ready")
        except Exception as e:
            bt.logging.error(f"Failed to verify database table exists: {e}")
            raise
    
    def get_existing_dates(self, start_date: str, end_date: str) -> Set[str]:
        """Get all dates that already exist in database within date range."""
        try:
            with self.SessionFactory() as session:
                result = session.execute(text(
                    "SELECT DISTINCT date FROM miner_port_values "
                    "WHERE date >= :start_date AND date <= :end_date"
                ), {"start_date": start_date, "end_date": end_date})
                
                existing_dates = {row[0] for row in result.fetchall()}
                bt.logging.info(f"Found {len(existing_dates)} existing dates in database")
                return existing_dates
                
        except Exception as e:
            bt.logging.error(f"Failed to fetch existing dates: {e}")
            return set()
    
    def insert_daily_returns(self, daily_returns: List[Dict], target_timestamp: datetime) -> bool:
        """Insert daily returns for a specific date using miner_port_values schema."""
        if not daily_returns:
            date_str = target_timestamp.strftime("%Y-%m-%d")
            bt.logging.warning(f"No returns to insert for {date_str}")
            return False
        
        try:
            with self.SessionFactory() as session:
                # Create model instances and add them (no deletion - should be handled by skip logic)
                for value_dict in daily_returns:
                    port_value = MinerPortValuesModel(**value_dict)
                    session.add(port_value)
                
                session.commit()
                
                date_str = target_timestamp.strftime("%Y-%m-%d")
                bt.logging.info(f"✓ Successfully inserted {len(daily_returns)} returns for {date_str}")
                return True
                
        except Exception as e:
            date_str = target_timestamp.strftime("%Y-%m-%d")
            bt.logging.error(f"Failed to insert returns for {date_str}: {e}")
            return False


@dataclass
class FilterStats:
    """Statistics for position filtering operations."""
    total_positions_before_filter: int = 0
    equities_positions_skipped: int = 0
    indices_positions_skipped: int = 0
    date_filtered_out: int = 0
    final_positions: int = 0
    
    def has_skipped_assets(self) -> bool:
        """Check if any equities or indices were skipped."""
        return self.equities_positions_skipped > 0 or self.indices_positions_skipped > 0


@dataclass
class DailyStats:
    """Statistics for a single day's return calculations."""
    successful_miners: int = 0
    failed_miners: int = 0
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0
    position_returns: List[float] = None
    portfolio_returns: List[float] = None
    extreme_returns: Dict = None
    trade_pair_usage: Dict[str, int] = None
    skip_stats: FilterStats = None
    elimination_stats: Dict = None
    
    def __post_init__(self):
        if self.position_returns is None:
            self.position_returns = []
        if self.portfolio_returns is None:
            self.portfolio_returns = []
        if self.extreme_returns is None:
            self.extreme_returns = {'best': None, 'worst': None}
        if self.trade_pair_usage is None:
            self.trade_pair_usage = defaultdict(int)
        if self.skip_stats is None:
            self.skip_stats = FilterStats()
        if self.elimination_stats is None:
            self.elimination_stats = {"total_hotkeys": 0, "eliminated_hotkeys": 0}


class PositionFilter:
    """Handles filtering of positions by date and asset type."""
    
    @staticmethod
    def filter_single_position(position: Position, cutoff_date_ms: int) -> Tuple[Optional[Position], str]:
        """
        Filter a single position by date and asset type.
        
        Returns:
            Tuple of (filtered_position, skip_reason). Position is None if skipped.
        """
        # Skip positions for equities and indices assets
        if position.trade_pair.is_equities:
            return None, "equities"
        elif position.trade_pair.is_indices:
            return None, "indices"
        
        # Filter orders to only include those before/at cutoff
        filtered_orders = [
            order for order in position.orders 
            if order.processed_ms <= cutoff_date_ms  # Inclusive comparison
        ]
        
        if not filtered_orders:
            return None, "date_filtered"
        
        # Create a copy of the position with filtered orders
        filtered_position = Position(
            miner_hotkey=position.miner_hotkey,
            position_uuid=position.position_uuid,
            open_ms=position.open_ms,
            close_ms=position.close_ms if position.close_ms and position.close_ms <= cutoff_date_ms else None,
            trade_pair=position.trade_pair,
            orders=filtered_orders,
            position_type=position.position_type,
            is_closed_position=position.is_closed_position and position.close_ms and position.close_ms <= cutoff_date_ms,
        )
        filtered_position.rebuild_position_with_updated_orders()
        return filtered_position, "kept"
    
    @staticmethod
    def filter_and_analyze_positions(
        all_positions: Dict[str, List[Position]],
        target_date_ms: int
    ) -> Tuple[Dict[str, List[Position]], Set[TradePair], FilterStats]:
        """
        Filter positions by date and identify trade pairs needing prices.
        
        Returns:
            Tuple of (filtered_positions_by_hotkey, trade_pairs_needing_prices, filter_stats)
        """
        filtered_positions_by_hotkey = {}
        trade_pairs_needing_prices = set()
        stats = FilterStats()
        
        for hotkey, positions in all_positions.items():
            stats.total_positions_before_filter += len(positions)
            filtered_positions = []
            
            for position in positions:
                filtered_position, skip_reason = PositionFilter.filter_single_position(position, target_date_ms)
                
                if skip_reason == "equities":
                    stats.equities_positions_skipped += 1
                    bt.logging.debug(f"Skipping {position.trade_pair.trade_pair} position - equities not supported")
                elif skip_reason == "indices":
                    stats.indices_positions_skipped += 1
                    bt.logging.debug(f"Skipping {position.trade_pair.trade_pair} position - indices not supported")
                elif skip_reason == "date_filtered":
                    stats.date_filtered_out += 1
                elif skip_reason == "kept" and filtered_position:
                    filtered_positions.append(filtered_position)
                    stats.final_positions += 1
                    
                    # Check if position needs a price (open at target date)
                    if not filtered_position.is_closed_position or (filtered_position.close_ms and filtered_position.close_ms > target_date_ms):
                        trade_pairs_needing_prices.add(filtered_position.trade_pair)
            
            if filtered_positions:
                filtered_positions_by_hotkey[hotkey] = filtered_positions
        
        return filtered_positions_by_hotkey, trade_pairs_needing_prices, stats


class PriceFetcher:
    """Handles multi-threaded price fetching for trade pairs."""
    
    def __init__(self, live_price_fetcher: LivePriceFetcher, max_workers: int = 30):
        self.live_price_fetcher = live_price_fetcher
        self.max_workers = max_workers
    
    def fetch_single_price_source(self, trade_pair: TradePair, target_date_ms: int) -> Tuple[TradePair, PriceSource, str]:
        """Fetch price source for a single trade pair."""
        target_datetime = datetime.fromtimestamp(target_date_ms / 1000, tz=timezone.utc)
        target_date_utc_str = target_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        bt.logging.debug(f"Fetching price for {trade_pair.trade_pair} at {target_date_utc_str}")
        
        price_source = self.live_price_fetcher.get_close_at_date(
            trade_pair=trade_pair,
            timestamp_ms=target_date_ms,
            verbose=False
        )
        
        if price_source is None:
            error_msg = f"Failed to fetch price for {trade_pair.trade_pair} at {target_date_utc_str}"
            raise Exception(error_msg)
        
        return trade_pair, price_source, None
    
    def fetch_multiple_price_sources(
        self, 
        trade_pairs: Set[TradePair], 
        target_date_ms: int
    ) -> Dict[TradePair, PriceSource]:
        """Fetch price sources for multiple trade pairs concurrently."""
        if not trade_pairs:
            return {}
        
        start_time = time.time()
        
        target_datetime = datetime.fromtimestamp(target_date_ms / 1000, tz=timezone.utc)
        target_date_utc_str = target_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        price_sources = {}
        errors = []
        
        bt.logging.info(f"Starting parallel price fetch for {len(trade_pairs)} trade pairs at {target_date_utc_str} using {self.max_workers} threads")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_trade_pair = {
                executor.submit(self.fetch_single_price_source, tp, target_date_ms): tp
                for tp in trade_pairs
            }
            
            for future in as_completed(future_to_trade_pair):
                trade_pair, price_source, error_msg = future.result()
                
                if error_msg is None:
                    price_sources[trade_pair] = price_source
                else:
                    errors.append(error_msg)
                    bt.logging.error(error_msg)
        
        elapsed_time = time.time() - start_time
        bt.logging.info(f"Price fetch results for {target_date_utc_str}: {len(price_sources)} successful, {len(errors)} failed (took {elapsed_time:.2f}s)")
        
        if errors:
            error_summary = f"Failed to fetch prices for {len(errors)} trade pairs: {errors[:3]}"
            if len(errors) > 3:
                error_summary += f" (and {len(errors) - 3} more)"
            raise Exception(error_summary)
        
        return price_sources


class ReturnCalculator:
    """Handles portfolio and position return calculations."""
    
    @staticmethod
    def calculate_position_return(
        position: Position,
        target_date_ms: int,
        cached_price_sources: Dict[TradePair, PriceSource]
    ) -> float:
        """Calculate return for a single position."""
        # If position is closed and closed before/at target date, use actual return
        if position.is_closed_position and position.close_ms and position.close_ms <= target_date_ms:
            return position.return_at_close
        
        # For open positions, calculate using cached price
        if position.trade_pair not in cached_price_sources:
            raise ValueError(f"Price not available for {position.trade_pair.trade_pair} at target date")
        
        price_source = cached_price_sources[position.trade_pair]
        
        # Create a copy to avoid modifying the original position
        position_copy = Position(
            miner_hotkey=position.miner_hotkey,
            position_uuid=position.position_uuid,
            open_ms=position.open_ms,
            close_ms=position.close_ms,
            trade_pair=position.trade_pair,
            orders=position.orders[:],
            position_type=position.position_type,
            is_closed_position=position.is_closed_position,
        )
        position_copy.rebuild_position_with_updated_orders()
        
        price = price_source.parse_appropriate_price(
            now_ms=target_date_ms,
            is_forex=position.trade_pair.is_forex,
            order_type=position.orders[0].order_type,
            position=position_copy
        )
        
        position_copy.set_returns(realtime_price=price, time_ms=target_date_ms)
        return position_copy.return_at_close


class PositionCategorizer:
    """Handles categorization of positions into crypto/forex subcategories."""
    
    @staticmethod
    def categorize_position(trade_pair_id: str) -> List[str]:
        """
        Determine which category(ies) a trade pair belongs to using the TradePair enum.
        
        Args:
            trade_pair_id: The ID of the trade pair
            
        Returns:
            List[str]: A list of category names
        """
        categories = []
        
        # Add to 'all' category by default
        categories.append('all')
        
        # Get the TradePair object from the ID
        trade_pair_obj = TradePair.from_trade_pair_id(trade_pair_id)
        
        if trade_pair_obj is None:
            bt.logging.warning(f"Unknown trade pair ID: {trade_pair_id}")
            return categories
        
        # Get the trade pair info (list format from enum)
        trade_pair_info = trade_pair_obj.value
        
        # Extract category and subcategory from the trade pair info
        # Format: [id, display_name, tick_size, min_leverage, max_leverage, category, subcategory]
        if len(trade_pair_info) >= 7:
            category = trade_pair_info[5]  # TradePairCategory
            subcategory = trade_pair_info[6]  # Subcategory (CryptoSubcategory or ForexSubcategory)
            
            # Add the primary category
            if category == TradePairCategory.CRYPTO:
                categories.append('crypto')
                
                # Add crypto subcategories
                if subcategory == CryptoSubcategory.MAJORS:
                    categories.append('crypto_majors')
                elif subcategory == CryptoSubcategory.ALTS:
                    categories.append('crypto_alts')
                    
            elif category == TradePairCategory.FOREX:
                categories.append('forex')
                
                # Add forex subcategories
                if subcategory == ForexSubcategory.G1:
                    categories.append('forex_group1')
                elif subcategory == ForexSubcategory.G2:
                    categories.append('forex_group2')
                elif subcategory == ForexSubcategory.G3:
                    categories.append('forex_group3')
                elif subcategory == ForexSubcategory.G4:
                    categories.append('forex_group4')
                elif subcategory == ForexSubcategory.G5:
                    categories.append('forex_group5')
        
        return categories


class PositionAnalyzer:
    """Analyzes positions for statistical insights."""
    
    @staticmethod
    def analyze_positions_for_date(
        positions: List[Position],
        target_date_ms: int,
        cached_price_sources: Dict[TradePair, PriceSource]
    ) -> Dict:
        """Analyze positions for detailed statistics."""
        stats = {
            'open_positions': 0,
            'closed_positions': 0,
            'returns': [],
            'leverage_distribution': [],
            'position_durations': [],
            'trade_pairs': set()
        }
        
        for position in positions:
            stats['trade_pairs'].add(position.trade_pair.trade_pair)
            
            # Calculate position duration up to target date
            if position.is_closed_position and position.close_ms and position.close_ms <= target_date_ms:
                stats['closed_positions'] += 1
                duration_ms = position.close_ms - position.open_ms
            else:
                stats['open_positions'] += 1
                duration_ms = target_date_ms - position.open_ms
            
            duration_days = duration_ms / MS_IN_24_HOURS
            stats['position_durations'].append(duration_days)
            
            # Get leverage info
            if position.orders:
                stats['leverage_distribution'].append(abs(position.orders[0].leverage))
            
            # Calculate individual position return
            try:
                position_return = ReturnCalculator.calculate_position_return(
                    position, target_date_ms, cached_price_sources
                )
                stats['returns'].append(position_return)
            except Exception as e:
                bt.logging.warning(f"Failed to calculate return for position {position.position_uuid}: {e}")
        
        return stats


class CategoryReturnCalculator:
    """Calculates returns by asset categories and subcategories."""
    
    @staticmethod
    def calculate_miner_returns_by_category(
        positions: List[Position],
        target_date_ms: int,
        cached_price_sources: Dict[TradePair, PriceSource]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate portfolio values by category for each miner based on positions.
        Follows the same logic as miner_returns.py reference implementation.
        
        Args:
            positions: List of positions for all miners
            target_date_ms: Target timestamp
            cached_price_sources: Cached price sources for calculations
            
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: Nested dict of miner_hotkey -> category -> {"return": value, "count": count}
        """
        from collections import defaultdict
        
        # Initialize data structure for results (following reference exactly)
        # Format: {miner_hotkey: {category: {"return": cumulative_portfolio_value, "count": count}}}
        miner_category_returns = defaultdict(
            lambda: defaultdict(lambda: {"return": 1.0, "count": 0}))
        
        # Process each position (following reference logic exactly)
        for position in positions:
            # Calculate return_at_close for this position at the target date
            position_return = ReturnCalculator.calculate_position_return(
                position, target_date_ms, cached_price_sources
            )
            
            # Fail fast - if we can't calculate return, something is wrong
            if not position_return:
                raise ValueError(f"Position {position.position_uuid} returned invalid return: {position_return}")
            
            # Fail fast - if return is not positive, something is wrong
            if position_return <= 0:
                raise ValueError(f"Position {position.position_uuid} returned non-positive return: {position_return}")
                
            miner_hotkey = position.miner_hotkey
            trade_pair_id = position.trade_pair.trade_pair_id  # Use ID not display name
            
            # Determine categories for this position
            categories = PositionCategorizer.categorize_position(trade_pair_id)
            
            # Add return to each relevant category
            for category in categories:
                # Directly multiply the return values together (reference logic)
                miner_category_returns[miner_hotkey][category]["return"] *= position_return
                miner_category_returns[miner_hotkey][category]["count"] += 1
        
        # Convert the results to the final format (following reference)
        results = {}
        for miner_hotkey, categories in miner_category_returns.items():
            results[miner_hotkey] = {}
            for category, data in categories.items():
                results[miner_hotkey][category] = {
                    "return": data["return"],
                    "count": data["count"]
                }
        
        # Log results (following reference)
        bt.logging.info(f"Calculated returns for {len(results)} miners")
        
        return results
    
    @staticmethod
    def prepare_insert_values(
        miner_returns: Dict[str, Dict[str, Dict[str, float]]],
        target_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """
        Prepare insert values for database from calculated portfolio values.
        
        Args:
            miner_returns: Dictionary of miner portfolio values by category
            target_timestamp: The timestamp for the database records
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries for database insertion
        """
        insert_values = []
        for miner_hotkey, categories in miner_returns.items():
            insert_values.append({
                "miner_hotkey": miner_hotkey,
                "timestamp": target_timestamp,
                "date": target_timestamp.strftime("%Y-%m-%d"),
                # Aggregate port values
                "all_port_value": categories.get("all", {}).get("return", 1.0),
                "crypto_port_value": categories.get("crypto", {}).get("return", 1.0),
                "forex_port_value": categories.get("forex", {}).get("return", 1.0),
                # Crypto subcategory port values
                "crypto_majors_port_value": categories.get("crypto_majors", {}).get("return", 1.0),
                "crypto_alts_port_value": categories.get("crypto_alts", {}).get("return", 1.0),
                # Forex subcategory port values
                "forex_group1_port_value": categories.get("forex_group1", {}).get("return", 1.0),
                "forex_group2_port_value": categories.get("forex_group2", {}).get("return", 1.0),
                "forex_group3_port_value": categories.get("forex_group3", {}).get("return", 1.0),
                "forex_group4_port_value": categories.get("forex_group4", {}).get("return", 1.0),
                "forex_group5_port_value": categories.get("forex_group5", {}).get("return", 1.0),
                # Count columns
                "all_count": categories.get("all", {}).get("count", 0),
                "crypto_count": categories.get("crypto", {}).get("count", 0),
                "forex_count": categories.get("forex", {}).get("count", 0),
                "crypto_majors_count": categories.get("crypto_majors", {}).get("count", 0),
                "crypto_alts_count": categories.get("crypto_alts", {}).get("count", 0),
                "forex_group1_count": categories.get("forex_group1", {}).get("count", 0),
                "forex_group2_count": categories.get("forex_group2", {}).get("count", 0),
                "forex_group3_count": categories.get("forex_group3", {}).get("count", 0),
                "forex_group4_count": categories.get("forex_group4", {}).get("count", 0),
                "forex_group5_count": categories.get("forex_group5", {}).get("count", 0)
            })
        return insert_values


class Logger:
    """Handles all logging and summary reporting."""
    
    @staticmethod
    def log_filtering_stats(date_str: str, stats: FilterStats):
        """Log position filtering statistics."""
        if stats.has_skipped_assets():
            bt.logging.info(f"Filtered positions for {date_str} UTC: "
                           f"{stats.final_positions} kept, "
                           f"{stats.equities_positions_skipped} equities skipped, "
                           f"{stats.indices_positions_skipped} indices skipped, "
                           f"{stats.date_filtered_out} date-filtered out")
    
    @staticmethod
    def log_daily_summary(date_str: str, daily_stats: DailyStats):
        """Log comprehensive daily return calculation summary."""
        bt.logging.info(f"📊 Daily Return Summary for {date_str} UTC:")
        
        # Miner and position counts
        bt.logging.info(f"   Miners: {daily_stats.successful_miners} successful, {daily_stats.failed_miners} failed")
        bt.logging.info(f"   Positions: {daily_stats.total_positions} total ({daily_stats.open_positions} open, {daily_stats.closed_positions} closed)")
        
        # Include filtering information if available
        if daily_stats.skip_stats and daily_stats.skip_stats.has_skipped_assets():
            skip_stats = daily_stats.skip_stats
            bt.logging.info(f"   Filtering: {skip_stats.equities_positions_skipped} equities + {skip_stats.indices_positions_skipped} indices skipped")
        
        # Include elimination filtering information
        if daily_stats.elimination_stats and daily_stats.elimination_stats.get("total_hotkeys", 0) > 0:
            elim_stats = daily_stats.elimination_stats
            eliminated_count = elim_stats.get("eliminated_hotkeys", 0)
            total_count = elim_stats.get("total_hotkeys", 0)
            if eliminated_count > 0:
                bt.logging.info(f"   Eliminations: {eliminated_count}/{total_count} miners eliminated and excluded")
            else:
                bt.logging.info(f"   Eliminations: 0/{total_count} miners eliminated")
        
        # Portfolio return statistics
        if daily_stats.portfolio_returns:
            portfolio_returns = daily_stats.portfolio_returns
            avg_return = sum(portfolio_returns) / len(portfolio_returns)
            avg_return_pct = (avg_return - 1.0) * 100
            median_return_pct = ((sorted(portfolio_returns)[len(portfolio_returns)//2]) - 1.0) * 100
            
            bt.logging.info(f"   Portfolio Returns: avg={avg_return_pct:.3f}%, median={median_return_pct:.3f}%")
            
            # Extreme returns
            if daily_stats.extreme_returns['best']:
                best_hotkey, best_pct = daily_stats.extreme_returns['best']
                bt.logging.info(f"   Best Performer: {best_hotkey[:12]}... ({best_pct:.3f}%)")
            
            if daily_stats.extreme_returns['worst']:
                worst_hotkey, worst_pct = daily_stats.extreme_returns['worst']
                bt.logging.info(f"   Worst Performer: {worst_hotkey[:12]}... ({worst_pct:.3f}%)")
            
            # Return distribution
            positive_returns = sum(1 for r in portfolio_returns if r > 1.0)
            bt.logging.info(f"   Return Distribution: {positive_returns}/{len(portfolio_returns)} miners positive ({positive_returns/len(portfolio_returns)*100:.1f}%)")
        
        # Position return statistics
        if daily_stats.position_returns:
            pos_returns = daily_stats.position_returns
            pos_positive = sum(1 for r in pos_returns if r > 1.0)
            avg_pos_return_pct = ((sum(pos_returns) / len(pos_returns)) - 1.0) * 100
            bt.logging.info(f"   Position Returns: avg={avg_pos_return_pct:.3f}%, {pos_positive}/{len(pos_returns)} positive ({pos_positive/len(pos_returns)*100:.1f}%)")
        
        # Trade pair usage
        if daily_stats.trade_pair_usage:
            top_pairs = sorted(daily_stats.trade_pair_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            pair_summary = ", ".join([f"{pair}({count})" for pair, count in top_pairs])
            bt.logging.info(f"   Top Trade Pairs: {pair_summary}")
        
        bt.logging.info(f"   ✓ Completed processing {date_str} UTC\n")
    
    @staticmethod
    def log_final_summary(df: pd.DataFrame, start_ms: int, end_ms: int):
        """Log comprehensive final summary of all results."""
        start_date = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
        end_date = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
        
        bt.logging.info("=" * 80)
        bt.logging.info("🎯 FINAL PORTFOLIO RETURNS SUMMARY")
        bt.logging.info("=" * 80)
        
        # Overall statistics
        total_days = len(df["date"].unique())
        total_miners = len(df["hotkey"].unique())
        total_calculations = len(df)
        
        bt.logging.info(f"📅 Period: {start_date} to {end_date} UTC ({total_days} days)")
        bt.logging.info(f"👥 Miners: {total_miners} unique miners")
        bt.logging.info(f"📊 Calculations: {total_calculations} daily return calculations")
        bt.logging.info("")
        
        # Top performing miners
        bt.logging.info("🏆 TOP PERFORMING MINERS (Final Day Return):")
        final_day_data = df[df["date"] == df["date"].max()]
        top_performers = final_day_data.nlargest(10, "return_pct")
        
        for i, (_, row) in enumerate(top_performers.iterrows(), 1):
            bt.logging.info(f"   {i:2d}. {row['hotkey'][:16]}... : {row['return_pct']:+8.3f}% ({row['num_positions']} pos)")
        
        bt.logging.info("")
        
        # Worst performing miners
        bt.logging.info("📉 WORST PERFORMING MINERS (Final Day Return):")
        worst_performers = final_day_data.nsmallest(5, "return_pct")
        
        for i, (_, row) in enumerate(worst_performers.iterrows(), 1):
            bt.logging.info(f"   {i:2d}. {row['hotkey'][:16]}... : {row['return_pct']:+8.3f}% ({row['num_positions']} pos)")
        
        bt.logging.info("")
        
        # Overall statistics per miner
        bt.logging.info("📈 MINER PERFORMANCE OVER TIME:")
        miner_stats = []
        
        for hotkey in df["hotkey"].unique():
            miner_data = df[df["hotkey"] == hotkey].sort_values("date")
            if len(miner_data) > 0:
                first_return = miner_data.iloc[0]["portfolio_return"]
                final_return = miner_data.iloc[-1]["portfolio_return"]
                total_return = final_return / first_return if first_return != 0 else final_return
                total_return_pct = (total_return - 1.0) * 100
                
                avg_daily_return = miner_data["return_pct"].mean()
                avg_positions = miner_data["num_positions"].mean()
                
                miner_stats.append({
                    'hotkey': hotkey,
                    'total_return_pct': total_return_pct,
                    'avg_daily_return_pct': avg_daily_return,
                    'avg_positions': avg_positions,
                    'days_active': len(miner_data)
                })
        
        # Sort by total return
        miner_stats.sort(key=lambda x: x['total_return_pct'], reverse=True)
        
        bt.logging.info("   Top 10 by Total Return:")
        for i, stats in enumerate(miner_stats[:10], 1):
            bt.logging.info(f"   {i:2d}. {stats['hotkey'][:16]}... : {stats['total_return_pct']:+8.3f}% total "
                           f"({stats['avg_daily_return_pct']:+6.3f}% avg daily, {stats['avg_positions']:.1f} avg pos, {stats['days_active']} days)")
        
        bt.logging.info("")
        
        # Daily statistics
        bt.logging.info("📊 DAILY PERFORMANCE STATISTICS:")
        daily_stats = df.groupby("date").agg({
            "return_pct": ["mean", "median", "std", "min", "max", "count"],
            "num_positions": ["mean", "sum"],
            "hotkey": "nunique"
        }).round(3)
        
        for date in daily_stats.index[-5:]:  # Show last 5 days
            day_stats = daily_stats.loc[date]
            bt.logging.info(f"   {date}: avg={day_stats[('return_pct', 'mean')]:+6.3f}% "
                           f"median={day_stats[('return_pct', 'median')]:+6.3f}% "
                           f"(σ={day_stats[('return_pct', 'std')]:5.3f}%) "
                           f"range=[{day_stats[('return_pct', 'min')]:+6.3f}%, {day_stats[('return_pct', 'max')]:+6.3f}%] "
                           f"{int(day_stats[('hotkey', 'nunique')])} miners "
                           f"{int(day_stats[('num_positions', 'sum')])} positions")
        
        bt.logging.info("")
        bt.logging.info("=" * 80)
        bt.logging.info("✅ Portfolio returns calculation completed successfully!")
        bt.logging.info("=" * 80)


class DateUtils:
    """Utility functions for date handling."""
    
    @staticmethod
    def determine_date_bounds(positions_dict: Dict[str, List[Position]]) -> Tuple[int, int]:
        """Determine start and end dates based on first and last order times."""
        if not positions_dict:
            raise ValueError("No positions found to determine date bounds")
        
        first_order_ms = float('inf')
        last_order_ms = 0
        
        for hotkey, positions in positions_dict.items():
            for position in positions:
                first_order_ms = min(first_order_ms, position.open_ms)
                last_order_ms = max(last_order_ms, position.close_ms or 0)
        
        if first_order_ms == float('inf') or last_order_ms == 0:
            raise ValueError("No valid orders found in positions")
        
        # Round to start of UTC day
        first_date_ms = (first_order_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS
        # Round to end of UTC day
        last_date_ms = ((last_order_ms // MS_IN_24_HOURS) + 1) * MS_IN_24_HOURS
        
        # Cap end date at today (beginning of current UTC day) to avoid processing future dates
        today_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        today_start_ms = (today_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS
        
        if last_date_ms > today_start_ms:
            bt.logging.info(f"Capping end date at today: {TimeUtil.millis_to_formatted_date_str(today_start_ms)}")
            last_date_ms = today_start_ms
        
        bt.logging.info(f"Date bounds determined from orders: "
                        f"{TimeUtil.millis_to_formatted_date_str(first_date_ms)} to "
                        f"{TimeUtil.millis_to_formatted_date_str(last_date_ms)}")
        
        return first_date_ms, last_date_ms


def get_database_url_from_config() -> Optional[str]:
    """
    Read database URL from config-development.json file in working directory.
    Ensures the URL includes the database name for a complete connection string.
    
    Returns:
        Complete database URL string if found, None otherwise
    """
    config_file = "config-development.json"
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Navigate to secrets.db_ptn_editor_url
            base_url = config.get('secrets', {}).get('db_ptn_editor_url')
            
            if base_url:
                # Parse URL to ensure it has a database name
                parts = base_url.split('/')
                
                if len(parts) >= 4 and parts[-1]:
                    # URL already complete
                    bt.logging.info(f"Found complete database URL in {config_file}")
                    return base_url
                elif len(parts) >= 3:
                    # Has connection info but missing database name
                    if not base_url.endswith('/'):
                        base_url += '/'
                    complete_url = f"{base_url}taoshi-ts-ptn"
                    bt.logging.info(f"Completed database URL from {config_file} by appending database name")
                    return complete_url
                else:
                    bt.logging.error(f"Invalid database URL format in {config_file}")
                    return None
            else:
                bt.logging.warning(f"No db_ptn_editor_url found in {config_file}")
                return None
        else:
            bt.logging.debug(f"Config file {config_file} not found in working directory")
            return None
            
    except json.JSONDecodeError as e:
        bt.logging.error(f"Failed to parse {config_file}: {e}")
        return None
    except Exception as e:
        bt.logging.error(f"Error reading {config_file}: {e}")
        return None


class EliminationTracker:
    """Handles elimination checking and filtering for miners."""
    
    def __init__(self, elimination_source_type: EliminationSource = EliminationSource.DATABASE):
        """Initialize elimination tracker with specified source."""
        self.elimination_source_manager = EliminationSourceManager(source_type=elimination_source_type)
        self.eliminations_by_hotkey = {}
        self.elimination_timestamps = {}  # hotkey -> elimination_time_ms
        self.loaded = False
        
    def load_all_eliminations(self, hotkeys: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Load all eliminations for the specified hotkeys (or all if None).
        Returns a dict mapping hotkeys to their elimination timestamps.
        """
        bt.logging.info("Loading elimination data for filtering...")
        
        try:
            # Load eliminations over all time (no time restrictions)
            self.eliminations_by_hotkey = self.elimination_source_manager.load_eliminations(
                start_time_ms=None,  # All time
                end_time_ms=None,    # All time
                hotkeys=hotkeys
            )
            
            # Create fast lookup: hotkey -> earliest elimination timestamp
            self.elimination_timestamps = {}
            
            for hotkey, eliminations in self.eliminations_by_hotkey.items():
                if eliminations:
                    # Get the earliest elimination time for this miner
                    earliest_elimination_ms = min(
                        elim.get('elimination_time_ms', float('inf')) 
                        for elim in eliminations
                    )
                    if earliest_elimination_ms != float('inf'):
                        self.elimination_timestamps[hotkey] = earliest_elimination_ms
            
            self.loaded = True
            
            # Log summary
            total_eliminations = sum(len(elims) for elims in self.eliminations_by_hotkey.values())
            summary = self.elimination_source_manager.get_eliminations_summary(self.eliminations_by_hotkey)
            
            bt.logging.info(f"Loaded {total_eliminations} elimination records for {len(self.eliminations_by_hotkey)} miners")
            bt.logging.info(f"Elimination summary: {summary}")
            
            if self.elimination_timestamps:
                earliest_elim = min(self.elimination_timestamps.values())
                latest_elim = max(self.elimination_timestamps.values())
                bt.logging.info(f"Elimination time range: "
                               f"{TimeUtil.millis_to_formatted_date_str(earliest_elim)} to "
                               f"{TimeUtil.millis_to_formatted_date_str(latest_elim)}")
            
            return self.elimination_timestamps
            
        except Exception as e:
            bt.logging.error(f"Failed to load eliminations: {e}")
            bt.logging.warning("Continuing without elimination filtering")
            self.elimination_timestamps = {}
            self.loaded = False
            return {}
    
    def is_hotkey_eliminated_at_date(self, hotkey: str, target_date_ms: int) -> bool:
        """
        Check if a hotkey is eliminated at the given date.
        Returns True if eliminated (should be skipped).
        """
        if not self.loaded or hotkey not in self.elimination_timestamps:
            return False  # Not eliminated or no elimination data
        
        elimination_time = self.elimination_timestamps[hotkey]
        return elimination_time <= target_date_ms  # Eliminated if elimination_time <= current_time
    
    def filter_non_eliminated_positions(
        self, 
        positions_by_hotkey: Dict[str, List[Position]], 
        target_date_ms: int
    ) -> Tuple[Dict[str, List[Position]], Dict[str, int]]:
        """
        Filter out positions from eliminated miners at the target date.
        
        Returns:
            Tuple of (filtered_positions_by_hotkey, elimination_stats)
        """
        if not self.loaded:
            # No elimination filtering if not loaded
            return positions_by_hotkey, {"total_hotkeys": len(positions_by_hotkey), "eliminated_hotkeys": 0}
        
        filtered_positions = {}
        elimination_stats = {
            "total_hotkeys": len(positions_by_hotkey),
            "eliminated_hotkeys": 0,
            "eliminated_hotkeys_list": []
        }
        
        for hotkey, positions in positions_by_hotkey.items():
            if self.is_hotkey_eliminated_at_date(hotkey, target_date_ms):
                elimination_stats["eliminated_hotkeys"] += 1
                elimination_stats["eliminated_hotkeys_list"].append(hotkey)
                
                elimination_time = self.elimination_timestamps[hotkey]
                elimination_date = TimeUtil.millis_to_formatted_date_str(elimination_time)
                bt.logging.debug(f"Skipping eliminated miner {hotkey[:16]}... (eliminated {elimination_date})")
            else:
                # Miner not eliminated, include their positions
                filtered_positions[hotkey] = positions
        
        return filtered_positions, elimination_stats
    
    def log_elimination_filtering_stats(self, date_str: str, elimination_stats: Dict[str, int]):
        """Log elimination filtering statistics for a given date."""
        if elimination_stats["eliminated_hotkeys"] > 0:
            bt.logging.info(f"   Elimination Filtering: {elimination_stats['eliminated_hotkeys']}/{elimination_stats['total_hotkeys']} miners eliminated on {date_str}")
            
            # Log details for eliminated miners with their elimination dates
            eliminated_list = elimination_stats.get("eliminated_hotkeys_list", [])
            if eliminated_list:
                # Show first few with elimination details
                sample_eliminated = []
                for i, hotkey in enumerate(eliminated_list[:3]):
                    if hotkey in self.elimination_timestamps:
                        elim_date = TimeUtil.millis_to_formatted_date_str(self.elimination_timestamps[hotkey])
                        sample_eliminated.append(f"{hotkey[:12]}...({elim_date[:10]})")
                    else:
                        sample_eliminated.append(f"{hotkey[:12]}...")
                
                sample_str = ", ".join(sample_eliminated)
                if len(eliminated_list) > 3:
                    sample_str += f" (and {len(eliminated_list) - 3} more)"
                bt.logging.info(f"   Eliminated miners: {sample_str}")
        else:
            bt.logging.info(f"   Elimination Filtering: 0/{elimination_stats['total_hotkeys']} miners eliminated on {date_str}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate daily portfolio returns for miners")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (defaults to first order date)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (defaults to last order date)",
    )
    parser.add_argument(
        "--hotkeys",
        type=str,
        help="Comma-separated list of hotkeys to analyze (defaults to all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (only used with --save-csv)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help="Maximum number of threads for price fetching (default: 5)",
    )
    parser.add_argument(
        "--elimination-source",
        type=str,
        default="DATABASE",
        choices=["DATABASE", "TEST", "DISK"],
        help="Source for elimination data (default: DATABASE)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database connection string (if not provided, will read from config-development.json or secrets)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip dates that already exist in database (default: True)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results to CSV file (in addition to or instead of database)",
    )
    return parser.parse_args()




def main():
    """Main function to calculate daily portfolio returns."""
    args = parse_args()
    
    # Configure logging
    bt.logging.set_trace(args.log_level == "DEBUG")
    bt.logging.set_debug(args.log_level == "DEBUG")
    
    # Parse hotkeys if provided
    hotkeys = None
    if args.hotkeys:
        hotkeys = [h.strip() for h in args.hotkeys.split(",")]
        bt.logging.info(f"Filtering for hotkeys: {hotkeys}")
    
    # Initialize position source manager
    position_source_manager = PositionSourceManager(source_type=PositionSource.DATABASE)
    
    # Load all positions first to determine date bounds
    bt.logging.info("Loading positions from database...")
    all_positions = position_source_manager.load_positions(
        start_time_ms=None,  # Get all positions to determine bounds
        end_time_ms=None,
        hotkeys=hotkeys
    )
    
    if not all_positions:
        bt.logging.error("No positions found in database")
        return
    
    # Initialize elimination tracker
    elimination_source = getattr(EliminationSource, args.elimination_source)
    elimination_tracker = EliminationTracker(elimination_source_type=elimination_source)
    
    # Load all eliminations for filtering
    elimination_timestamps = elimination_tracker.load_all_eliminations(hotkeys=hotkeys)
    
    # Log elimination loading summary
    if elimination_timestamps:
        bt.logging.info(f"✓ Elimination filtering enabled: {len(elimination_timestamps)} miners have eliminations on record")
    else:
        bt.logging.info("ℹ Elimination filtering disabled: No elimination data available or failed to load")
    
    # Determine date bounds dynamically if not provided
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
    else:
        # Determine bounds from positions
        start_ms, end_ms = DateUtils.determine_date_bounds(all_positions)
        
        # Override with user-provided dates if any
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_ms = int(start_date.timestamp() * 1000)
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_ms = int(end_date.timestamp() * 1000)
    
    bt.logging.info(f"Calculating returns from {TimeUtil.millis_to_formatted_date_str(start_ms)} "
                    f"to {TimeUtil.millis_to_formatted_date_str(end_ms)}")
    
    # Initialize database manager (default behavior)
    db_manager = None
    existing_dates = set()
    
    try:
        # Get database connection string
        database_url = args.database_url
        if not database_url:
            # Try to get from config-development.json
            database_url = get_database_url_from_config()
            
        if not database_url:
            # Fallback: try to get from ValiUtils secrets
            try:
                secrets = ValiUtils.get_secrets()
                database_url = secrets.get('database_url') or secrets.get('db_ptn_editor_url')
            except Exception as e:
                bt.logging.debug(f"Could not get database URL from ValiUtils secrets: {e}")
        
        if database_url:
            bt.logging.info("Initializing database connection...")
            db_manager = DatabaseManager(database_url)
            
            if args.skip_existing:
                # Get existing dates in batch upfront
                start_date_str = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                end_date_str = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                existing_dates = db_manager.get_existing_dates(start_date_str, end_date_str)
                
                if existing_dates:
                    bt.logging.info(f"Will skip {len(existing_dates)} existing dates: {sorted(list(existing_dates))[:5]}{'...' if len(existing_dates) > 5 else ''}")
        else:
            if not args.save_csv:
                bt.logging.error("No database URL available and --save-csv not specified")
                bt.logging.error("Either provide --database-url or use --save-csv flag")
                return
            bt.logging.warning("No database URL provided, will save to CSV only")
    except Exception as e:
        bt.logging.error(f"Failed to initialize database: {e}")
        if not args.save_csv:
            bt.logging.error("Database initialization failed and --save-csv not specified")
            bt.logging.error("Use --save-csv flag to save results to file instead")
            return
        bt.logging.warning("Database failed, continuing with CSV output")

    # Initialize live price fetcher
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)

    # Results storage (only used if CSV output is requested)
    results = [] if args.save_csv else None
    
    # Calculate total number of days to process
    total_days = int((end_ms - start_ms) // MS_IN_24_HOURS) + 1
    
    # Step through each day
    current_ms = start_ms
    day_counter = 0
    while current_ms <= end_ms:
        day_start_time = time.time()
        
        day_counter += 1
        current_date = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc)
        current_date_utc_str = current_date.strftime('%Y-%m-%d %H:%M:%S UTC')
        current_date_short = current_date.strftime('%Y-%m-%d')
        
        bt.logging.info(f"=== Processing date: {current_date_short} ({current_date_utc_str}) [{day_counter}/{total_days}] ===")
        
        # Check if this date already exists in database
        if current_date_short in existing_dates:
            bt.logging.info(f"⏭️  Skipping {current_date_short} - already exists in database")
            current_ms += MS_IN_24_HOURS
            continue
        
        try:
            # Step 1: Filter positions and identify trade pairs needing prices in single pass
            filtered_positions_by_hotkey, trade_pairs_needing_prices, skip_stats = PositionFilter.filter_and_analyze_positions(
                all_positions, current_ms
            )
            
            # Log filtering statistics
            Logger.log_filtering_stats(current_date_short, skip_stats)
            
            if not filtered_positions_by_hotkey:
                day_elapsed_time = time.time() - day_start_time
                bt.logging.info(f"No valid positions remaining after filtering on {current_date_short} UTC, skipping... (took {day_elapsed_time:.2f}s)")
                current_ms += MS_IN_24_HOURS
                continue
            
            # Step 1.5: Filter out eliminated miners
            filtered_positions_by_hotkey, elimination_stats = elimination_tracker.filter_non_eliminated_positions(
                filtered_positions_by_hotkey, current_ms
            )
            
            # Log elimination filtering statistics
            elimination_tracker.log_elimination_filtering_stats(current_date_short, elimination_stats)
            
            if not filtered_positions_by_hotkey:
                day_elapsed_time = time.time() - day_start_time
                bt.logging.info(f"No valid positions remaining after elimination filtering on {current_date_short} UTC, skipping... (took {day_elapsed_time:.2f}s)")
                current_ms += MS_IN_24_HOURS
                continue
            
            bt.logging.info(f"Found {len(filtered_positions_by_hotkey)} active miners with {sum(len(positions) for positions in filtered_positions_by_hotkey.values())} valid positions on {current_date_short} UTC")
            
            if not trade_pairs_needing_prices:
                bt.logging.info(f"No open positions needing prices on {current_date_short} UTC, using closed position returns...")
            else:
                bt.logging.info(f"Need to fetch prices for {len(trade_pairs_needing_prices)} trade pairs: {[tp.trade_pair for tp in trade_pairs_needing_prices]}")
            
            # Step 2: Multi-threaded price fetching for all required trade pairs
            price_fetcher = PriceFetcher(live_price_fetcher, max_workers=args.max_workers)
            try:
                cached_price_sources = price_fetcher.fetch_multiple_price_sources(
                    trade_pairs_needing_prices, current_ms
                )
                bt.logging.info(f"✓ Successfully completed price fetching for {current_date_short} UTC")
                
            except Exception as e:
                day_elapsed_time = time.time() - day_start_time
                bt.logging.error(f"✗ CRITICAL ERROR: Failed to fetch prices for {current_date_short} UTC: {e} (took {day_elapsed_time:.2f}s)")
                bt.logging.error("Stopping script execution due to price fetching failure")
                raise RuntimeError(f"Price fetching failed for date {current_date_short}: {e}") from e
            
            # Step 3: Calculate returns for each miner using categorized approach
            # Flatten all positions for category-based calculation
            all_filtered_positions = []
            for positions in filtered_positions_by_hotkey.values():
                all_filtered_positions.extend(positions)
            
            # Calculate returns by category
            miner_category_returns = CategoryReturnCalculator.calculate_miner_returns_by_category(
                all_filtered_positions, current_ms, cached_price_sources
            )
            
            # Prepare insert values for miner_port_values table
            daily_returns = CategoryReturnCalculator.prepare_insert_values(
                miner_category_returns, current_date
            )
            
            # Calculate stats for logging (using old method for compatibility)
            daily_stats = DailyStats(skip_stats=skip_stats, elimination_stats=elimination_stats)
            
            for hotkey, categories in miner_category_returns.items():
                try:
                    # Get overall portfolio return
                    portfolio_return = categories.get("all", {}).get("return", 1.0)
                    num_positions = categories.get("all", {}).get("count", 0)
                    
                    # Collect position statistics for each miner
                    if hotkey in filtered_positions_by_hotkey:
                        positions = filtered_positions_by_hotkey[hotkey]
                        position_stats = PositionAnalyzer.analyze_positions_for_date(
                            positions, current_ms, cached_price_sources
                        )
                        daily_stats.total_positions += len(positions)
                        daily_stats.open_positions += position_stats['open_positions']
                        daily_stats.closed_positions += position_stats['closed_positions']
                        daily_stats.position_returns.extend(position_stats['returns'])
                        
                        # Track trade pair usage
                        for pos in positions:
                            daily_stats.trade_pair_usage[pos.trade_pair.trade_pair] += 1
                    
                    # Track portfolio return
                    daily_stats.portfolio_returns.append(portfolio_return)
                    daily_stats.successful_miners += 1
                    
                    # Track extreme returns
                    return_pct = (portfolio_return - 1.0) * 100
                    if daily_stats.extreme_returns['best'] is None or return_pct > daily_stats.extreme_returns['best'][1]:
                        daily_stats.extreme_returns['best'] = (hotkey, return_pct)
                    if daily_stats.extreme_returns['worst'] is None or return_pct < daily_stats.extreme_returns['worst'][1]:
                        daily_stats.extreme_returns['worst'] = (hotkey, return_pct)
                    
                except Exception as e:
                    bt.logging.error(f"Failed to calculate return for {hotkey} on {current_date_short} UTC: {e}")
                    daily_stats.failed_miners += 1
                    continue
            
            # Save results based on configuration
            if db_manager and daily_returns:
                # Primary: Insert into database
                success = db_manager.insert_daily_returns(daily_returns, current_date)
                if success:
                    bt.logging.info(f"💾 Successfully saved {len(daily_returns)} returns to database for {current_date_short}")
                else:
                    bt.logging.error(f"Failed to save returns to database for {current_date_short}")
                    raise RuntimeError(f"Database insertion failed for {current_date_short}")
                
                # Also save to CSV if requested (convert back to simple format for CSV compatibility)
                if args.save_csv and results is not None:
                    for miner_data in daily_returns:
                        results.append({
                            "date": miner_data["date"],
                            "hotkey": miner_data["miner_hotkey"],
                            "portfolio_return": miner_data["all_port_value"],
                            "return_pct": (miner_data["all_port_value"] - 1.0) * 100,
                            "num_positions": miner_data["all_count"],
                        })
                    
            elif args.save_csv and results is not None:
                # CSV only mode (convert to simple format)
                for miner_data in daily_returns:
                    results.append({
                        "date": miner_data["date"],
                        "hotkey": miner_data["miner_hotkey"],
                        "portfolio_return": miner_data["all_port_value"],
                        "return_pct": (miner_data["all_port_value"] - 1.0) * 100,
                        "num_positions": miner_data["all_count"],
                    })
            else:
                # No valid output method
                bt.logging.error(f"No valid output method configured for {current_date_short}")
                raise RuntimeError("No database connection and CSV output not enabled")
            
            # Log comprehensive daily summary
            Logger.log_daily_summary(current_date_short, daily_stats)
            
            # Log day processing time
            day_elapsed_time = time.time() - day_start_time
            bt.logging.info(f"✓ Completed processing {current_date_short} UTC (took {day_elapsed_time:.2f}s)\n")
            
        except Exception as e:
            day_elapsed_time = time.time() - day_start_time
            bt.logging.error(f"✗ CRITICAL ERROR: Failed to process date {current_date_short} UTC: {e} (took {day_elapsed_time:.2f}s)")
            bt.logging.error("Stopping script execution due to date processing failure")
            raise RuntimeError(f"Date processing failed for {current_date_short}: {e}") from e
        
        # Move to next day
        current_ms += MS_IN_24_HOURS
    
    # Handle CSV output if requested
    if args.save_csv and results:
        output_file = args.output or "daily_portfolio_returns.csv"
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        if db_manager:
            bt.logging.info(f"📄 CSV output also saved to {output_file} ({len(results)} records)")
        else:
            bt.logging.info(f"📄 Results saved to {output_file} ({len(results)} records)")
        
        # Print comprehensive final summary for CSV output
        Logger.log_final_summary(df, start_ms, end_ms)
    elif db_manager:
        bt.logging.info("✅ All daily returns successfully saved to database")
    else:
        bt.logging.warning("No output method was used")
    
    # Final summary
    if db_manager:
        processed_days = day_counter - len(existing_dates)
        bt.logging.info("=" * 80)
        bt.logging.info("📊 DATABASE INSERTION SUMMARY")
        bt.logging.info("=" * 80)
        bt.logging.info(f"📅 Total days in range: {total_days}")
        bt.logging.info(f"⏭️  Skipped existing: {len(existing_dates)}")
        bt.logging.info(f"✅ Processed new days: {processed_days}")
        bt.logging.info(f"💾 Database: daily_portfolio_returns table updated")
        bt.logging.info("=" * 80)




if __name__ == "__main__":
    bt.logging.enable_info()
    main()