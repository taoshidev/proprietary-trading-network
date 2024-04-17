# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import bittensor as bt
from typing import Optional, List
from pydantic import BaseModel


class PriceSource(BaseModel):
    source: str = 'unknown'
    timespan_ms: int = 0
    open: float = None
    close: float = None
    vwap: Optional[float] = None
    high: float = None
    low: float = None
    start_ms: int = 0
    websocket: bool = False
    lag_ms: int = 0
    volume: Optional[float] = 0.0

    def __eq__(self, other):
        if not isinstance(other, PriceSource):
            return NotImplemented
        return (self.source == other.source and
                self.start_ms == other.start_ms and
                self.timespan_ms == other.timespan_ms and
                self.open == other.open and
                self.close == other.close and
                self.high == other.high and
                self.low == other.low)



    def __hash__(self):
        return hash((self.source,
                    self.start_ms,
                    self.timespan_ms,
                    self.open,
                    self.close,
                    self.high,
                    self.low))

    @property
    def end_ms(self):
        if self.websocket:
            return self.start_ms
        else:
            return self.start_ms + self.timespan_ms

    def time_delta_from_now_ms(self, now_ms: int) -> int:
        if self.websocket:
            return abs(now_ms - self.start_ms)
        else:
            return min(abs(now_ms - self.start_ms),
                       abs(now_ms - self.end_ms))

    def parse_best_price(self, now_ms: int) -> float:
        if self.websocket:
            return self.open
        else:
            if abs(now_ms - self.start_ms) < abs(now_ms - self.end_ms):
                return self.open
            else:
                return self.close

    @staticmethod
    def update_order_with_newest_price_sources(order, candidate_price_sources, hotkey, trade_pair_str):
        orig_price = order.price
        order_time_ms = order.processed_ms
        existing_dict = {ps.source: ps for ps in order.price_sources}
        candidates_dict = {ps.source: ps for ps in candidate_price_sources}
        new_price_sources = []
        # We need to create new price sources. If there is overlap, take the one with the smallest time lag to order_time_ms
        any_changes = False
        for k, candidate_ps in candidates_dict.items():
            if k in existing_dict:
                existing_ps = existing_dict[k]
                if candidate_ps.time_delta_from_now_ms(order_time_ms) < existing_ps.time_delta_from_now_ms(order_time_ms):
                    bt.logging.warning(f"Found a better price source for {hotkey} {trade_pair_str}! Replacing {existing_ps.debug_str(order_time_ms)} with {candidate_ps.debug_str(order_time_ms)}")
                    new_price_sources.append(candidate_ps)
                    any_changes = True
                else:
                    new_price_sources.append(existing_ps)
            else:
                bt.logging.warning(
                    f"Found a new price source for {hotkey} {trade_pair_str}! Adding {candidate_ps.debug_str(order_time_ms)}")
                new_price_sources.append(candidate_ps)
                any_changes = True

        for k, existing_ps in existing_dict.items():
            if k not in candidates_dict:
                new_price_sources.append(existing_ps)

        new_price = PriceSource.get_winning_price(new_price_sources, order_time_ms)
        new_price_sources = PriceSource.non_null_events_sorted(new_price_sources, order_time_ms)
        if any_changes:
            order.price = new_price
            order.price_sources = new_price_sources
            return True
        return False

    @staticmethod
    def get_winning_event(events, now_ms):
        best_event = None
        best_time_delta = float('inf')
        for event in events:
            if event:
                time_delta = event.time_delta_from_now_ms(now_ms)
                if best_event is None or time_delta < best_time_delta:
                    best_event = event
                    best_time_delta = time_delta
        return best_event

    @staticmethod
    def get_winning_price(events, now_ms):
        winner = PriceSource.get_winning_event(events, now_ms)
        return winner.parse_best_price(now_ms) if winner else None

    @staticmethod
    def non_null_events_sorted(events, now_ms):
        ans = sorted(events, key=lambda x: x.time_delta_from_now_ms(now_ms))
        for a in ans:
            a.lag_ms = a.time_delta_from_now_ms(now_ms)
        return ans

    def debug_str(self, time_target_ms):
        return f"(src={self.source} price={self.open} delta_ms={self.time_delta_from_now_ms(time_target_ms)})"


