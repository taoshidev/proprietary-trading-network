# developer: Taoshidev
# Copyright © 2024 Taoshi Inc
import bittensor as bt
from typing import Optional
from pydantic import BaseModel

# from vali_objects.utils.price_slippage_model import PriceSlippageModel


class QuoteSource(BaseModel):
    source: str = 'unknown'
    timestamp_ms: int = 0
    bid: float = None
    ask: float = None
    websocket: bool = False
    lag_ms: int = 0

    def __eq__(self, other):
        if not isinstance(other, QuoteSource):
            return NotImplemented
        return (self.source == other.source and
                self.timestamp_ms == other.timestamp_ms and
                self.bid == other.bid and
                self.ask == other.ask)


    def __hash__(self):
        return hash((self.source,
                    self.start_ms,
                    self.timestamp_ms,
                    self.bid,
                    self.ask))

    def get_start_time_ms(self):
        return self.timestamp_ms

    def time_delta_from_now_ms(self, now_ms: int) -> int:
        return abs(now_ms - self.timestamp_ms)

    @staticmethod
    def update_order_with_newest_quote_sources(order, candidate_quote_sources, hotkey, trade_pair_str):
        from vali_objects.utils.price_slippage_model import PriceSlippageModel
        order_time_ms = order.processed_ms
        existing_dict = {ps.source: ps for ps in order.quote_sources}
        candidates_dict = {ps.source: ps for ps in candidate_quote_sources}
        new_quote_sources = []
        # We need to create new price sources. If there is overlap, take the one with the smallest time lag to order_time_ms
        any_changes = False
        for k, candidate_ps in candidates_dict.items():
            if k in existing_dict:
                existing_ps = existing_dict[k]
                if candidate_ps.time_delta_from_now_ms(order_time_ms) < existing_ps.time_delta_from_now_ms(order_time_ms):  # Prefer the ws price in the past rather than the future
                    bt.logging.warning(f"Found a better quote source for {hotkey} {trade_pair_str}! Replacing {existing_ps.debug_str(order_time_ms)} with {candidate_ps.debug_str(order_time_ms)}")
                    new_quote_sources.append(candidate_ps)
                    any_changes = True
                else:
                    new_quote_sources.append(existing_ps)
            else:
                bt.logging.warning(
                    f"Found a new quote source for {hotkey} {trade_pair_str}! Adding {candidate_ps.debug_str(order_time_ms)}")
                new_quote_sources.append(candidate_ps)
                any_changes = True

        for k, existing_ps in existing_dict.items():
            if k not in candidates_dict:
                new_quote_sources.append(existing_ps)

        bid, ask = QuoteSource.get_winning_quote(new_quote_sources, order_time_ms)
        new_slippage = 0#PriceSlippageModel().calculate_slippage(bid, ask, order)
        new_quote_sources = QuoteSource.non_null_events_sorted(new_quote_sources, order_time_ms)
        if any_changes:
            order.quote_sources = new_quote_sources
            order.slippage = new_slippage
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
    def get_winning_quote(events, now_ms):
        winner = QuoteSource.get_winning_event(events, now_ms)
        return winner.bid, winner.ask if winner else None

    @staticmethod
    def non_null_events_sorted(events, now_ms):
        ans = sorted(events, key=lambda x: x.time_delta_from_now_ms(now_ms))
        for a in ans:
            a.lag_ms = a.time_delta_from_now_ms(now_ms)
        return ans

    def debug_str(self, time_target_ms):
        return f"(src={self.source} bid={self.bid} ask={self.ask} delta_ms={self.time_delta_from_now_ms(time_target_ms)})"


