import json
import time
from collections import defaultdict
from typing import List

import bittensor as bt
from time_util.time_util import TimeUtil, UnifiedMarketCalendar
from vali_config import TradePair
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker
from vali_objects.vali_dataclasses.price_source import PriceSource

POLYGON_PROVIDER_NAME = "Polygon"
TWELVEDATA_PROVIDER_NAME = "TwelveData"


class BaseDataService():
    def __init__(self, trade_pair_category_to_longest_allowed_lag_s, timespan_to_ms, provider_name):
        self.provider_name = provider_name
        self.n_events_global = 0

        self.trade_pair_to_price_history = defaultdict(list)
        self.closed_market_prices = {tp: None for tp in TradePair}
        self.latest_websocket_events = {}
        self.trade_pair_to_recent_events = defaultdict(RecentEventTracker)
        self.trade_pair_category_to_longest_allowed_lag_s = trade_pair_category_to_longest_allowed_lag_s
        self.timespan_to_ms = timespan_to_ms
        self.trade_pair_lookup = {pair.trade_pair: pair for pair in TradePair}
        self.trade_pair_to_longest_seen_lag_s = {}
        self.market_calendar = UnifiedMarketCalendar()

        for trade_pair in TradePair:
            assert trade_pair.trade_pair_category in self.trade_pair_category_to_longest_allowed_lag_s, \
                f"Trade pair {trade_pair} has no allowed lag time"

    def get_close_rest(
            self,
            trade_pair: TradePair
    ) -> PriceSource | None:
        pass

    def is_market_open(self, trade_pair: TradePair) -> bool:
        return self.market_calendar.is_market_open(trade_pair, TimeUtil.now_in_millis())

    def get_close(self, trade_pair: TradePair) -> PriceSource | None:
        event = self.get_websocket_event(trade_pair)
        if not event:
            bt.logging.info(
                f"Fetching REST close for trade pair {trade_pair.trade_pair} using {self.provider_name} REST: {trade_pair.trade_pair}")
            event = self.get_close_rest(trade_pair)
            bt.logging.info(f"Received {self.provider_name} REST data for {trade_pair.trade_pair}: {event}")
        else:
            bt.logging.info(f"Using {self.provider_name} websocket data for {trade_pair.trade_pair}")
        if not event:
            bt.logging.warning(
                f"Failed to get close for {trade_pair.trade_pair} using {self.provider_name} websocket or REST.")
        return event

    def get_closes_websocket(self, trade_pairs: List[TradePair], trade_pair_to_last_order_time_ms) -> dict[str: PriceSource]:
        events = {}
        for trade_pair in trade_pairs:
            symbol = trade_pair.trade_pair
            if symbol not in self.trade_pair_to_recent_events:
                continue

            # Get the closest aligned event
            time_ms = trade_pair_to_last_order_time_ms[trade_pair]
            symbol = trade_pair.trade_pair
            latest_event = self.trade_pair_to_recent_events[symbol].get_closest_event(time_ms)
            events[trade_pair] = latest_event
            """
            event = self.latest_websocket_events[symbol]
            lag_s = self.get_websocket_lag_for_trade_pair_s(symbol)
            is_stale = lag_s > self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
            if is_stale:
                bt.logging.warning(
                    f"Found stale {self.provider_name} websocket data for {symbol}. Lag: {lag_s} seconds. "
                    f"Max allowed lag for category: "
                    f"{self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]} seconds."
                    f"Ignoring this data.")
            """

        return events

    def get_closes_rest(self, trade_pairs: List[TradePair]) -> dict[str: float]:
        pass

    def get_closes(self, trade_pairs: List[TradePair]) -> dict:
        closes = self.get_closes_websocket(trade_pairs)
        missing_trade_pairs = []
        for tp in trade_pairs:
            if tp not in closes or closes[tp] is None:
                missing_trade_pairs.append(tp)
        if closes:
            debug = {k.trade_pair: v for k, v in closes.items()}
            bt.logging.info(f"Received {self.provider_name} websocket data: {debug}")

        if missing_trade_pairs:
            rest_closes = self.get_closes_rest(missing_trade_pairs)
            debug = {k.trade_pair: v for k, v in rest_closes.items()}
            bt.logging.info(f"Received stale/websocket-less data using {self.provider_name} REST: {debug}")
            closes.update(rest_closes)

        return closes

    def get_websocket_lag_for_trade_pair_s(self, tp: str, now_ms: int) -> float | None:
        cur_event = self.latest_websocket_events.get(tp)
        if cur_event:
            return (now_ms - cur_event.end_ms) / 1000.0
        return None

    def spill_price_history(self):
        # Write the price history to disk in a format that will let us plot it
        filename = f"price_history_{self.provider_name}.json"
        with open(filename, 'w') as f:
            json.dump(self.trade_pair_to_price_history, f)

    def debug_log(self):
        now_ms = TimeUtil.now_in_millis()
        trade_pairs_to_track = list(self.latest_websocket_events.keys())
        for tp in trade_pairs_to_track:
            lag_s = self.get_websocket_lag_for_trade_pair_s(tp, now_ms)
            if tp not in self.trade_pair_to_longest_seen_lag_s:
                self.trade_pair_to_longest_seen_lag_s[tp] = lag_s
            else:
                if lag_s > self.trade_pair_to_longest_seen_lag_s[tp]:
                    self.trade_pair_to_longest_seen_lag_s[tp] = lag_s
        # log how long it has been since the last ping
        formatted_lags = {tp: f"{lag:.2f}" for tp, lag in self.trade_pair_to_longest_seen_lag_s.items()}
        bt.logging.info(f"{self.provider_name} Worst lags seen: {formatted_lags}")
        # Log the last time since websocket ping
        now_ms = TimeUtil.now_in_millis()
        formatted_lags = {tp: f"{(now_ms - price_source.end_ms) / 1000.0:.2f}" for tp, price_source in
                          self.latest_websocket_events.items()}
        bt.logging.info(f"{self.provider_name} Current websocket lags (s): {formatted_lags}")
        # Log the prices
        formatted_prices = {tp: f"{price_source.close:.2f}" for tp, price_source in  # noqa: F841
                            self.latest_websocket_events.items()}
        #bt.logging.info(f"{self.provider_name} Latest websocket prices: {formatted_prices}")
        bt.logging.info(f'{self.provider_name} websocket n_events_global: {self.n_events_global}')
        #if self.provider_name == POLYGON_PROVIDER_NAME:
        #    # Log which trade pairs are likely in closed markets
        #    closed_trade_pairs = {}
        #    for trade_pair in TradePair:
        #        if not self.is_market_open(trade_pair):
        #            closed_trade_pairs[trade_pair.trade_pair] = self.closed_market_prices[trade_pair]
        #
        #    bt.logging.info(f"{self.provider_name} Market closed with closing prices for {closed_trade_pairs}")

    def get_price_before_market_close(self, trade_pair: TradePair) -> float | None:
        pass

    def get_websocket_event(self, trade_pair: TradePair) -> PriceSource | None:
        symbol = trade_pair.trade_pair
        cur_event = self.latest_websocket_events.get(symbol)
        if not cur_event:
            return None

        timestamp_ms = cur_event.end_ms
        max_allowed_lag_s = self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
        lag_s = time.time() - timestamp_ms / 1000.0
        is_stale = lag_s > max_allowed_lag_s
        if is_stale:
            bt.logging.info(f"Found stale TD websocket data for {trade_pair.trade_pair}. Lag_s: {lag_s} "
                            f"seconds. Max allowed lag for category: {max_allowed_lag_s} seconds. Ignoring this data.")
        return cur_event
