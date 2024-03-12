import traceback
import time

from data_generator.twelvedata_service import TwelveDataService
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from shared_objects.challenge_utils import ChallengeBase
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

class MDDChecker(ChallengeBase):
    MAX_DAILY_DRAWDOWN = 'MAX_DAILY_DRAWDOWN'
    MAX_TOTAL_DRAWDOWN = 'MAX_TOTAL_DRAWDOWN'
    def __init__(self, config, metagraph):
        super().__init__(config, metagraph)
        secrets = ValiUtils.get_secrets()
        self.all_trade_pairs = [trade_pair for trade_pair in TradePair]
        self.twelvedata = TwelveDataService(api_key=secrets["twelvedata_apikey"])
    
    def mdd_check(self):
        if time.time() - self.get_last_update_time() < ValiConfig.MDD_CHECK_REFRESH_TIME_S:
            time.sleep(1)
            return

        bt.logging.info("running mdd checker")
        self._load_latest_eliminations_from_disk()

        try:
            signal_closing_prices = self.twelvedata.get_closes(trade_pairs=self.all_trade_pairs)
            for trade_pair, closing_price in signal_closing_prices.items():
                bt.logging.info(f"Live price for {trade_pair} is {closing_price}")

            hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
                self.metagraph.hotkeys, sort_positions=True, eliminations=self.eliminations
            )
            bt.logging.info(f"found hotkey positions: {hotkey_positions}")
            for hotkey, sorted_positions in hotkey_positions.items():
                self._search_for_miner_dd_failures(hotkey, sorted_positions, signal_closing_prices)

            self._write_eliminations_from_memory_to_disk()

        except Exception:
            bt.logging.error(traceback.format_exc())

        self.set_last_update_time()

    def _hotkey_in_eliminations(self, hotkey):
        return any(hotkey == x['hotkey'] for x in self.eliminations)

    def _replay_all_closed_positions(self, hotkey, sorted_per_position_return, current_dd):
        elimination_occurred = False
        if len(sorted_per_position_return) == 0:
            bt.logging.info(f"no existing closed positions for [{hotkey}]")
            return elimination_occurred, current_dd

        # Already sorted
        for position_return in sorted_per_position_return:
            current_dd *= position_return
            mdd_failure = self._is_beyond_mdd(current_dd)
            if mdd_failure:
                self.close_positions_and_append_elimination_row(hotkey, current_dd, mdd_failure)
                elimination_occurred = True
                return elimination_occurred, current_dd

        # Replay of closed positions complete. If
        return elimination_occurred, current_dd


    def _search_for_miner_dd_failures(self, hotkey, sorted_positions, signal_closing_prices):
        current_dd = 1
        # Log sorted positions length
        if len(sorted_positions) == 0:
            return
        # Already eliminated
        if self._hotkey_in_eliminations(hotkey):
            return

        sorted_per_position_return = PositionUtils.get_return_per_closed_position(sorted_positions)

        elimination_occurred, current_dd = self._replay_all_closed_positions(hotkey, sorted_per_position_return, current_dd)
        if elimination_occurred:
            return

        bt.logging.info(f"reviewing open positions for [{hotkey}]. Current_dd [{current_dd}]")
        open_positions = [position for position in sorted_positions if not position.is_closed_position]
        open_position_trade_pairs = {
            position.position_uuid: position.trade_pair for position in open_positions
        }

        bt.logging.info(f"number of open positions [{len(open_positions)}]")

        for open_position in open_positions:
            position_closing_price = signal_closing_prices[
                open_position_trade_pairs[open_position.position_uuid]
            ]
            current_return = open_position.calculate_unrealized_pnl(position_closing_price)
            bt.logging.success(f"current return for [{open_position.position_uuid}] is [{current_return}]")
            open_position.current_return = current_return
            current_dd *= current_return

            bt.logging.info(f"updating - current return [{current_return}]")
            bt.logging.info(f"updating - current dd [{current_dd}]")
            bt.logging.info(f"updating - net leverage [{open_position._net_leverage}]")

        mdd_failure = self._is_beyond_mdd(current_dd)
        if mdd_failure:
            self.close_positions_and_append_elimination_row(hotkey, current_dd, mdd_failure)

    def _is_beyond_mdd(self, dd):
        time_now = TimeUtil.generate_start_timestamp(0)
        if (dd < ValiConfig.MAX_DAILY_DRAWDOWN and time_now.hour == 0 and time_now.minute < 5):
            return MDDChecker.MAX_DAILY_DRAWDOWN
        elif (dd < ValiConfig.MAX_TOTAL_DRAWDOWN):
            return MDDChecker.MAX_TOTAL_DRAWDOWN
        else:
            return None





                

