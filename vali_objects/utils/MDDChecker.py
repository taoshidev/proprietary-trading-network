import shutil
import traceback
import time

from data_generator.twelvedata_service import TwelveDataService
from time_util.time_util import TimeUtil
from vali_config import ValiConfig, TradePair
from vali_objects.utils.challenge_utils import ChallengeBase
from vali_objects.utils.position_utils import PositionUtils
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

class MDDChecker(ChallengeBase):
    def __init__(self, config, metagraph):
        super().__init__(config, metagraph)   
        secrets = ValiUtils.get_secrets()
        self.all_trade_pairs = [trade_pair for trade_pair in TradePair]
        self.twelvedata = TwelveDataService(api_key=secrets["twelvedata_apikey"])     

    def mdd_check(self):
        bt.logging.info("running mdd checker")
        bt.logging.info(f"Subtensor: {self.subtensor}")

        while True:
            try:
                self._handle_eliminations()
            except Exception:
                bt.logging.error(traceback.format_exc())

    def generate_elimination_row(self, hotkey, dd, reason):
        return {'hotkey': hotkey, 'elimination_time': time.now(), 'dd': dd, 'reason': reason}
    
    def _handle_eliminations(self):
        if time.time() - self.last_update_time_s < ValiConfig.MDD_CHECK_REFRESH_TIME_S:
            time.sleep(1)
            return
        
        self._load_eliminations_from_cache()
        bt.logging.debug("checking mdd.")

        try:
            signal_closing_prices = self.twelvedata.get_closes(trade_pairs=self.all_trade_pairs)
            hotkey_positions = PositionUtils.get_all_miner_positions_by_hotkey(
                self.metagraph.hotkeys, sort_positions=True, eliminations=self.eliminations
            )

            for hotkey, positions in hotkey_positions.items():
                current_dd = self._calculate_miner_dd(hotkey, positions, signal_closing_prices)

                mdd_failure = self._is_beyond_mdd(current_dd)
                if mdd_failure:
                    self.eliminations.append(self.generate_elimination_row(hotkey, current_dd, mdd_failure))

            self._write_updated_eliminations(self.eliminations)

        except Exception:
            bt.logging.error(traceback.format_exc())

        self.last_update_time_s = time.time()


    def _calculate_miner_dd(self, hotkey, positions, signal_closing_prices):
        current_dd = 1
        if len(positions) > 0:
            per_position_return = PositionUtils.get_return_per_closed_position(positions)
            bt.logging.debug(f"per position return [{per_position_return}]")

            if len(per_position_return) > 0:
                max_portfolio_return = max(per_position_return)
                max_index = per_position_return.index(max_portfolio_return)

                if max_portfolio_return < current_dd:
                    max_portfolio_return = current_dd

                bt.logging.info(f"max port return for [{hotkey}] is [{max_portfolio_return}]")

                for i, position_return in enumerate(per_position_return):
                    if i > max_index:
                        closed_position_return = position_return / max_portfolio_return
                        mdd_failure = self._is_beyond_mdd(closed_position_return)
                        if mdd_failure:
                            self.eliminations.append(self.generate_elimination_row(hotkey, current_dd, mdd_failure))

                last_position_ind = len(per_position_return) - 1
                if max_index != last_position_ind:
                    current_dd = per_position_return[last_position_ind] / max_portfolio_return
            else:
                bt.logging.debug(f"no existing closed positions for [{hotkey}]")

            eliminated_hotkeys = set(x['hotkey'] for x in self.eliminations)
            if hotkey not in eliminated_hotkeys:
                bt.logging.debug(f"reviewing open positions for [{hotkey}]")
                open_positions = [position for position in positions if not position.is_closed_position]
                open_position_trade_pairs = {
                    position.position_uuid: position.trade_pair for position in open_positions
                }

                bt.logging.debug(f"number of open positions [{len(open_positions)}]")

                for open_position in open_positions:
                    position_closing_price = signal_closing_prices[
                        open_position_trade_pairs[open_position.position_uuid]
                    ]
                    current_return = open_position.calculate_unrealized_pnl(position_closing_price)
                    open_position.current_return = current_return
                    current_dd *= current_return

                    bt.logging.debug(f"updating - current return [{current_return}]")
                    bt.logging.debug(f"updating - current dd [{current_dd}]")
                    bt.logging.debug(f"updating - net leverage [{open_position._net_leverage}]")

        return current_dd

    def _is_beyond_mdd(self, dd):
        time_now = TimeUtil.generate_start_timestamp(0)
        if (dd < ValiConfig.MAX_DAILY_DRAWDOWN and time_now.hour == 0 and time_now.minute < 5):
            return 'MAX_DAILY_DRAWDOWN'
        elif (dd < ValiConfig.MAX_TOTAL_DRAWDOWN):
            return 'MAX_TOTAL_DRAWDOWN'
        else:
            return None
    





                

