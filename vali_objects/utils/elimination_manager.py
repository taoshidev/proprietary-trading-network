# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import shutil
from copy import deepcopy
from typing import Dict
from time_util.time_util import TimeUtil
from vali_objects.position import Position
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource


class EliminationManager(CacheController):
    """"
    We basically want to zero out the weights of the eliminated miners
    for long enough that BT deregisters them. However, there is no guarantee that they get deregistered and
    we may need to handle the case where we allow the miner to participate again. In this case, the elimination
    would already be cleared and their weight would be calculated as normal.
    """

    def __init__(self, metagraph, position_manager, challengeperiod_manager,
                 running_unit_tests=False, shutdown_dict=None, ipc_manager=None, is_backtesting=False):
        super().__init__(metagraph=metagraph, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.shutdown_dict = shutdown_dict
        self.challengeperiod_manager = challengeperiod_manager
        self.running_unit_tests = running_unit_tests
        self.first_refresh_ran = False

        if ipc_manager:
            self.eliminations = ipc_manager.list()
        else:
            self.eliminations = []
        self.eliminations.extend(self.get_eliminations_from_disk())
        if len(self.eliminations) == 0:
            ValiBkpUtils.write_file(
                ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                {CacheController.ELIMINATIONS: []}
            )

    def handle_perf_ledger_eliminations(self, position_locks):
        perf_ledger_eliminations = self.position_manager.perf_ledger_manager.get_perf_ledger_eliminations()
        n_eliminations = 0
        for e in perf_ledger_eliminations:
            if self.hotkey_in_eliminations(e['hotkey']):
                continue

            n_eliminations += 1
            self.eliminations.append(e)
            self.eliminations[-1] = e  # ipc list does not update the object without using __setitem__

            price_info = e['price_info']
            trade_pair_to_price_source_used_for_elimination_check = {}
            for k, v in price_info.items():
                trade_pair = TradePair.get_latest_tade_pair_from_trade_pair_str(k)
                elimination_initiated_time_ms = e['elimination_initiated_time_ms']
                trade_pair_to_price_source_used_for_elimination_check[trade_pair] = PriceSource(source='elim', open=v,
                                                                                                close=v,
                                                                                                start_ms=elimination_initiated_time_ms,
                                                                                                timespan_ms=1000,
                                                                                                websocket=False,
                                                                                                trade_pair=trade_pair)
            self.handle_eliminated_miner(e['hotkey'], trade_pair_to_price_source_used_for_elimination_check, position_locks)

        if n_eliminations:
            self.save_eliminations()
            bt.logging.info(f'Wrote {n_eliminations} perf ledger eliminations to disk')

    def add_manual_flat_order(self, hotkey: str, position: Position, corresponding_elimination, position_locks,
                              source_for_elimination):
        """
        Add flat orders to the positions for a miner that has been eliminated
        """
        elimination_time_ms = corresponding_elimination['elimination_initiated_time_ms'] if corresponding_elimination else TimeUtil.now_in_millis()
        with position_locks.get_lock(hotkey, position.trade_pair.trade_pair_id):
            # Position could have updated in the time between mdd_check being called and this function being called
            position_refreshed = self.position_manager.get_miner_position_by_uuid(hotkey, position.position_uuid)
            if position_refreshed is None:
                bt.logging.warning(
                    f"Unexpectedly could not find position with uuid {position.position_uuid} for hotkey {hotkey} and trade pair {position.trade_pair.trade_pair_id}. Not add flat orders")
                return

            position = position_refreshed
            if position.is_closed_position:
                return

            fake_flat_order_time = elimination_time_ms
            if position.orders and position.orders[-1].processed_ms > elimination_time_ms:
                bt.logging.warning(
                    f'Unexpectedly found a position with a processed_ms {position.orders[-1].processed_ms} greater than the elimination time {elimination_time_ms} ')
                fake_flat_order_time = position.orders[-1].processed_ms + 1

            flat_order = Position.generate_fake_flat_order(position, fake_flat_order_time)
            position.add_order(flat_order)
            if source_for_elimination:
                position.orders[-1].price_sources.append(source_for_elimination)
            self.position_manager.save_miner_position(position, delete_open_position_if_exists=True)
            bt.logging.info(f'Added flat order for miner {hotkey} that has been eliminated. '
                            f'Trade pair: {position.trade_pair.trade_pair_id}. flat order: {flat_order}. '
                            f'position uuid {position.position_uuid}. Source for elimination {source_for_elimination}')

    def handle_eliminated_miner(self, hotkey: str,
                                trade_pair_to_price_source_used_for_elimination_check: Dict[TradePair, PriceSource],
                                position_locks):

        for p in self.position_manager.get_positions_for_one_hotkey(hotkey, only_open_positions=True):
            source_for_elimination = trade_pair_to_price_source_used_for_elimination_check.get(p.trade_pair)
            corresponding_elimination = self.hotkey_in_eliminations(hotkey)
            if corresponding_elimination:
                self.add_manual_flat_order(hotkey, p, corresponding_elimination, position_locks, source_for_elimination)

    def handle_challenge_period_eliminations(self, position_locks):
        eliminations_with_reasons = self.challengeperiod_manager.eliminations_with_reasons
        if not eliminations_with_reasons:
            return

        hotkeys = list(eliminations_with_reasons.keys())
        for hotkey in hotkeys:
            if self.hotkey_in_eliminations(hotkey):
                continue
            elim_reason = eliminations_with_reasons[hotkey][0]
            elim_mdd = eliminations_with_reasons[hotkey][1]
            self.append_elimination_row(hotkey=hotkey, current_dd=elim_mdd, mdd_failure=elim_reason)
            self.handle_eliminated_miner(hotkey, {}, position_locks)

        self.challengeperiod_manager.eliminations_with_reasons = {}

    def handle_first_refresh(self, position_locks):
        if self.is_backtesting or self.first_refresh_ran:
            return

        eliminated_hotkeys = self.get_eliminated_hotkeys()
        hotkey_to_positions = self.position_manager.get_positions_for_hotkeys(eliminated_hotkeys,
                                                                              only_open_positions=True)
        for hotkey, open_positions in hotkey_to_positions.items():
            if not open_positions:
                bt.logging.info(
                    f"Hotkey {hotkey} has been eliminated but has no open positions. Not adding flat orders")
            for p in open_positions:
                self.add_manual_flat_order(hotkey, p, self.hotkey_in_eliminations(hotkey), position_locks, None)

        self.first_refresh_ran = True

    def process_eliminations(self, position_locks):
        if not self.refresh_allowed(ValiConfig.ELIMINATION_CHECK_INTERVAL_MS) and \
                not bool(self.challengeperiod_manager.eliminations_with_reasons):
            return

        bt.logging.info("running elimination manager")
        self.handle_first_refresh(position_locks)
        self.handle_perf_ledger_eliminations(position_locks)
        self.handle_challenge_period_eliminations(position_locks)
        # self._handle_plagiarism_eliminations()
        self.handle_mdd_eliminations(position_locks)
        self._delete_eliminated_expired_miners()

        self.set_last_update_time()

    def _handle_plagiarism_eliminations(self, position_locks):
        bt.logging.debug("checking plagiarism.")
        if self.shutdown_dict:
            return
        self.challengeperiod_manager._refresh_plagiarism_scores_in_memory_and_disk()
        # miner_copying_json[miner_hotkey] = current_hotkey_mc
        for miner_hotkey, current_plagiarism_score in self.challengeperiod_manager.miner_plagiarism_scores.items():
            if self.shutdown_dict:
                return
            if self.hotkey_in_eliminations(miner_hotkey):
                continue
            if current_plagiarism_score > ValiConfig.MAX_MINER_PLAGIARISM_SCORE:
                self.append_elimination_row(miner_hotkey, current_plagiarism_score, 'plagiarism')
                self.handle_eliminated_miner(miner_hotkey, {}, position_locks)

    def is_zombie_hotkey(self, hotkey):
        if hotkey in self.metagraph.hotkeys:
            return False

        if any(x['hotkey'] == hotkey for x in self.eliminations):
            return False

        return True

    def hotkey_in_eliminations(self, hotkey):
        for x in self.eliminations:
            if x['hotkey'] == hotkey:
                return deepcopy(x)
        return None

    def _delete_eliminated_expired_miners(self):
        deleted_hotkeys = set()
        # self.eliminations were just refreshed in process_eliminations
        any_challenege_period_changes = False
        for x in self.eliminations:
            if self.shutdown_dict:
                return
            hotkey = x['hotkey']
            elimination_initiated_time_ms = x['elimination_initiated_time_ms']
            # Don't delete this miner until it hits the minimum elimination time.
            if TimeUtil.now_in_millis() - elimination_initiated_time_ms < ValiConfig.ELIMINATION_FILE_DELETION_DELAY_MS:
                continue
            # We will not delete this miner's cache until it has been deregistered by BT
            if hotkey in self.metagraph.hotkeys:
                bt.logging.trace(f"miner [{hotkey}] has not been deregistered by BT yet. Not deleting miner dir.")
                continue

            # If the miner is no longer in the metagraph, we can remove them from the challengeperiod information
            if hotkey in self.challengeperiod_manager.challengeperiod_testing:
                any_challenege_period_changes = True
                self.challengeperiod_manager.challengeperiod_testing.pop(hotkey)

            if hotkey in self.challengeperiod_manager.challengeperiod_success:
                any_challenege_period_changes = True
                self.challengeperiod_manager.challengeperiod_success.pop(hotkey)

            miner_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests) + hotkey
            all_positions = self.position_manager.get_positions_for_one_hotkey(hotkey)
            for p in all_positions:
                self.position_manager.delete_position(p)
            try:
                shutil.rmtree(miner_dir)
            except FileNotFoundError:
                bt.logging.info(f"miner dir not found. Already deleted. [{miner_dir}]")
            bt.logging.info(
                f"miner eliminated with hotkey [{hotkey}] with max dd of [{x.get('dd', 'N/A')}]. reason: [{x['reason']}]"
                f"Removing miner dir [{miner_dir}]"
            )
            deleted_hotkeys.add(hotkey)

        # Write the challengeperiod information to disk
        if any_challenege_period_changes:
            self.challengeperiod_manager._write_challengeperiod_from_memory_to_disk()

        if deleted_hotkeys:
            self.delete_eliminations(deleted_hotkeys)

        all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=self.running_unit_tests)
        for hotkey in CacheController.get_directory_names(all_miners_dir):
            if self.shutdown_dict:
                return
            miner_dir = all_miners_dir + hotkey
            if self.is_zombie_hotkey(hotkey):
                try:
                    shutil.rmtree(miner_dir)
                    bt.logging.info(f"Zombie miner dir removed [{miner_dir}]")
                except FileNotFoundError:
                    bt.logging.info(f"Zombie miner dir not found. Already deleted. [{miner_dir}]")

    def save_eliminations(self):
        if not self.is_backtesting:
            self.write_eliminations_to_disk(self.eliminations)

    def write_eliminations_to_disk(self, eliminations):
        if not isinstance(eliminations, list):
            eliminations = list(eliminations)  # proxy list
        vali_eliminations = {CacheController.ELIMINATIONS: eliminations}
        bt.logging.trace(f"Writing [{len(eliminations)}] eliminations from memory to disk: {vali_eliminations}")
        output_location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        ValiBkpUtils.write_file(output_location, vali_eliminations)

    def clear_eliminations(self):
        ValiBkpUtils.write_file(ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests),
                                {CacheController.ELIMINATIONS: []})
        del self.eliminations[:]

    def get_eliminated_hotkeys(self):
        return set([x['hotkey'] for x in self.eliminations]) if self.eliminations else set()

    def get_eliminations_from_memory(self):
        return list(self.eliminations)  # ListProxy is not JSON serializable

    def get_eliminations_from_disk(self) -> list:
        location = ValiBkpUtils.get_eliminations_dir(running_unit_tests=self.running_unit_tests)
        cached_eliminations = ValiUtils.get_vali_json_file(location, CacheController.ELIMINATIONS)
        bt.logging.trace(f"Loaded [{len(cached_eliminations)}] eliminations from disk. Dir: {location}")
        return cached_eliminations

    def append_elimination_row(self, hotkey, current_dd, mdd_failure, t_ms=None, price_info=None, return_info=None):
        elimination_row = self.generate_elimination_row(hotkey, current_dd, mdd_failure, t_ms=t_ms,
                                                        price_info=price_info, return_info=return_info)
        self.eliminations.append(elimination_row)
        self.eliminations[-1] = elimination_row  # ipc list does not update the object without using __setitem__
        self.save_eliminations()
        bt.logging.info(f"miner eliminated with hotkey [{hotkey}]. Info [{elimination_row}]")

    def delete_eliminations(self, deleted_hotkeys):
        # with self.eliminations_lock:
        items_to_remove = [x for x in self.eliminations if x['hotkey'] in deleted_hotkeys]
        for item in items_to_remove:
            self.eliminations.remove(item)
        self.save_eliminations()

    def handle_mdd_eliminations(self, position_locks):
        """
        Checks the mdd of each miner and eliminates any miners that surpass MAX_TOTAL_DRAWDOWN
        """
        from vali_objects.utils.ledger_utils import LedgerUtils
        bt.logging.info("checking main competition for maximum drawdown eliminations.")
        if self.shutdown_dict:
            return
        challengeperiod_success_hotkeys = list(self.challengeperiod_manager.get_challengeperiod_success().keys())

        filtered_ledger = self.position_manager.perf_ledger_manager.filtered_ledger_for_scoring(
            hotkeys=challengeperiod_success_hotkeys)
        for miner_hotkey, ledger in filtered_ledger.items():
            if self.shutdown_dict:
                return
            if self.hotkey_in_eliminations(miner_hotkey):
                continue

            miner_exceeds_mdd, drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element=ledger)

            if miner_exceeds_mdd:
                self.append_elimination_row(miner_hotkey, drawdown_percentage, 'MAX_TOTAL_DRAWDOWN')
                self.handle_eliminated_miner(miner_hotkey, {}, position_locks)
