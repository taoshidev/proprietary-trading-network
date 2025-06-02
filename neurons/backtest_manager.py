import time
from collections import defaultdict

from runnable.generate_request_minerstatistics import MinerStatisticsManager
from shared_objects.sn8_multiprocessing import get_spark_session, get_multiprocessing_pool
from tests.shared_objects.mock_classes import MockMetagraph
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
import vali_objects.position as position_file
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.live_price_fetcher import LivePriceFetcher
from vali_objects.utils.position_lock import PositionLocks
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, ParallelizationMode

from vali_objects.utils.price_slippage_model import PriceSlippageModel

class BacktestManager:

    def __init__(self, positions_at_t_f, start_time_ms, secrets, scoring_func,
                 capital=ValiConfig.CAPITAL, use_slippage=None,
                 fetch_slippage_data=False, recalculate_slippage=False, rebuild_all_positions=False,
                 parallel_mode=ParallelizationMode.PYSPARK, build_portfolio_ledgers_only=False,
                 pool_size=0, target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS):
        if not secrets:
            raise Exception(
                "unable to get secrets data from "
                "validation/miner_secrets.json. Please ensure it exists"
            )
        self.secrets = secrets
        self.scoring_func = scoring_func
        self.start_time_ms = start_time_ms
        self.parallel_mode = parallel_mode
        # Used in calculating position attributes
        position_file.ALWAYS_USE_SLIPPAGE = use_slippage

        # Stop Spark session if we created it
        spark, should_close = get_spark_session(self.parallel_mode)
        pool = get_multiprocessing_pool(self.parallel_mode, pool_size)
        self.spark = spark
        self.pool = pool
        self.should_close = should_close
        self.target_ledger_window_ms = target_ledger_window_ms

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        # IMPORTANT: Only update this variable in-place. Otherwise, the reference will be lost in the helper classes.
        self.metagraph = MockMetagraph(hotkeys=list(positions_at_t_f.keys()))
        shutdown_dict = {}

        self.elimination_manager = EliminationManager(self.metagraph, None,  # Set after self.pm creation
                                                      None, shutdown_dict=shutdown_dict, is_backtesting=True)

        self.live_price_fetcher = LivePriceFetcher(secrets=self.secrets, disable_ws=True, is_backtesting=True)

        self.perf_ledger_manager = PerfLedgerManager(self.metagraph,
                                                     shutdown_dict=shutdown_dict,
                                                     live_price_fetcher=None, # Don't want SSL objects to be pickled
                                                     is_backtesting=True,
                                                     position_manager=None,
                                                     parallel_mode=parallel_mode,
                                                     secrets=self.secrets,
                                                     build_portfolio_ledgers_only=build_portfolio_ledgers_only,
                                                     target_ledger_window_ms=target_ledger_window_ms)


        self.position_manager = PositionManager(metagraph=self.metagraph,
                                                perf_ledger_manager=self.perf_ledger_manager,
                                                elimination_manager=self.elimination_manager,
                                                is_backtesting=True,
                                                challengeperiod_manager=None)


        self.challengeperiod_manager = ChallengePeriodManager(self.metagraph,
                                                              perf_ledger_manager=self.perf_ledger_manager,
                                                              position_manager=self.position_manager,
                                                              is_backtesting=True)

        # Attach the position manager to the other objects that need it
        for idx, obj in enumerate([self.perf_ledger_manager, self.position_manager, self.elimination_manager]):
            obj.position_manager = self.position_manager

        self.position_manager.challengeperiod_manager = self.challengeperiod_manager

        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager
        self.position_manager.perf_ledger_manager = self.perf_ledger_manager

        self.weight_setter = SubtensorWeightSetter(self.metagraph, position_manager=self.position_manager, is_backtesting=True)
        self.position_locks = PositionLocks(hotkey_to_positions=positions_at_t_f, is_backtesting=True)
        self.plagiarism_detector = PlagiarismDetector(self.metagraph)
        self.miner_statistics_manager = MinerStatisticsManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector
        )
        self.psm = PriceSlippageModel(self.live_price_fetcher, is_backtesting=True, fetch_slippage_data=fetch_slippage_data,
                                      recalculate_slippage=recalculate_slippage, capital=capital)


        #Until slippage is added to the db, this will always have to be done since positions are sometimes rebuilt and would require slippage attributes on orders and initial_entry_price calculation
        self.psm.update_historical_slippage(positions_at_t_f)

        self.init_order_queue_and_current_positions(self.start_time_ms, positions_at_t_f, rebuild_all_positions=rebuild_all_positions)


    def update_current_hk_to_positions(self, cutoff_ms):
        #cutoff_ms_formatted = TimeUtil.millis_to_formatted_date_str(cutoff_ms)
        #current_oq = [TimeUtil.millis_to_formatted_date_str(o[0].processed_ms) for o in self.order_queue]
        #print(f'current_oq {current_oq}')
        while self.order_queue and self.order_queue[-1][0].processed_ms <= cutoff_ms:
            time_formatted = TimeUtil.millis_to_formatted_date_str(self.order_queue[-1][0].processed_ms)
            order, position = self.order_queue.pop()
            existing_positions = [p for p in self.position_manager.get_positions_for_one_hotkey(position.miner_hotkey)
                                  if p.position_uuid == position.position_uuid]
            assert len(existing_positions) <= 1, f"Found multiple positions with the same UUID: {existing_positions}"
            existing_position = existing_positions[0] if existing_positions else None
            if existing_position:
                print(f'OQU: Added order to existing position {position.trade_pair.trade_pair_id} at {time_formatted}')
                existing_position.orders.append(order)
                existing_position.rebuild_position_with_updated_orders()
                self.position_manager.save_miner_position(existing_position)
            else:  # first order. position must be inserted into list
                print(f'OQU: Created new position {position.trade_pair.trade_pair_id} at {time_formatted} for hk {position.miner_hotkey}')
                position.orders = [order]
                position.rebuild_position_with_updated_orders()
                self.position_manager.save_miner_position(position)

    def init_order_queue_and_current_positions(self, cutoff_ms, positions_at_t_f, rebuild_all_positions=False):
        self.order_queue = []  # (order, position)
        for hk, positions in positions_at_t_f.items():
            for position in positions:
                if all(o.processed_ms <= cutoff_ms for o in position.orders):
                    if rebuild_all_positions:
                        position.rebuild_position_with_updated_orders()
                    self.position_manager.save_miner_position(position)
                    continue
                orders_to_keep = []
                for order in position.orders:
                    if order.processed_ms <= cutoff_ms:
                        orders_to_keep.append(order)
                    else:
                        self.order_queue.append((order, position))
                if orders_to_keep:
                    if len(orders_to_keep) != len(position.orders):
                        position.orders = orders_to_keep
                        position.rebuild_position_with_updated_orders()
                    self.position_manager.save_miner_position(position)

        self.order_queue.sort(key=lambda x: x[0].processed_ms, reverse=True)
        current_hk_to_positions = self.position_manager.get_positions_for_all_miners()
        print(f'Order queue size: {len(self.order_queue)},'
              f' Current positions n hotkeys: {len(current_hk_to_positions)},'
              f' Current positions n total: {sum(len(v) for v in current_hk_to_positions.values())}')

    def update(self, current_time_ms:int, run_challenge=True, run_elimination=True):
        self.update_current_hk_to_positions(current_time_ms)

        if self.parallel_mode == ParallelizationMode.SERIAL:
            self.perf_ledger_manager.update(t_ms=current_time_ms)
        else:
            existing_perf_ledgers = self.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
            # Get positions and existing ledgers
            hotkey_to_positions, _ = self.perf_ledger_manager.get_positions_perf_ledger()

            # Run the parallel update
            updated_perf_ledgers = self.perf_ledger_manager.update_perf_ledgers_parallel(self.spark, self.pool,
                 hotkey_to_positions, existing_perf_ledgers, parallel_mode=self.parallel_mode, now_ms=current_time_ms, is_backtesting=True)

            PerfLedgerManager.print_bundles(updated_perf_ledgers)
        if run_challenge:
            self.challengeperiod_manager.refresh(current_time=current_time_ms)
        else:
            self.challengeperiod_manager.add_all_miners_to_success(current_time_ms=current_time_ms, run_elimination=run_elimination)
        if run_elimination:
            self.elimination_manager.process_eliminations(self.position_locks)
        self.weight_setter.set_weights(None, None, None, current_time=current_time_ms)


if __name__ == '__main__':
    test_positions = [
        {'miner_hotkey': '5DaW56UxJ9Dk14mvraGSEZhy1c91WyLuT2JnNrnKrwnzmZxk',
         'position_uuid': '51c02f65-ff69-3180-e035-524d01f178fe', 'open_ms': 1738199964405,
         'trade_pair': TradePair.NZDJPY, 'orders': [
            {'trade_pair': TradePair.NZDJPY, 'order_type': OrderType.SHORT, 'leverage': -5.0, 'price': 87.446,
             'processed_ms': 1738199964405, 'order_uuid': '51c02f65-ff69-3180-e035-524d01f178fe',
             'price_sources': [
                 {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 87.446, 'close': 87.446, 'vwap': 87.446,
                  'high': 87.446, 'low': 87.446, 'start_ms': 1738199964000, 'websocket': True, 'lag_ms': 405,
                  'volume': 1.0},
                 {'source': 'Tiingo_rest', 'timespan_ms': 0, 'open': 87.443, 'close': 87.443, 'vwap': 87.443,
                  'high': 87.443, 'low': 87.443, 'start_ms': 1738199966615, 'websocket': True, 'lag_ms': 2210,
                  'volume': None}], 'src': 0},
            {'trade_pair': TradePair.NZDJPY, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 87.419,
             'processed_ms': 1738203392320, 'order_uuid': '4463ff0f-beb2-7a24-468d-cb4401647f70',
             'price_sources': [
                 {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 87.419, 'close': 87.419, 'vwap': 87.419,
                  'high': 87.419, 'low': 87.419, 'start_ms': 1738203392000, 'websocket': True, 'lag_ms': 320,
                  'volume': 1.0},
                 {'source': 'Tiingo_rest', 'timespan_ms': 0, 'open': 87.419, 'close': 87.419, 'vwap': 87.419,
                  'high': 87.419, 'low': 87.419, 'start_ms': 1738203393228, 'websocket': True, 'lag_ms': 908,
                  'volume': None}], 'src': 0}], 'current_return': 1.001543809894106, 'close_ms': 1738203392320,
         'return_at_close': 1.0011932695606431, 'net_leverage': 0.0, 'average_entry_price': 87.446,
         'position_type': OrderType.FLAT, 'is_closed_position': True},

        {'miner_hotkey': '5F7RwQwCK2NCZaiUWoSnvH4G5QF99FSuhrXwZbsFB8aW3Ft4',
         'position_uuid': '72170a0a-07cd-4554-899c-7df31dc0eb32', 'open_ms': 1738085801452,
         'trade_pair': TradePair.ETHUSD, 'orders': [
            {'trade_pair': TradePair.ETHUSD, 'order_type': OrderType.LONG, 'leverage': 0.02, 'price': 3174.78,
             'processed_ms': 1738085801452, 'order_uuid': 'd677bb77-2d52-4091-946c-0c30a929702d', 'price_sources': [
                {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 3174.78, 'close': 3174.78, 'vwap': 3174.78,
                 'high': 3174.78, 'low': 3174.78, 'start_ms': 1738085802000, 'websocket': True, 'lag_ms': 548,
                 'volume': 0.01436904},
                {'source': 'Tiingo_gdax_rest', 'timespan_ms': 0, 'open': 3174.49, 'close': 3174.49, 'vwap': 3174.49,
                 'high': 3174.49, 'low': 3174.49, 'start_ms': 1738085803198, 'websocket': True, 'lag_ms': 1746,
                 'volume': None},
                {'source': 'Polygon_rest', 'timespan_ms': 1000, 'open': 3174.8, 'close': 3174.7, 'vwap': 3174.5739,
                 'high': 3174.8, 'low': 3174.44, 'start_ms': 1738085798000, 'websocket': False, 'lag_ms': 2453,
                 'volume': 0.09203032}], 'src': 0}], 'current_return': 1.0001816188838282, 'close_ms': 0,
         'return_at_close': 1.000161024239802, 'net_leverage': 0.02, 'average_entry_price': 3174.78,
         'position_type': OrderType.LONG, 'is_closed_position': False},

        {'miner_hotkey': '5FkMNsY29L9BFbk68RWrPHvQys2L9JKdm9Fua6LTEt9gMPvw',
         'position_uuid': '6f764448-c805-48fe-9090-28a6ff7aab7c', 'open_ms': 1737989160817,
         'trade_pair': TradePair.ETHUSD, 'orders': [
            {'trade_pair': TradePair.ETHUSD, 'order_type': OrderType.LONG, 'leverage': 0.05, 'price': 3128.73,
             'processed_ms': 1737989160817, 'order_uuid': 'd845bf92-bca6-40c2-bfbf-67410a40970b', 'price_sources': [
                {'source': 'Tiingo_gdax_rest', 'timespan_ms': 0, 'open': 3128.73, 'close': 3128.73, 'vwap': 3128.73,
                 'high': 3128.73, 'low': 3128.73, 'start_ms': 1737989160117, 'websocket': True, 'lag_ms': 700,
                 'volume': None}, {'source': 'Polygon_rest', 'timespan_ms': 1000, 'open': 3127.42, 'close': 3127.42,
                                   'vwap': 3127.4146, 'high': 3127.42, 'low': 3126.8, 'start_ms': 1737989157000,
                                   'websocket': False, 'lag_ms': 2818, 'volume': 11.5375611},
                {'source': 'Polygon_ws', 'timespan_ms': 0, 'open': 3103.18, 'close': 3103.18, 'vwap': 3103.18,
                 'high': 3103.18, 'low': 3103.18, 'start_ms': 1737988201000, 'websocket': True, 'lag_ms': 959817,
                 'volume': 0.06100116}], 'src': 0}], 'current_return': 1.001196651676559, 'close_ms': 0,
         'return_at_close': 1.0011292087584787, 'net_leverage': 0.05, 'average_entry_price': 3128.7300000000005,
         'position_type': OrderType.LONG, 'is_closed_position': False},

        {'miner_hotkey': '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx',
         'position_uuid': 'ad0bec72-7727-49fa-8007-7079514432a5', 'open_ms': 1737364369013,
         'trade_pair': TradePair.GBPUSD, 'orders': [
            {'trade_pair': TradePair.GBPUSD, 'order_type': OrderType.SHORT, 'leverage': -0.25, 'price': 1.218715,
             'processed_ms': 1737364369013, 'order_uuid': 'ad0bec72-7727-49fa-8007-7079514432a5',
             'price_sources': [], 'src': 0},
            {'trade_pair': TradePair.GBPUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 0.0,
             'processed_ms': 1738011582117, 'order_uuid': '5a2344159707-7008-af94-7277-27ceb0da',
             'price_sources': [], 'src': 1}], 'current_return': 1.0, 'close_ms': 1738011582117,
         'return_at_close': 0.9999825, 'net_leverage': 0.0, 'average_entry_price': 1.218715,
         'position_type': OrderType.FLAT, 'is_closed_position': True},

        {'miner_hotkey': '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx',
         'position_uuid': '4f6ebe79-0204-45a1-9966-d766a77f6143', 'open_ms': 1737364362020,
         'trade_pair': TradePair.NZDUSD, 'orders': [
            {'trade_pair': TradePair.NZDUSD, 'order_type': OrderType.SHORT, 'leverage': -0.25, 'price': 0.56035,
             'processed_ms': 1737364362020, 'order_uuid': '4f6ebe79-0204-45a1-9966-d766a77f6143',
             'price_sources': [], 'src': 0},
            {'trade_pair': TradePair.NZDUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 0.0,
             'processed_ms': 1738011582117, 'order_uuid': '3416f77a667d-6699-1a54-4020-97ebe6f4',
             'price_sources': [], 'src': 1}], 'current_return': 1.0, 'close_ms': 1738011582117,
         'return_at_close': 0.9999825, 'net_leverage': 0.0, 'average_entry_price': 0.56035,
         'position_type': OrderType.FLAT, 'is_closed_position': True},

        {'miner_hotkey': '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx',
         'position_uuid': '5fcfe64b-b2f0-4060-addc-6379f44cf53c', 'open_ms': 1736151925629,
         'trade_pair': TradePair.EURUSD, 'orders': [
            {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.SHORT, 'leverage': -0.5, 'price': 1.03342,
             'processed_ms': 1736151925629, 'order_uuid': '5fcfe64b-b2f0-4060-addc-6379f44cf53c',
             'price_sources': [], 'src': 0},
            {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.SHORT, 'leverage': -0.25, 'price': 1.0292,
             'processed_ms': 1737102024911, 'order_uuid': '3c5512b5-741c-4d38-9656-6b50f48bfe24',
             'price_sources': [], 'src': 0},
            {'trade_pair': TradePair.EURUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 0.0,
             'processed_ms': 1738011582117, 'order_uuid': 'c35fc44f9736-cdda-0604-0f2b-b46efcf5',
             'price_sources': [], 'src': 1}], 'current_return': 1.0020417642391284, 'close_ms': 1738011582117,
         'return_at_close': 1.0012786740908333, 'net_leverage': 0.0, 'average_entry_price': 1.0320133333333332,
         'position_type': OrderType.FLAT, 'is_closed_position': True},

        {'miner_hotkey': '5HCJ6okRkmCsu7iLEWotBxgcZy11RhbxSzs8MXT4Dei9osUx',
         'position_uuid': '2a003e6f-2835-4e33-aa3f-ea50e0e1ed08', 'open_ms': 1736843416198,
         'trade_pair': TradePair.AUDUSD, 'orders': [
            {'trade_pair': TradePair.AUDUSD, 'order_type': OrderType.SHORT, 'leverage': -0.4, 'price': 0.62034,
             'processed_ms': 1736843416198, 'order_uuid': '2a003e6f-2835-4e33-aa3f-ea50e0e1ed08',
             'price_sources': [], 'src': 0},
            {'trade_pair': TradePair.AUDUSD, 'order_type': OrderType.FLAT, 'leverage': 0.0, 'price': 0.0,
             'processed_ms': 1738011582117, 'order_uuid': '80de1e0e05ae-f3aa-33e4-5382-f6e300a2',
             'price_sources': [], 'src': 1}], 'current_return': 1.0, 'close_ms': 1738011582117,
         'return_at_close': 0.999972, 'net_leverage': 0.0, 'average_entry_price': 0.62034,
         'position_type': OrderType.FLAT, 'is_closed_position': True}
    ]
    t0 = time.time()
    hk_to_positions = defaultdict(list)
    start_time_ms = min(min(o['processed_ms'] for o in pos['orders']) for pos in test_positions)
    max_order_time_ms = max(max(o['processed_ms'] for o in pos['orders']) for pos in test_positions)
    for pos in test_positions:
        hk_to_positions[pos['miner_hotkey']].append(Position(**pos))

    secrets = ValiUtils.get_secrets()  # {'polygon_apikey': '123', 'tiingo_apikey': '456'}
    btm = BacktestManager(hk_to_positions, start_time_ms, secrets, None, capital=500_000,
                          use_slippage=True, fetch_slippage_data=False, recalculate_slippage=False,
                          parallel_mode=ParallelizationMode.PYSPARK,
                          build_portfolio_ledgers_only=True)
    for t_ms in range(start_time_ms, max_order_time_ms + 1, 1000 * 60 * 60 * 24):
        btm.update(t_ms, run_challenge=False)
        perf_ledger_bundles = btm.perf_ledger_manager.get_perf_ledgers(portfolio_only=False)
        hk_to_perf_ledger_tps = {}
        for k, v in perf_ledger_bundles.items():
            hk_to_perf_ledger_tps[k] = list(v.keys())
        print('hk_to_perf_ledger_tps', hk_to_perf_ledger_tps)
        print('formatted weights', btm.weight_setter.checkpoint_results)

    tf = time.time()
    print(f'Finished in {tf - t0} seconds')
