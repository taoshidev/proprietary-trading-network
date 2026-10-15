"""
Standard subaccounts in SUBACCOUNT_FUNDED (created through SUBACCOUNT_CHALLENGE, not HL-linked) are
tracked for auto-promotion with the pro stats: calmar, daily consistency and trading days. Their
max_drawdown is the live pro ratchet combined with the ledger replay of the whole funded period
(LedgerUtils.eod_hwm_drawdown), and a ledger that cannot replay it is rebuilt once.
"""
import pytest

from tests.vali_tests.test_challengeperiod_pro import (  # noqa: F401 - manager is a pytest fixture
    ACCOUNT_SIZE,
    CP_MS,
    DAILY_MS,
    HOTKEY,
    MIDNIGHT_MS,
    NOW_MS,
    _even_ledger,
    manager,
)
from vali_objects.challenge_period.challengeperiod_manager import DrawdownStats, ProStats
from vali_objects.enums.misc import TradePairReturnStatus
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import CollateralRecord, MinerAccount
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.penalty.penalty_ledger import PenaltyInputType, PenaltyLedgerManager
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfCheckpoint, PerfLedger

HL_ADDRESS = "0x" + "d" * 40
DAYS = 60
REALIZED_USD = 7_000.0  # 7% on the account


def _account(hotkey: str = HOTKEY, hl_address: str | None = None,
             asset_class: MinerAssetClass | None = None) -> MinerAccount:
    record = CollateralRecord(account_size=ACCOUNT_SIZE, account_size_theta=ACCOUNT_SIZE / ValiConfig.COST_PER_THETA,
                              update_time_ms=NOW_MS - 100 * DAILY_MS, is_first_record=True)
    return MinerAccount(miner_hotkey=hotkey, collateral_records=[record], hl_address=hl_address,
                        asset_class=asset_class)


def _ledger(low: float = 1.0, close: float = 1.0, days: int = DAYS) -> PerfLedger:
    """DAYS of small gains and REALIZED_USD realized, with every checkpoint at `close` and dipping to `low`."""
    ledger = _even_ledger(days, realized_pnl_usd=REALIZED_USD)
    for cp in ledger.cps:
        cp.equity_ret = close
        cp.min_equity_ret = low
    return ledger


def _seed(manager, hotkey: str = HOTKEY, bucket: MinerBucket = MinerBucket.SUBACCOUNT_FUNDED,
          current_equity: float = 1.0, eod_hwm: float = 1.0) -> None:
    manager.set_miner_bucket(hotkey, bucket, NOW_MS - DAYS * DAILY_MS)
    manager.miner_states[hotkey].drawdown = DrawdownStats(current_equity=current_equity, current_balance=current_equity,
                                                          eod_hwm=eod_hwm, last_eod_equity=1.0,
                                                          last_eod_checked_ms=MIDNIGHT_MS)


def _refresh(manager, ledgers: dict, accounts: dict | None = None) -> None:
    accounts = accounts if accounts is not None else {hk: _account(hk) for hk in ledgers}
    manager._refresh_pro_stats(list(ledgers), ledgers, accounts)


@pytest.fixture(autouse=True)
def _standard_subaccount(manager):
    """The entity record of a subaccount created through SUBACCOUNT_CHALLENGE, and no rebuild pending."""
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {"initial_bucket": None}
    manager._perf_ledger_client.get_perf_ledger_hks_to_invalidate.return_value = {}


# ── LedgerUtils.eod_hwm_drawdown ──────────────────────────────────────────────

def _cp(end_ms: int, low: float | None, close: float) -> PerfCheckpoint:
    return PerfCheckpoint(last_update_ms=end_ms, prev_portfolio_ret=1.0, accum_ms=CP_MS,
                          equity_ret=close, min_equity_ret=low)


DAY0 = MIDNIGHT_MS - 10 * DAILY_MS


def test_an_intraday_peak_never_raises_the_mark():
    """The account ran to 1.05 before noon and fell back to 0.98: 2% under the 1.0 mark, not 6.7%."""
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=0.98, close=1.0)])
    ledger.cps[0].mpv = 1.05
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(0.98)


def test_a_midday_close_never_becomes_the_mark():
    """Closing noon at 1.10 and midnight at 1.00 leaves no drawdown against the 1.0 mark."""
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=1.0, close=1.10),
                             _cp(DAY0 + DAILY_MS, low=1.0, close=1.0),
                             _cp(DAY0 + DAILY_MS + CP_MS, low=1.0, close=1.0)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(1.0)


def test_the_mark_moves_at_midnight_and_governs_the_next_day():
    ledger = PerfLedger(cps=[
        _cp(DAY0 + CP_MS, low=1.00, close=1.05),
        _cp(DAY0 + DAILY_MS, low=1.04, close=1.10),           # closes day 0 at 1.10, still on the 1.0 mark
        _cp(DAY0 + DAILY_MS + CP_MS, low=1.045, close=1.08),  # day 1 runs on the 1.10 mark: 5% under
    ])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(1.045 / 1.10)


def test_a_lower_close_never_lowers_the_mark():
    ledger = PerfLedger(cps=[
        _cp(DAY0 + DAILY_MS, low=1.0, close=1.10),
        _cp(DAY0 + 2 * DAILY_MS, low=1.0, close=1.02),
        _cp(DAY0 + 2 * DAILY_MS + CP_MS, low=1.02, close=1.02),
    ])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(1.0 / 1.10)


def test_the_mark_is_floored_at_the_starting_equity():
    ledger = PerfLedger(cps=[_cp(DAY0 + DAILY_MS, low=0.97, close=0.98),
                             _cp(DAY0 + DAILY_MS + CP_MS, low=0.96, close=0.97)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(0.96)


def test_an_unknown_low_makes_the_replay_unknown():
    ledger = PerfLedger(cps=[_cp(DAY0 + CP_MS, low=None, close=1.0), _cp(DAY0 + DAILY_MS, low=0.99, close=1.0)])
    assert LedgerUtils.eod_hwm_drawdown(ledger) is None


def test_it_replays_the_live_ratchet_over_a_ledger_built_minute_by_minute():
    """Built through update_pl, the replay equals the worst sampled equity over the mark in force when
    it was sampled: the closing equity of every UTC day already over, floored at 1.0."""
    ledger = PerfLedger()
    samples = []
    for minute in range(1, 4 * 24 * 60, 11):
        t_ms = DAY0 + minute * 60_000
        # Drifts up across the days so the mark moves, with dips of up to 4%
        unrealized = (minute / (4 * 24 * 60)) * 12_000 - ((minute * 53) % 400) * 10.0
        ledger.update_pl(1.0 + unrealized / ACCOUNT_SIZE, t_ms, "hk", TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
                         0.0, unrealized, account_size=ACCOUNT_SIZE, cumulative_fees_usd=0.0)
        samples.append((t_ms, (ACCOUNT_SIZE + unrealized) / ACCOUNT_SIZE))

    worst = 1.0
    for t_ms, equity in samples:
        day_start = (t_ms // DAILY_MS) * DAILY_MS
        day_closes = {}
        for s_ms, e in samples:
            if s_ms < day_start:
                day_closes[s_ms // DAILY_MS] = e  # ends as the last sample of each finished day
        worst = min(worst, equity / max([1.0, *day_closes.values()]))

    assert worst < 0.99, "fixture should produce a drawdown"
    assert LedgerUtils.eod_hwm_drawdown(ledger) == pytest.approx(worst)


# ── Auto-promotion candidates get pro stats ──────────────────────────────────

def test_a_candidate_gets_pro_stats(manager):
    """7% realized over a 4% replayed drawdown: calmar 1.75, measured as for pro."""
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})

    stats = manager.miner_states[HOTKEY].pro_stats
    assert stats.max_drawdown == pytest.approx(0.96)
    assert stats.calmar == pytest.approx(1.75)
    assert stats.trading_days == DAYS
    assert stats.daily_consistency == pytest.approx(1 / DAYS)


def test_the_live_ratchet_and_the_ledger_replay_both_count(manager):
    _seed(manager, current_equity=0.95)
    _refresh(manager, {HOTKEY: _ledger(low=0.98)})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.95)

    # Recovered live, but the ledger now replays a deeper dip, and the ratchet never gives back
    manager.miner_states[HOTKEY].drawdown.current_equity = 1.02
    _refresh(manager, {HOTKEY: _ledger(low=0.93)})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.93)
    _refresh(manager, {HOTKEY: _ledger(low=0.99)})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.93)


@pytest.mark.parametrize("record", [
    {"initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value},             # Instant Funded
    {"initial_bucket": MinerBucket.PRO_CHALLENGE_FROM_STANDARD.value},   # Instant Funded
    None,                                                                # no entity record
])
def test_only_subaccounts_created_through_the_challenge_are_candidates(manager, record):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = record
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.miner_states[HOTKEY].pro_stats == ProStats()
    manager._perf_ledger_client.set_hotkey_to_invalidate.assert_not_called()


@pytest.mark.parametrize("account_kwargs", [{"hl_address": HL_ADDRESS}, {"asset_class": MinerAssetClass.HL_ALL}])
def test_hl_linked_subaccounts_are_not_candidates(manager, account_kwargs):
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)}, {HOTKEY: _account(**account_kwargs)})
    assert manager.miner_states[HOTKEY].pro_stats == ProStats()


@pytest.mark.parametrize("bucket", [MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.MAINCOMP])
def test_other_standard_buckets_get_no_pro_stats(manager, bucket):
    _seed(manager, bucket=bucket)
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.miner_states[HOTKEY].pro_stats == ProStats()


@pytest.mark.parametrize("bucket", [MinerBucket.PRO_CHALLENGE_TRANSITION, MinerBucket.PRO_CHALLENGE_FROM_STANDARD,
                                    MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED])
def test_the_pro_track_keeps_the_live_ratchet_alone(manager, bucket):
    _seed(manager, bucket=bucket, current_equity=0.97)
    _refresh(manager, {HOTKEY: _ledger(low=0.50)})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.97)
    manager._perf_ledger_client.set_hotkey_to_invalidate.assert_not_called()


def test_the_creation_record_is_looked_up_once(manager):
    _seed(manager)
    for _ in range(3):
        _refresh(manager, {HOTKEY: _ledger()})
    manager._entity_client.get_subaccount_info_for_synthetic.assert_called_once_with(HOTKEY)


def test_a_failed_lookup_skips_the_account_until_the_next_refresh(manager):
    _seed(manager)
    manager._entity_client.get_subaccount_info_for_synthetic.side_effect = RuntimeError("entity server down")
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.miner_states[HOTKEY].pro_stats == ProStats()

    manager._entity_client.get_subaccount_info_for_synthetic.side_effect = None
    _refresh(manager, {HOTKEY: _ledger(low=0.96)})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.96)


def test_a_candidate_with_no_ledger_yet_keeps_the_live_ratchet(manager):
    _seed(manager, current_equity=0.98)
    _refresh(manager, {HOTKEY: None})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.98)


# ── Rebuilding ledgers that cannot replay the funded period ──────────────────

def _legacy_ledger() -> PerfLedger:
    """Checkpoints started before they recorded their lowest equity."""
    ledger = _ledger(low=0.99)
    ledger.cps[0].min_equity_ret = None
    return ledger


def test_a_ledger_without_every_low_is_rebuilt_once_and_holds_the_live_ratchet(manager):
    _seed(manager, current_equity=0.98)
    _refresh(manager, {HOTKEY: _legacy_ledger()})

    manager._perf_ledger_client.set_hotkey_to_invalidate.assert_called_once_with(HOTKEY, 0)
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.98)

    # Rebuilt but still unknown: never requested again
    _refresh(manager, {HOTKEY: _legacy_ledger()})
    _refresh(manager, {HOTKEY: _legacy_ledger()})
    manager._perf_ledger_client.set_hotkey_to_invalidate.assert_called_once()

    # Once the rebuilt ledger replays the period, it counts
    _refresh(manager, {HOTKEY: _ledger(low=0.94)})
    assert manager.miner_states[HOTKEY].pro_stats.max_drawdown == pytest.approx(0.94)


def test_rebuilds_are_capped_while_earlier_ones_are_pending(manager):
    cap = ValiConfig.AUTO_PROMOTION_MAX_PENDING_LEDGER_REBUILDS
    hotkeys = [f"hk{i}" for i in range(cap + 2)]
    for hotkey in hotkeys:
        _seed(manager, hotkey)
    invalidate = manager._perf_ledger_client.set_hotkey_to_invalidate

    _refresh(manager, {hotkey: _legacy_ledger() for hotkey in hotkeys})
    first = [call.args[0] for call in invalidate.call_args_list]
    assert first == hotkeys[:cap]

    # Still pending: nothing new. A pending ledger is left out of scoring, so it is absent here
    manager._perf_ledger_client.get_perf_ledger_hks_to_invalidate.return_value = {hk: 0 for hk in first}
    _refresh(manager, {hotkey: _legacy_ledger() for hotkey in hotkeys[cap:]})
    assert invalidate.call_count == cap

    # Done: the rest go in
    manager._perf_ledger_client.get_perf_ledger_hks_to_invalidate.return_value = {}
    _refresh(manager, {hotkey: _legacy_ledger() for hotkey in hotkeys[cap:]})
    assert [call.args[0] for call in invalidate.call_args_list] == hotkeys


def test_a_failed_rebuild_request_is_retried(manager):
    _seed(manager)
    manager._perf_ledger_client.set_hotkey_to_invalidate.side_effect = RuntimeError("perf ledger server down")
    _refresh(manager, {HOTKEY: _legacy_ledger()})

    manager._perf_ledger_client.set_hotkey_to_invalidate.side_effect = None
    _refresh(manager, {HOTKEY: _legacy_ledger()})
    assert manager._perf_ledger_client.set_hotkey_to_invalidate.call_count == 2


# ── Nothing else reads the candidates' stats ─────────────────────────────────

def test_no_penalty_reads_the_ratcheted_drawdown_of_a_funded_account():
    for name, config in PenaltyLedgerManager.PENALTIES_CONFIG.items():
        if config.input_type == PenaltyInputType.LEDGER_MAX_DRAWDOWN:
            assert config.buckets is not None and MinerBucket.SUBACCOUNT_FUNDED not in config.buckets, name


def test_a_candidate_never_latches_a_soft_breach_or_shows_pro_stats(manager):
    _seed(manager)
    _refresh(manager, {HOTKEY: _ledger(low=0.50)})  # calmar far below every pro line
    manager._latch_soft_breaches([HOTKEY], NOW_MS)

    assert manager.miner_states[HOTKEY].pro_stats.soft_breach_days == []
    assert manager.get_pro_stats(HOTKEY) is None


def test_an_admin_edit_cannot_silently_deepen_a_candidates_drawdown(manager):
    _seed(manager, current_equity=1.0)
    _refresh(manager, {HOTKEY: _ledger(low=0.99)})

    success, message, _ = manager.set_miner_drawdown_stats(HOTKEY, {"eod_hwm": 1.05}, now_ms=NOW_MS)
    assert not success and "max_drawdown" in message
    success, _, _ = manager.set_miner_drawdown_stats(HOTKEY, {"eod_hwm": 1.05}, force=True, now_ms=NOW_MS)
    assert success


def test_an_admin_edit_on_a_funded_non_candidate_needs_no_force(manager):
    manager._entity_client.get_subaccount_info_for_synthetic.return_value = {
        "initial_bucket": MinerBucket.SUBACCOUNT_FUNDED.value}
    _seed(manager, current_equity=1.0)
    _refresh(manager, {HOTKEY: _ledger(low=0.99)})

    success, _, _ = manager.set_miner_drawdown_stats(HOTKEY, {"eod_hwm": 1.05}, now_ms=NOW_MS)
    assert success
