import copy

from tests.shared_objects.test_utilities import generate_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO


class TestLedgerPenalty(TestBase):
    """
    This class will test penalties that apply to ledgers.
    """

    def setUp(self):
        super().setUp()

    def test_max_drawdown_threshold(self):
        l1 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]  # 1% drawdown

        l2 = copy.deepcopy(l1)
        l2_cps = l2.cps
        l2_cps[-1].mdd = 0.8  # 20% drawdown only on the most recent checkpoint

        l3 = copy.deepcopy(l1)
        l3_cps = l3.cps
        l3_cps[0].mdd = 0.8  # 20% drawdown only on the first checkpoint

        l4 = generate_ledger(0.1, mdd=0.8)[TP_ID_PORTFOLIO]  # 20% drawdown

        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l1), 1.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l2), 0.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l3), 0.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l4), 0.0)

    def test_is_beyond_max_drawdown(self):
        l1 = generate_ledger(0.1, mdd=0.99)[TP_ID_PORTFOLIO]  # 1% drawdown
        l1_ledger = l1

        l2 = copy.deepcopy(l1)
        l2_ledger = l2
        l2_cps = l2_ledger.cps
        l2_cps[-1].mdd = 0.89  # 11% drawdown only on the most recent checkpoint

        l3 = copy.deepcopy(l1)
        l3_ledger = l3
        l3_cps = l3_ledger.cps
        l3_cps[0].mdd = 0.89  # 11% drawdown only on the first checkpoint

        l4 = generate_ledger(0.1, mdd=0.89)[TP_ID_PORTFOLIO]  # 11% drawdown
        l4_ledger = l4

        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(None), (False, 0))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l1_ledger), (False, 1))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l2_ledger), (True, 11))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l3_ledger), (True, 11))
        self.assertEqual(LedgerUtils.is_beyond_max_drawdown(l4_ledger), (True, 11))
