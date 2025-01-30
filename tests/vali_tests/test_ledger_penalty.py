import copy
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
from vali_objects.utils.ledger_utils import LedgerUtils

from tests.shared_objects.test_utilities import generate_ledger


class TestLedgerPenalty(TestBase):
    """
    This class will test penalties that apply to ledgers.
    """

    def setUp(self):
        super().setUp()

    def test_max_drawdown_threshold(self):
        l1 = generate_ledger(0.1, mdd=0.99)  # 1% drawdown
        l1_cps = l1[TP_ID_PORTFOLIO].cps
        l2 = copy.deepcopy(l1)
        l2_cps = l2[TP_ID_PORTFOLIO].cps
        l2_cps[-1].mdd = 0.8  # 20% drawdown only on the most recent checkpoint

        l3 = copy.deepcopy(l1)
        l3_cps = l3[TP_ID_PORTFOLIO].cps
        l3_cps[0].mdd = 0.8  # 20% drawdown only on the first checkpoint

        l4 = generate_ledger(0.1, mdd=0.8)  # 20% drawdown
        l4_cps = l4[TP_ID_PORTFOLIO].cps

        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l1_cps), 1.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l2_cps), 0.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l3_cps), 1.0)
        self.assertEqual(LedgerUtils.max_drawdown_threshold_penalty(l4_cps), 0.0)