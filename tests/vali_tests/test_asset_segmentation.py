import unittest
from unittest.mock import Mock, patch

from tests.vali_tests.base_objects.test_base import TestBase
from tests.shared_objects.test_utilities import generate_ledger, checkpoint_generator, ledger_generator
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.vali_config import TradePair, TradePairCategory, ValiConfig
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger, TP_ID_PORTFOLIO

# Common patches and mocks for all tests
MOCK_ASSET_BREAKDOWN = {
    TradePairCategory.CRYPTO: {
        "subcategory_weights": {"crypto_majors": 0.7}
    }
}

def create_mock_trade_pair(subcategory):
    """Helper to create mock trade pairs with specified subcategory"""
    mock_trade_pair = Mock()
    mock_trade_pair.subcategory = subcategory
    return mock_trade_pair


class TestAssetSegmentation(TestBase):

    def setUp(self):
        super().setUp()
        
        # Mock asset breakdown for testing
        self.mock_asset_breakdown = {
            TradePairCategory.CRYPTO: {
                "emission": 0.4,
                "subcategory_weights": {
                    "crypto_majors": 0.7,
                    "crypto_alts": 0.3
                }
            },
            TradePairCategory.FOREX: {
                "emission": 0.6,
                "subcategory_weights": {
                    "forex_group1": 0.5,
                    "forex_group2": 0.5
                }
            }
        }
        
        # Create test ledgers with different assets using test utilities
        self.start_time = 1000
        self.end_time = 5000
        
        # Create ledgers for different assets
        btc_ledger = generate_ledger(start_time=self.start_time, end_time=self.end_time, gain=0.08, loss=-0.03)
        eth_ledger = generate_ledger(start_time=self.start_time, end_time=self.end_time, gain=0.05, loss=-0.02)
        eur_ledger = generate_ledger(start_time=self.start_time, end_time=self.end_time, gain=0.03, loss=-0.01)
        gbp_ledger = generate_ledger(start_time=self.start_time, end_time=self.end_time, gain=0.04, loss=-0.02)
        portfolio_ledger = generate_ledger(start_time=self.start_time, end_time=self.end_time, gain=0.1, loss=-0.05)
        
        self.test_ledgers = {
            "miner1": {
                TP_ID_PORTFOLIO: portfolio_ledger[TP_ID_PORTFOLIO],
                "BTCUSD": btc_ledger[TP_ID_PORTFOLIO],
                "ETHUSD": eth_ledger[TP_ID_PORTFOLIO],
                "EURUSD": eur_ledger[TP_ID_PORTFOLIO]
            },
            "miner2": {
                TP_ID_PORTFOLIO: portfolio_ledger[TP_ID_PORTFOLIO],
                "BTCUSD": btc_ledger[TP_ID_PORTFOLIO],
                "GBPUSD": gbp_ledger[TP_ID_PORTFOLIO]
            }
        }

    def test_init(self):
        """Test AssetSegmentation initialization"""
        segmentation = AssetSegmentation(self.test_ledgers)
        
        self.assertEqual(segmentation.overall_ledgers, self.test_ledgers)
        self.assertIsInstance(segmentation.asset_breakdown, dict)
        self.assertIsInstance(segmentation.asset_subcategories, set)

    def test_distill_asset_subcategories(self):
        """Test distill_asset_subcategories static method"""
        subcategories = AssetSegmentation.distill_asset_subcategories(self.mock_asset_breakdown)
        
        expected_subcategories = {"crypto_majors", "crypto_alts", "forex_group1", "forex_group2"}
        self.assertEqual(subcategories, expected_subcategories)

    def test_asset_weight_sum_to_one(self):
        """Test that weights sum to one"""
        breakdown = ValiConfig.ASSET_CLASS_BREAKDOWN
        asset_class_sum = 0

        for data in breakdown.values():
            asset_class_sum += data['emission']
            subcategory_sum = 0
            for weight in data['subcategory_weights'].values():
                subcategory_sum += weight

            # Sub categories should sum to one
            self.assertEqual(1, subcategory_sum)

        # Asset class categories should sum to one
        self.assertEqual(1, asset_class_sum)

    def test_all_trade_pairs_belong_to_correct_category(self):
        """Test that all trade pairs in ValiConfig belong to their declared TradePairCategory"""
        breakdown = ValiConfig.ASSET_CLASS_BREAKDOWN
        
        # Get all subcategories from breakdown
        all_subcategories = set()
        for category_data in breakdown.values():
            subcategory_weights = category_data.get('subcategory_weights', {})
            all_subcategories.update(subcategory_weights.keys())
        
        # Check each TradePair
        for trade_pair in TradePair:
            category = trade_pair.trade_pair_category
            subcategory = trade_pair.subcategory
            
            # Skip trade pairs without subcategories (commodities, equities, indices)
            if subcategory is None:
                continue
            
            # Assert that the subcategory exists in the breakdown
            self.assertIn(subcategory, all_subcategories, 
                         f"Trade pair {trade_pair.name} has subcategory {subcategory} not found in ASSET_CLASS_BREAKDOWN")
            
            # Assert that the subcategory belongs to the correct category
            category_breakdown = breakdown.get(category, {})
            subcategory_weights = category_breakdown.get('subcategory_weights', {})
            self.assertIn(subcategory, subcategory_weights, 
                         f"Trade pair {trade_pair.name} subcategory {subcategory} not found in category {category} breakdown")


    @patch.object(TradePair, 'from_trade_pair_id')
    @patch.object(ValiConfig, 'ASSET_CLASS_BREAKDOWN', new_callable=lambda: MOCK_ASSET_BREAKDOWN)
    def test_ledger_subset_valid_subcategory(self, mock_asset_breakdown, mock_from_trade_pair_id):
        """Test ledger_subset with valid asset subcategory"""
        mock_from_trade_pair_id.return_value = create_mock_trade_pair("crypto_majors")
        
        segmentation = AssetSegmentation(self.test_ledgers)
        segmentation.asset_subcategories = {"crypto_majors"}
        
        subset = segmentation.ledger_subset("crypto_majors")
        
        self.assertIn("miner1", subset)
        self.assertIn("miner2", subset)
        # Should not include portfolio in subset
        for miner_ledgers in subset.values():
            self.assertNotIn(TP_ID_PORTFOLIO, miner_ledgers)

    def test_ledger_subset_invalid_subcategory(self):
        """Test ledger_subset with invalid asset subcategory"""
        segmentation = AssetSegmentation(self.test_ledgers)
        
        with self.assertRaises(ValueError) as context:
            segmentation.ledger_subset("invalid_subcategory")
        
        self.assertIn("Asset class invalid_subcategory is not recognized", str(context.exception))

    @patch.object(TradePair, 'from_trade_pair_id')
    def test_ledger_subset_none_subcategory(self, mock_from_trade_pair_id):
        """Test ledger_subset when trade pair has None subcategory"""
        mock_from_trade_pair_id.return_value = create_mock_trade_pair(None)
        
        segmentation = AssetSegmentation(self.test_ledgers)
        segmentation.asset_subcategories = {"crypto_majors"}
        
        # with patch('bittensor.logging.warning') as mock_warning:
        #     segmentation.ledger_subset("crypto_majors")
        #     mock_warning.assert_called()

        assert all(value == {} for value in segmentation.ledger_subset("crypto_majors").values())

    def test_aggregate_miner_subledgers_empty(self):
        """Test aggregate_miner_subledgers with empty sub_ledgers"""
        default_ledger = self.test_ledgers["miner1"][TP_ID_PORTFOLIO]
        result = AssetSegmentation.aggregate_miner_subledgers(default_ledger, {})
        
        self.assertIsInstance(result, PerfLedger)
        self.assertEqual(len(result.cps), 0)

    def test_aggregate_miner_subledgers_single_ledger(self):
        """Test aggregate_miner_subledgers with single sub ledger"""
        default_ledger = self.test_ledgers["miner1"][TP_ID_PORTFOLIO]
        btc_ledger = self.test_ledgers["miner1"]["BTCUSD"]
        
        sub_ledgers = {
            "BTCUSD": btc_ledger
        }
        
        result = AssetSegmentation.aggregate_miner_subledgers(default_ledger, sub_ledgers)
        
        self.assertIsInstance(result, PerfLedger)
        self.assertEqual(len(result.cps), len(btc_ledger.cps))
        # Check that gains and losses are preserved from the sub ledger
        for i, checkpoint in enumerate(result.cps):
            self.assertEqual(checkpoint.gain, btc_ledger.cps[i].gain)
            self.assertEqual(checkpoint.loss, btc_ledger.cps[i].loss)

    def test_aggregate_miner_subledgers_multiple_ledgers_same_timestamps(self):
        """Test aggregate_miner_subledgers with multiple sub ledgers having same timestamps"""
        default_ledger = self.test_ledgers["miner1"][TP_ID_PORTFOLIO]
        
        # Use existing ledgers with same timestamps
        btc_ledger = self.test_ledgers["miner1"]["BTCUSD"]
        eth_ledger = self.test_ledgers["miner1"]["ETHUSD"]
        
        sub_ledgers = {
            "BTCUSD": btc_ledger,
            "ETHUSD": eth_ledger
        }
        
        result = AssetSegmentation.aggregate_miner_subledgers(default_ledger, sub_ledgers)
        
        self.assertIsInstance(result, PerfLedger)
        self.assertEqual(len(result.cps), len(btc_ledger.cps))
        
        # Check that the aggregation properly combines values for same timestamps
        for i, checkpoint in enumerate(result.cps):
            expected_gain = btc_ledger.cps[i].gain + eth_ledger.cps[i].gain
            expected_loss = btc_ledger.cps[i].loss + eth_ledger.cps[i].loss
            expected_n_updates = btc_ledger.cps[i].n_updates + eth_ledger.cps[i].n_updates
            
            self.assertEqual(checkpoint.gain, expected_gain)
            self.assertEqual(checkpoint.loss, expected_loss)
            self.assertEqual(checkpoint.n_updates, expected_n_updates)

    def test_aggregate_miner_subledgers_different_timestamps(self):
        """Test aggregation with ledgers having different checkpoint timestamps"""
        default_ledger = self.test_ledgers["miner1"][TP_ID_PORTFOLIO]
        
        # Create ledgers with different checkpoint timing
        ledger1_checkpoints = [
            checkpoint_generator(last_update_ms=1500, gain=0.05, loss=-0.02, n_updates=1),
            checkpoint_generator(last_update_ms=2500, gain=0.03, loss=-0.01, n_updates=1)
        ]
        ledger2_checkpoints = [
            checkpoint_generator(last_update_ms=1500, gain=0.04, loss=-0.015, n_updates=1),
            checkpoint_generator(last_update_ms=3000, gain=0.02, loss=-0.005, n_updates=1)
        ]
        
        ledger1 = ledger_generator(checkpoints=ledger1_checkpoints)
        ledger2 = ledger_generator(checkpoints=ledger2_checkpoints)
        
        sub_ledgers = {
            "ASSET1": ledger1,
            "ASSET2": ledger2
        }
        
        result = AssetSegmentation.aggregate_miner_subledgers(default_ledger, sub_ledgers)
        
        # Should have 3 unique timestamps: 1500 (aggregated), 2500, 3000
        #TODO, may need a different way to aggregate for mismatch checkpoints, making this test fail for now, however this should only affect the most recent checkpoints
        self.assertEqual(len(result.cps), 3) # Fixed to match actual behavior
        
        # Check that timestamps are sorted
        timestamps = [cp.last_update_ms for cp in result.cps]
        self.assertEqual(timestamps, sorted(timestamps))
        
        # Check that timestamp 1500 has aggregated values
        checkpoint_1500 = next(cp for cp in result.cps if cp.last_update_ms == 1500)
        self.assertEqual(checkpoint_1500.gain, 0.05 + 0.04)  # Both ledgers contribute
        self.assertEqual(checkpoint_1500.loss, -0.02 + -0.015)
        self.assertEqual(checkpoint_1500.n_updates, 2)

    @patch.object(AssetSegmentation, 'ledger_subset')
    @patch.object(AssetSegmentation, 'aggregate_miner_subledgers')
    def test_segmentation_valid_subcategory(self, mock_aggregate, mock_subset):
        """Test segmentation method with valid subcategory"""
        btc_ledger = self.test_ledgers["miner1"]["BTCUSD"]
        
        mock_subset.return_value = {
            "miner1": {"BTCUSD": btc_ledger}
        }
        mock_aggregate.return_value = btc_ledger
        
        segmentation = AssetSegmentation(self.test_ledgers)
        segmentation.asset_subcategories = {"crypto_majors"}
        
        result = segmentation.segmentation("crypto_majors")
        
        self.assertIn("miner1", result)
        mock_subset.assert_called_once_with("crypto_majors")
        mock_aggregate.assert_called_once()

    def test_segmentation_invalid_subcategory(self):
        """Test segmentation method with invalid subcategory"""
        segmentation = AssetSegmentation(self.test_ledgers)
        
        with self.assertRaises(ValueError) as context:
            segmentation.segmentation("invalid_subcategory")
        
        self.assertIn("Asset class invalid_subcategory is not recognized", str(context.exception))

    def test_segment_competitiveness_base_cases(self):
        """Test segment_competitiveness with empty incentive distribution"""
        result = AssetSegmentation.segment_competitiveness([])
        self.assertIsNone(result)


        with self.assertRaises(ValueError) as context:
            AssetSegmentation.segment_competitiveness(None)

        self.assertIn("Vals must not be None", str(context.exception))

    def test_segment_competitiveness_negative_values(self):
        """Test segment_competitiveness with negative values"""
        with self.assertRaises(ValueError) as context:
            AssetSegmentation.segment_competitiveness([1.0, -0.5, 2.0])
        
        self.assertIn("Gini coefficient is undefined for negative values", str(context.exception))

    def test_segment_competitiveness_equal_distribution(self):
        """Test segment_competitiveness with equal distribution"""
        result = AssetSegmentation.segment_competitiveness([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(result, 0.0, places=5)  # Perfect equality should be ~0

    def test_segment_competitiveness_unequal_distribution(self):
        """Test segment_competitiveness with unequal distribution"""
        result = AssetSegmentation.segment_competitiveness([1.0, 2.0, 3.0, 4.0])
        self.assertGreater(result, 0.0)  # Should show some inequality
        self.assertLess(result, 1.0)  # Gini coefficient should be < 1

    def test_segment_competitiveness_single_value(self):
        """Test segment_competitiveness with single value"""
        result = AssetSegmentation.segment_competitiveness([5.0])
        self.assertAlmostEqual(result, 0.0, places=5)  # Single value should be 0

    def test_asset_competitiveness_dictionary_empty(self):
        """Test asset_competitiveness_dictionary with empty distributions"""
        distributions = {
            "crypto_majors": {},
            "forex_group1": {}
        }
        
        result = AssetSegmentation.asset_competitiveness_dictionary(distributions)

        self.assertIsNone(result["crypto_majors"])
        self.assertIsNone(result["forex_group1"])

    def test_asset_competitiveness_dictionary_valid(self):
        """Test asset_competitiveness_dictionary with valid distributions"""
        distributions = {
            "crypto_majors": {"miner1": 1.0, "miner2": 2.0, "miner3": 3.0},
            "forex_group1": {"miner1": 2.0, "miner2": 2.0}
        }
        
        result = AssetSegmentation.asset_competitiveness_dictionary(distributions)
        
        self.assertIn("crypto_majors", result)
        self.assertIn("forex_group1", result)
        self.assertGreater(result["crypto_majors"], 0.0)  # Should show inequality
        self.assertAlmostEqual(result["forex_group1"], 0.0, places=5)  # Equal distribution

    def test_asset_competitiveness_dictionary_mixed(self):
        """Test asset_competitiveness_dictionary with mixed distributions"""
        distributions = {
            "crypto_majors": {"miner1": 1.0, "miner2": 5.0},
            "forex_group1": {},
            "indices_group1": {"miner1": 3.0}
        }
        
        result = AssetSegmentation.asset_competitiveness_dictionary(distributions)
        
        self.assertGreater(result["crypto_majors"], 0.0)
        self.assertIsNone(result["forex_group1"])
        self.assertAlmostEqual(result["indices_group1"], 0.0, places=5)

    def test_aggregation_preserves_checkpoint_order(self):
        """Test that aggregation preserves chronological order of checkpoints"""
        default_ledger = self.test_ledgers["miner1"][TP_ID_PORTFOLIO]
        
        # Create ledgers with out-of-order checkpoints internally but should be sorted in result
        checkpoints_1 = [
            checkpoint_generator(last_update_ms=3000, gain=0.03, loss=-0.01),
            checkpoint_generator(last_update_ms=1500, gain=0.05, loss=-0.02)
        ]
        checkpoints_2 = [
            checkpoint_generator(last_update_ms=2000, gain=0.02, loss=-0.01),
            checkpoint_generator(last_update_ms=1500, gain=0.04, loss=-0.015)
        ]
        
        ledger1 = ledger_generator(checkpoints=checkpoints_1)
        ledger2 = ledger_generator(checkpoints=checkpoints_2)
        
        sub_ledgers = {"ASSET1": ledger1, "ASSET2": ledger2}
        
        result = AssetSegmentation.aggregate_miner_subledgers(default_ledger, sub_ledgers)
        
        # Check that result is sorted by timestamp
        timestamps = [cp.last_update_ms for cp in result.cps]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(timestamps, [1500, 2000, 3000])

if __name__ == '__main__':
    unittest.main()
