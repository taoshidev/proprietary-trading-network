import os
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from collateral_sdk import Network

from tests.shared_objects.mock_classes import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.validator_contract_manager import ValidatorContractManager, CollateralRecord
from vali_objects.vali_config import ValiConfig


class TestContractManager(TestBase):
    def setUp(self):
        super().setUp()
        
        # Test miners
        self.MINER_1 = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        self.MINER_2 = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the CollateralManager to avoid actual contract calls
        with patch('vali_objects.utils.contract_manager.CollateralManager') as mock_collateral_manager:
            self.mock_collateral_manager_instance = MagicMock()
            mock_collateral_manager.return_value = self.mock_collateral_manager_instance
            
            # Initialize ContractManager with test directory
            self.contract_manager = ContractManager(
                network=Network.TESTNET,
                data_dir=self.temp_dir
            )
        
        # Set up mock collateral balances
        self.mock_balances = {
            self.MINER_1: 1000000,  # 1M theta
            self.MINER_2: 500000    # 500K theta
        }
        
        self.mock_collateral_manager_instance.balance_of.side_effect = lambda hotkey: self.mock_balances.get(hotkey, 0)
    
    def tearDown(self):
        super().tearDown()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_collateral_record_creation(self):
        """Test CollateralRecord creation and properties"""
        timestamp_ms = int(time.time() * 1000)
        account_size = 10000.0
        
        record = CollateralRecord(account_size, timestamp_ms)
        
        self.assertEqual(record.account_size, account_size)
        self.assertEqual(record.update_time_ms, timestamp_ms)
        self.assertIsInstance(record.valid_date_timestamp, int)
        self.assertIsInstance(record.valid_date_str, str)
        
        # Test date string format
        self.assertRegex(record.valid_date_str, r'^\d{4}-\d{2}-\d{2}$')

    def test_set_and_get_miner_account_size(self):
        """Test setting and getting miner account sizes"""
        account_size_1 = 15000.0
        account_size_2 = 25000.0
        
        # Initially should return default ValiConfig.CAPITAL
        self.assertEqual(self.contract_manager.get_miner_account_size(self.MINER_1), ValiConfig.CAPITAL)
        
        # Set account sizes
        self.contract_manager.set_miner_account_size(self.MINER_1, account_size_1)
        self.contract_manager.set_miner_account_size(self.MINER_2, account_size_2)
        
        # Verify retrieval
        self.assertEqual(self.contract_manager.get_miner_account_size(self.MINER_1), account_size_1)
        self.assertEqual(self.contract_manager.get_miner_account_size(self.MINER_2), account_size_2)
        
        # Update account size and verify latest is returned
        new_account_size = 30000.0
        time.sleep(1)
        self.contract_manager.set_miner_account_size(self.MINER_1, new_account_size)
        self.assertEqual(self.contract_manager.get_miner_account_size(self.MINER_1), new_account_size)

    def test_account_size_persistence(self):
        """Test that account sizes are saved to and loaded from disk"""
        account_size = 20000.0
        
        # Set account size
        self.contract_manager.set_miner_account_size(self.MINER_1, account_size)
        self.assertEqual(self.contract_manager.get_miner_account_size(self.MINER_1), account_size)
        
        # Create new ContractManager instance to test loading
        with patch('vali_objects.utils.contract_manager.CollateralManager') as mock_collateral_manager:
            mock_collateral_manager.return_value = self.mock_collateral_manager_instance
            
            new_contract_manager = ContractManager(network=Network.TESTNET, data_dir=self.temp_dir)
            
            # Verify account size is loaded correctly
            self.assertEqual(new_contract_manager.get_miner_account_size(self.MINER_1), account_size)
            self.assertEqual(new_contract_manager.get_miner_account_size(self.MINER_2), ValiConfig.CAPITAL)

    def test_multiple_account_size_records(self):
        """Test that multiple records are stored and sorted correctly"""
        base_time = int(time.time() * 1000)
        
        # Add multiple records with different timestamps
        self.contract_manager.set_miner_account_size(self.MINER_1, 10000.0)
        time.sleep(0.001)  # Ensure different timestamps
        self.contract_manager.set_miner_account_size(self.MINER_1, 15000.0)
        time.sleep(0.001)
        self.contract_manager.set_miner_account_size(self.MINER_1, 20000.0)
        
        # Should return the latest (20000.0)
        self.assertEqual(self.contract_manager.get_miner_account_size(self.MINER_1), 20000.0)
        
        # Verify records are stored
        records = self.contract_manager.miner_account_sizes[self.MINER_1]
        self.assertEqual(len(records), 3)
        
        # Verify records are sorted by update_time_ms
        for i in range(1, len(records)):
            self.assertGreaterEqual(records[i].update_time_ms, records[i-1].update_time_ms)

    def test_cleanup_old_records(self):
        """Test cleanup of old account size records"""
        current_time = int(time.time() * 1000)
        
        # Create records with different ages
        old_time = current_time - (150 * 24 * 60 * 60 * 1000)  # 150 days ago
        recent_time = current_time - (50 * 24 * 60 * 60 * 1000)  # 50 days ago
        
        # Manually add old and recent records
        old_record = CollateralRecord(10000.0, old_time)
        recent_record = CollateralRecord(20000.0, recent_time)
        current_record = CollateralRecord(30000.0, current_time)
        
        self.contract_manager.miner_account_sizes[self.MINER_1] = [old_record, recent_record, current_record]
        
        # Clean up records older than 120 days
        cutoff_time = current_time - (120 * 24 * 60 * 60 * 1000)
        removed_count = self.contract_manager._cleanup_old_records(self.MINER_1, cutoff_time)
        
        # Should have removed 1 record (the 150-day old one)
        self.assertEqual(removed_count, 1)
        self.assertEqual(len(self.contract_manager.miner_account_sizes[self.MINER_1]), 2)
        
        # Remaining records should be the recent ones
        remaining_sizes = [r.account_size for r in self.contract_manager.miner_account_sizes[self.MINER_1]]
        self.assertIn(20000.0, remaining_sizes)
        self.assertIn(30000.0, remaining_sizes)
        self.assertNotIn(10000.0, remaining_sizes)

    def test_needs_account_size_update(self):
        """Test the logic for determining if account size needs updating"""
        current_time = int(time.time() * 1000)
        target_date_timestamp = CollateralRecord.valid_from_ms(current_time)
        
        # Miner with no records should need update
        self.assertTrue(self.contract_manager._needs_account_size_update(self.MINER_1, target_date_timestamp))
        
        # Add a current record
        current_record = CollateralRecord(10000.0, current_time)
        self.contract_manager.miner_account_sizes[self.MINER_1] = [current_record]
        
        # Should not need update for current day
        self.assertFalse(self.contract_manager._needs_account_size_update(self.MINER_1, target_date_timestamp))
        
        # Should need update for future day
        future_timestamp = target_date_timestamp + (24 * 60 * 60 * 1000)  # Next day
        self.assertTrue(self.contract_manager._needs_account_size_update(self.MINER_1, future_timestamp))

    def test_get_recent_account_sizes(self):
        """Test retrieving recent account sizes for multiple miners"""
        current_time = int(time.time() * 1000)
        
        # Set account sizes for both miners
        self.contract_manager.set_miner_account_size(self.MINER_1, 15000.0)
        self.contract_manager.set_miner_account_size(self.MINER_2, 25000.0)
        
        # Get recent account sizes
        recent_sizes = self.contract_manager.get_recent_account_sizes([self.MINER_1, self.MINER_2], current_time)
        
        self.assertEqual(len(recent_sizes), 2)
        self.assertEqual(recent_sizes[self.MINER_1], 15000.0)
        self.assertEqual(recent_sizes[self.MINER_2], 25000.0)
        
        # Test with no hotkeys specified (should return all)
        all_sizes = self.contract_manager.get_recent_account_sizes(timestamp_ms=current_time)
        self.assertGreaterEqual(len(all_sizes), 2)

    def test_get_historical_account_sizes(self):
        """Test retrieving historical account size records"""
        # Add multiple records for a miner
        self.contract_manager.set_miner_account_size(self.MINER_1, 10000.0)
        time.sleep(0.001)
        self.contract_manager.set_miner_account_size(self.MINER_1, 15000.0)
        
        # Get historical records
        historical = self.contract_manager.get_historical_account_sizes([self.MINER_1])
        
        self.assertIn(self.MINER_1, historical)
        self.assertEqual(len(historical[self.MINER_1]), 2)
        
        # Verify record types
        for record in historical[self.MINER_1]:
            self.assertIsInstance(record, CollateralRecord)

    def test_collateral_balance_caching(self):
        """Test that collateral balances are cached properly"""
        # First call should hit the mock
        balance1 = self.contract_manager.get_miner_collateral_balance(self.MINER_1, use_cache=False)
        self.assertEqual(balance1, self.mock_balances[self.MINER_1])
        self.mock_collateral_manager_instance.balance_of.assert_called_with(self.MINER_1)
        
        # Reset mock call count
        self.mock_collateral_manager_instance.balance_of.reset_mock()
        
        # Second call with cache should not hit the mock
        balance2 = self.contract_manager.get_miner_collateral_balance(self.MINER_1, use_cache=True)
        self.assertEqual(balance2, balance1)
        self.mock_collateral_manager_instance.balance_of.assert_not_called()
        
        # Call without cache should hit the mock again
        balance3 = self.contract_manager.get_miner_collateral_balance(self.MINER_1, use_cache=False)
        self.assertEqual(balance3, balance1)
        self.mock_collateral_manager_instance.balance_of.assert_called_once()

    def test_capital_allocation_calculation(self):
        """Test capital allocation calculation based on collateral and account size"""
        tao_price_usd = 50.0
        account_size = 20000.0
        
        # Set account size
        self.contract_manager.set_miner_account_size(self.MINER_1, account_size)
        
        # Calculate allocation (collateral = 1M theta * $50 = $50M, account_size = $20K)
        allocated_capital, collateral_ratio = self.contract_manager.calculate_capital_allocation(
            self.MINER_1, tao_price_usd
        )
        
        # Should allocate minimum of collateral value and account size
        expected_collateral_usd = self.mock_balances[self.MINER_1] * tao_price_usd
        expected_allocated = min(expected_collateral_usd, account_size)
        expected_ratio = expected_collateral_usd / account_size
        
        self.assertEqual(allocated_capital, expected_allocated)
        self.assertEqual(collateral_ratio, expected_ratio)
