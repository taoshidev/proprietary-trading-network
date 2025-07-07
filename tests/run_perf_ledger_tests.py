#!/usr/bin/env python3
"""
Comprehensive test runner for performance ledger tests.

This script runs all the performance ledger tests with proper mocking of external dependencies.
"""

import sys
import unittest
import os
from unittest.mock import patch, Mock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import test modules
from tests.vali_tests.test_perf_ledgers_comprehensive import TestPerfLedgerCore, TestPerfLedgerManager
from tests.vali_tests.test_perf_ledger_performance import TestPerfLedgerDeltaBuilding, TestParallelVsSerialModes
from tests.vali_tests.test_perf_ledger_advanced import (
    TestPerfLedgerSerialization, TestPerfLedgerEliminationLogic, TestCheckpointManagement
)
from tests.vali_tests.test_perf_ledger_portfolio_alignment import TestPortfolioTradeParAlignment
from tests.vali_tests.test_perf_ledger_stress_edge_cases import TestPerfLedgerStressTests, TestPerfLedgerEdgeCases
from tests.vali_tests.test_perf_ledger_real_world_scenarios import TestRealWorldTradingScenarios, TestIntegrationScenarios


def mock_external_dependencies():
    """Mock external dependencies that might not be available in test environment"""
    
    # Mock Polygon data service
    polygon_patcher = patch('data_generator.polygon_data_service.PolygonDataService')
    mock_polygon = polygon_patcher.start()
    mock_polygon.return_value.unified_candle_fetcher.return_value = {}
    
    # Mock Tiingo data service
    tiingo_patcher = patch('data_generator.tiingo_data_service.TiingoDataService')
    mock_tiingo = tiingo_patcher.start()
    mock_tiingo.return_value.get_data.return_value = {}
    
    # Mock price data service
    pds_patcher = patch('vali_objects.vali_dataclasses.perf_ledger.PriceDataService')
    mock_pds = pds_patcher.start()
    mock_pds.return_value.unified_candle_fetcher.return_value = {}
    
    # Mock bittensor logging
    bt_patcher = patch('bittensor.logging')
    mock_bt = bt_patcher.start()
    mock_bt.info = Mock()
    mock_bt.warning = Mock()
    mock_bt.error = Mock()
    mock_bt.success = Mock()
    mock_bt.trace = Mock()
    
    return [polygon_patcher, tiingo_patcher, pds_patcher, bt_patcher]


def run_test_suite():
    """Run the complete performance ledger test suite"""
    
    print("Setting up mocks for external dependencies...")
    patchers = mock_external_dependencies()
    
    try:
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestPerfLedgerCore,
            TestPerfLedgerManager,
            TestPerfLedgerDeltaBuilding,
            TestParallelVsSerialModes,
            TestPerfLedgerSerialization,
            TestPerfLedgerEliminationLogic,
            TestCheckpointManagement,
            TestPortfolioTradeParAlignment,
            TestPerfLedgerStressTests,
            TestPerfLedgerEdgeCases,
            TestRealWorldTradingScenarios,
            TestIntegrationScenarios
        ]
        
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        print(f"Running {suite.countTestCases()} performance ledger tests...")
        print("=" * 80)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(
            verbosity=2,
            stream=sys.stdout,
            descriptions=True,
            failfast=False
        )
        
        result = runner.run(suite)
        
        # Print summary
        print("=" * 80)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")
        
        # Return success/failure
        return len(result.failures) == 0 and len(result.errors) == 0
        
    finally:
        # Clean up patches
        print("\nCleaning up mocks...")
        for patcher in patchers:
            try:
                patcher.stop()
            except RuntimeError:
                pass  # Already stopped


def run_specific_test_class(class_name):
    """Run tests for a specific test class"""
    
    class_map = {
        'core': TestPerfLedgerCore,
        'manager': TestPerfLedgerManager,
        'delta': TestPerfLedgerDeltaBuilding,
        'parallel': TestParallelVsSerialModes,
        'serialization': TestPerfLedgerSerialization,
        'elimination': TestPerfLedgerEliminationLogic,
        'checkpoint': TestCheckpointManagement,
        'alignment': TestPortfolioTradeParAlignment,
        'stress': TestPerfLedgerStressTests,
        'edge': TestPerfLedgerEdgeCases,
        'realworld': TestRealWorldTradingScenarios,
        'integration': TestIntegrationScenarios
    }
    
    if class_name not in class_map:
        print(f"Unknown test class: {class_name}")
        print(f"Available classes: {', '.join(class_map.keys())}")
        return False
    
    print(f"Setting up mocks for {class_name} tests...")
    patchers = mock_external_dependencies()
    
    try:
        test_class = class_map[class_name]
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        print(f"Running {suite.countTestCases()} tests from {test_class.__name__}...")
        print("=" * 60)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    finally:
        for patcher in patchers:
            try:
                patcher.stop()
            except RuntimeError:
                pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test class
        class_name = sys.argv[1]
        success = run_specific_test_class(class_name)
    else:
        # Run all tests
        success = run_test_suite()
    
    sys.exit(0 if success else 1)