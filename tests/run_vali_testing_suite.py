# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import unittest
import sys
from vali_config import ValiConfig

if __name__ == '__main__':
    # Create a test loader
    loader = unittest.TestLoader()
    
    if len(sys.argv) > 1:
        # Get the test file name from the command line argument
        test_file = sys.argv[1]
        
        # Load the specific test file
        suite = loader.discover(start_dir=ValiConfig.BASE_DIR + "/tests/vali_tests/", pattern=test_file)
    else:
        # Discover all test files in the specified directory
        start_dir = ValiConfig.BASE_DIR + "/tests/vali_tests/"
        suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create a test runner
    runner = unittest.TextTestRunner()
    
    # Run the tests
    result = runner.run(suite)