#!/usr/bin/env python3
"""
Example script demonstrating how to run backtest_manager.py with database positions.

This script shows how to configure and run backtesting using positions loaded
directly from the database instead of cached disk files.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_backtest_with_database_positions():
    """
    Run backtest_manager.py with database positions enabled.

    This function modifies the configuration in backtest_manager.py to enable
    database position loading and runs the backtest.
    """

    # Path to backtest_manager.py
    backtest_manager_path = Path(__file__).parent / "neurons" / "backtest_manager.py"

    if not backtest_manager_path.exists():
        print(f"Error: backtest_manager.py not found at {backtest_manager_path}")
        return False

    print("=" * 60)
    print("Running BacktestManager with Database Positions")
    print("=" * 60)
    print()
    print("Configuration:")
    print("- use_database_positions = True")
    print("- Data source: taoshi.ts.ptn database")
    print("- Time range: 2025-01-01 to 2025-01-05")
    print("- Single miner test mode")
    print()

    # Read the current file
    with open(backtest_manager_path, 'r') as f:
        content = f.read()

    # Check if database position integration is present
    if "use_database_positions" not in content:
        print("Error: Database position integration not found in backtest_manager.py")
        print("Please run the integration script first.")
        return False

    # Backup original file
    backup_path = backtest_manager_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Created backup at: {backup_path}")

    try:
        # Modify configuration to enable database positions
        modified_content = content.replace(
            "use_database_positions = False",
            "use_database_positions = True"
        )

        # Ensure test positions is disabled
        modified_content = modified_content.replace(
            "use_test_positions = True",
            "use_test_positions = False"
        )

        # Write modified content
        with open(backtest_manager_path, 'w') as f:
            f.write(modified_content)

        print("Modified backtest_manager.py to enable database positions")
        print()

        # Run the backtest
        print("Starting backtest...")
        result = subprocess.run([
            sys.executable, str(backtest_manager_path)
        ], capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)
        print()
        print("STDERR:")
        print(result.stderr)
        print()
        print(f"Return code: {result.returncode}")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running backtest: {e}")
        return False

    finally:
        # Restore original file
        with open(backup_path, 'r') as f:
            original_content = f.read()

        with open(backtest_manager_path, 'w') as f:
            f.write(original_content)

        print("Restored original backtest_manager.py from backup")
        os.remove(backup_path)
        print(f"Removed backup file: {backup_path}")

def check_dependencies():
    """Check if required dependencies are available."""
    print("Checking dependencies...")

    try:
        import taoshi.ts.ptn  # noqa: F401
        print("✓ taoshi.ts.ptn module is available")
        return True
    except ImportError:
        print("✗ taoshi.ts.ptn module is not available")
        print("  Please install and configure the taoshi.ts.ptn module")
        return False

def main():
    """Main function to run the database backtest example."""
    print("Database Positions Backtest Example")
    print("===================================")
    print()

    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed. Cannot proceed.")
        return 1

    print()

    # Run backtest with database positions
    success = run_backtest_with_database_positions()

    if success:
        print()
        print("✓ Backtest completed successfully!")
        print("Database positions were loaded and processed.")
        return 0
    else:
        print()
        print("✗ Backtest failed.")
        print("Check the error messages above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
