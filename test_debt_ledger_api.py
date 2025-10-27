#!/usr/bin/env python3
"""
Standalone script to test Debt Ledger API endpoint and visualize data for validation.

Usage:
    python test_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY
    python test_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY --save-plot output.png
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Dict, List, Any
import urllib.request
import urllib.error

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


class DebtLedgerTester:
    """Test and visualize debt ledger data from the API."""

    def __init__(self, api_host: str, api_port: int, api_key: str):
        self.api_host = api_host
        self.api_port = api_port
        self.api_key = api_key
        self.base_url = f"http://{api_host}:{api_port}"

    def fetch_debt_ledger(self, hotkey: str) -> Dict[str, Any]:
        """Fetch debt ledger data from the API."""
        url = f"{self.base_url}/debt-ledger/{hotkey}"

        print(f"Fetching debt ledger data from: {url}")

        try:
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {self.api_key}')

            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    print(f"✓ Successfully fetched debt ledger data")
                    return data
                else:
                    print(f"✗ HTTP Error: {response.status}")
                    sys.exit(1)

        except urllib.error.HTTPError as e:
            print(f"✗ HTTP Error {e.code}: {e.reason}")
            print(f"Response: {e.read().decode('utf-8')}")
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"✗ Connection Error: {e.reason}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)

    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate the structure and content of debt ledger data."""
        print("\n=== Data Validation ===")

        issues = []

        # Check required fields
        required_fields = ['hotkey', 'entries']
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")

        if 'entries' in data:
            entries = data['entries']
            print(f"Total entries: {len(entries)}")

            if not entries:
                issues.append("No entries found in debt ledger")
            else:
                # Validate entry structure
                required_entry_fields = ['checkpoint_id', 'timestamp_ms', 'debt_percentage', 'share_percentage', 'total_debt']

                for i, entry in enumerate(entries[:5]):  # Check first 5 entries
                    for field in required_entry_fields:
                        if field not in entry:
                            issues.append(f"Entry {i} missing field: {field}")

                # Validate percentages sum to reasonable values
                if len(entries) > 0:
                    latest = entries[-1]
                    debt_pct = latest.get('debt_percentage', 0)
                    share_pct = latest.get('share_percentage', 0)

                    print(f"Latest debt percentage: {debt_pct:.6f}")
                    print(f"Latest share percentage: {share_pct:.6f}")

                    if debt_pct < 0 or debt_pct > 1:
                        issues.append(f"Invalid debt_percentage: {debt_pct} (should be 0-1)")
                    if share_pct < 0 or share_pct > 1:
                        issues.append(f"Invalid share_percentage: {share_pct} (should be 0-1)")

        if issues:
            print("\n⚠️  Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✓ All validation checks passed")
            return True

    def print_summary(self, data: Dict[str, Any]):
        """Print summary statistics of the debt ledger."""
        print("\n=== Debt Ledger Summary ===")

        hotkey = data.get('hotkey', 'Unknown')
        entries = data.get('entries', [])

        print(f"Hotkey: {hotkey}")
        print(f"Total entries: {len(entries)}")

        if not entries:
            print("No entries to summarize")
            return

        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(entries)
        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

        # Latest values
        latest = df.iloc[-1]
        print(f"\n--- Latest Entry (Checkpoint #{int(latest['checkpoint_id'])}) ---")
        print(f"Timestamp: {latest['datetime']}")
        print(f"Debt Percentage: {latest['debt_percentage']:.8f} ({latest['debt_percentage']*100:.6f}%)")
        print(f"Share Percentage: {latest['share_percentage']:.8f} ({latest['share_percentage']*100:.6f}%)")
        print(f"Total Debt: {latest['total_debt']:.2f}")

        # Statistics
        print(f"\n--- Statistics ---")
        print(f"Debt Percentage:")
        print(f"  Min:  {df['debt_percentage'].min():.8f}")
        print(f"  Max:  {df['debt_percentage'].max():.8f}")
        print(f"  Mean: {df['debt_percentage'].mean():.8f}")
        print(f"  Std:  {df['debt_percentage'].std():.8f}")

        print(f"\nShare Percentage:")
        print(f"  Min:  {df['share_percentage'].min():.8f}")
        print(f"  Max:  {df['share_percentage'].max():.8f}")
        print(f"  Mean: {df['share_percentage'].mean():.8f}")
        print(f"  Std:  {df['share_percentage'].std():.8f}")

        print(f"\nTotal Debt:")
        print(f"  Min:  {df['total_debt'].min():.2f}")
        print(f"  Max:  {df['total_debt'].max():.2f}")
        print(f"  Mean: {df['total_debt'].mean():.2f}")

        # Time range
        time_range = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
        print(f"\nTime Range: {time_range:.2f} hours ({time_range/24:.2f} days)")

        # Recent changes
        if len(df) > 1:
            recent_change_debt = df['debt_percentage'].iloc[-1] - df['debt_percentage'].iloc[-2]
            recent_change_share = df['share_percentage'].iloc[-1] - df['share_percentage'].iloc[-2]
            print(f"\nRecent Change (last checkpoint):")
            print(f"  Debt %:  {recent_change_debt:+.8f} ({recent_change_debt*100:+.6f}%)")
            print(f"  Share %: {recent_change_share:+.8f} ({recent_change_share*100:+.6f}%)")

    def plot_debt_ledger(self, data: Dict[str, Any], save_path: str = None):
        """Create comprehensive visualization of debt ledger data."""
        entries = data.get('entries', [])

        if not entries:
            print("No entries to plot")
            return

        # Convert to DataFrame
        df = pd.DataFrame(entries)
        df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Debt Ledger Analysis - {data.get("hotkey", "Unknown")[:16]}...',
                     fontsize=16, fontweight='bold')

        # 1. Debt Percentage Over Time
        ax1 = axes[0, 0]
        ax1.plot(df['datetime'], df['debt_percentage'] * 100,
                marker='o', markersize=4, linewidth=2, color='#e74c3c', alpha=0.8)
        ax1.set_title('Debt Percentage Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Debt Percentage (%)')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax1.tick_params(axis='x', rotation=45)

        # Add value annotations for first, last, min, max
        first_idx = 0
        last_idx = len(df) - 1
        min_idx = df['debt_percentage'].idxmin()
        max_idx = df['debt_percentage'].idxmax()

        for idx, label in [(first_idx, 'First'), (last_idx, 'Last'),
                           (min_idx, 'Min'), (max_idx, 'Max')]:
            ax1.annotate(f'{label}: {df.iloc[idx]["debt_percentage"]*100:.4f}%',
                        xy=(df.iloc[idx]['datetime'], df.iloc[idx]['debt_percentage']*100),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

        # 2. Share Percentage Over Time
        ax2 = axes[0, 1]
        ax2.plot(df['datetime'], df['share_percentage'] * 100,
                marker='s', markersize=4, linewidth=2, color='#3498db', alpha=0.8)
        ax2.set_title('Share Percentage Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Share Percentage (%)')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax2.tick_params(axis='x', rotation=45)

        # Add annotations
        for idx, label in [(first_idx, 'First'), (last_idx, 'Last')]:
            ax2.annotate(f'{label}: {df.iloc[idx]["share_percentage"]*100:.4f}%',
                        xy=(df.iloc[idx]['datetime'], df.iloc[idx]['share_percentage']*100),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

        # 3. Total Debt Over Time
        ax3 = axes[1, 0]
        ax3.plot(df['datetime'], df['total_debt'],
                marker='D', markersize=4, linewidth=2, color='#2ecc71', alpha=0.8)
        ax3.set_title('Total Debt Over Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Total Debt')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax3.tick_params(axis='x', rotation=45)

        # Format y-axis with commas
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        # 4. Recent Entries Table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Show last 10 entries
        recent_df = df.tail(10)[['checkpoint_id', 'datetime', 'debt_percentage', 'share_percentage', 'total_debt']].copy()
        recent_df['debt_percentage'] = recent_df['debt_percentage'].apply(lambda x: f'{x*100:.6f}%')
        recent_df['share_percentage'] = recent_df['share_percentage'].apply(lambda x: f'{x*100:.6f}%')
        recent_df['total_debt'] = recent_df['total_debt'].apply(lambda x: f'{x:,.2f}')
        recent_df['datetime'] = recent_df['datetime'].apply(lambda x: x.strftime('%m/%d %H:%M'))
        recent_df['checkpoint_id'] = recent_df['checkpoint_id'].astype(int)

        # Create table
        table = ax4.table(cellText=recent_df.values,
                         colLabels=['CP ID', 'Date/Time', 'Debt %', 'Share %', 'Total Debt'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(recent_df) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        ax4.set_title('Recent Entries (Last 10 Checkpoints)', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Test Debt Ledger API endpoint and visualize data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY
  python test_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY --save-plot debt_ledger.png
  python test_debt_ledger_api.py --hotkey 5FRWVox3FD5Jc2VnS7FUCCf8UJgLKfGdEnMAN7nU3LrdMWHu --host localhost
        """
    )

    parser.add_argument('--hotkey', type=str, required=True,
                       help='Miner hotkey to query')
    parser.add_argument('--host', type=str, default='34.187.155.10',
                       help='API host (default: 34.187.155.10)')
    parser.add_argument('--port', type=int, default=48888,
                       help='API port (default: 48888)')
    parser.add_argument('--api-key', type=str, default='diQDNkoB3urHC9yOFo7iZOsTo09S?2hm9u',
                       help='API key for authentication')
    parser.add_argument('--save-plot', type=str, metavar='PATH',
                       help='Save plot to file instead of displaying')
    parser.add_argument('--save-json', type=str, metavar='PATH',
                       help='Save raw JSON response to file')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (validation only)')

    args = parser.parse_args()

    # Create tester
    tester = DebtLedgerTester(args.host, args.port, args.api_key)

    # Fetch data
    data = tester.fetch_debt_ledger(args.hotkey)

    # Save raw JSON if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Raw JSON saved to: {args.save_json}")

    # Validate data
    is_valid = tester.validate_data(data)

    # Print summary
    tester.print_summary(data)

    # Plot data
    if not args.no_plot:
        tester.plot_debt_ledger(data, save_path=args.save_plot)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
