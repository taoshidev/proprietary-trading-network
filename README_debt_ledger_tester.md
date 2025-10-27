# Debt Ledger API Testing Script

A standalone Python script to fetch, validate, and visualize debt ledger data from the PTN REST API.

## Features

- ✅ Fetches debt ledger data from REST API endpoint
- ✅ Validates data structure and values
- ✅ Prints comprehensive summary statistics
- ✅ Creates 4-panel visualization:
  - Debt Percentage Over Time
  - Share Percentage Over Time
  - Total Debt Over Time
  - Recent Entries Table (last 10 checkpoints)
- ✅ Saves plots to file or displays interactively
- ✅ Optionally saves raw JSON response

## Installation

Install required dependencies:

```bash
pip install matplotlib pandas numpy
```

Or if using conda:

```bash
conda install matplotlib pandas numpy
```

## Usage

### Basic Usage

Fetch and visualize debt ledger for a specific hotkey:

```bash
python test_debt_ledger_api.py --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY
```

### Save Plot to File

```bash
python test_debt_ledger_api.py \
  --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY \
  --save-plot debt_ledger_analysis.png
```

### Save Raw JSON Data

```bash
python test_debt_ledger_api.py \
  --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY \
  --save-json debt_ledger_data.json
```

### Test Against Localhost

```bash
python test_debt_ledger_api.py \
  --hotkey 5FRWVox3FD5Jc2VnS7FUCCf8UJgLKfGdEnMAN7nU3LrdMWHu \
  --host localhost \
  --port 48888
```

### Validation Only (No Plot)

```bash
python test_debt_ledger_api.py \
  --hotkey 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY \
  --no-plot
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--hotkey` | Miner hotkey to query (required) | - |
| `--host` | API host address | `34.187.155.10` |
| `--port` | API port | `48888` |
| `--api-key` | API key for authentication | `diQDNkoB3urHC9yOFo7iZOsTo09S?2hm9u` |
| `--save-plot` | Save plot to file path | Interactive display |
| `--save-json` | Save raw JSON to file path | No save |
| `--no-plot` | Skip plotting (validation only) | False |

## Output

### Console Output

The script prints:

1. **Fetch Status**: Success/failure of API call
2. **Validation Results**: Checks for data structure issues
3. **Summary Statistics**:
   - Latest checkpoint values (debt %, share %, total debt)
   - Min/Max/Mean/Std for all metrics
   - Time range covered
   - Recent changes

### Visualization

The plot contains 4 panels:

**Panel 1: Debt Percentage Over Time**
- Red line plot showing debt % evolution
- Annotations for First/Last/Min/Max values
- Useful for spotting anomalies or trends

**Panel 2: Share Percentage Over Time**
- Blue line plot showing share % evolution
- Annotations for First/Last values
- Shows voting power allocation

**Panel 3: Total Debt Over Time**
- Green line plot showing total debt
- Shows absolute debt values

**Panel 4: Recent Entries Table**
- Last 10 checkpoints
- Checkpoint ID, timestamp, debt %, share %, total debt
- Easy to compare with TaoStats

## Cross-Checking with TaoStats

To validate the data:

1. **Run the script** to get latest values:
   ```bash
   python test_debt_ledger_api.py --hotkey YOUR_HOTKEY --save-plot validation.png
   ```

2. **Check TaoStats**: Visit https://taostats.io/subnets/netuid-8/

3. **Compare**:
   - Latest share percentage should match validator's voting weight
   - Debt percentage trends should correlate with performance
   - Total debt should match emission distributions

4. **Look for**:
   - Share % should be between 0-1 (0-100%)
   - Debt % should be between 0-1 (0-100%)
   - Values should change gradually, not jump erratically
   - Recent entries table should show logical progression

## Example Output

```
Fetching debt ledger data from: http://34.187.155.10:48888/debt-ledger/5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY
✓ Successfully fetched debt ledger data

=== Data Validation ===
Total entries: 150
Latest debt percentage: 0.002345
Latest share percentage: 0.012500
✓ All validation checks passed

=== Debt Ledger Summary ===
Hotkey: 5DUi8ZCaNabsR6bnHfs471y52cUN1h9DcugjRbEBo341aKhY
Total entries: 150

--- Latest Entry (Checkpoint #1234) ---
Timestamp: 2025-10-27 21:30:45
Debt Percentage: 0.00234567 (0.234567%)
Share Percentage: 0.01250000 (1.250000%)
Total Debt: 12345.67

--- Statistics ---
Debt Percentage:
  Min:  0.00123456
  Max:  0.00345678
  Mean: 0.00234567
  Std:  0.00012345

...
```

## Troubleshooting

### ImportError: No module named 'matplotlib'

Install dependencies:
```bash
pip install matplotlib pandas numpy
```

### HTTP Error 401: Unauthorized

Check your API key:
```bash
python test_debt_ledger_api.py --hotkey YOUR_HOTKEY --api-key YOUR_API_KEY
```

### Connection Error

Verify the server is running:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://34.187.155.10:48888/health
```

### Empty Plot / No Data

The hotkey may not have any debt ledger entries yet. Try a different hotkey that has been active in the subnet.

## API Endpoint Format

The script calls:
```
GET /debt-ledger/{hotkey}
Authorization: Bearer {api_key}
```

Expected response:
```json
{
  "hotkey": "5DUi8...",
  "entries": [
    {
      "checkpoint_id": 1234,
      "timestamp_ms": 1730064645000,
      "debt_percentage": 0.002345,
      "share_percentage": 0.0125,
      "total_debt": 12345.67
    },
    ...
  ]
}
```

## Notes

- The script uses the API key provided by default: `diQDNkoB3urHC9yOFo7iZOsTo09S?2hm9u`
- Timestamps are automatically converted to readable datetime format
- Percentages are displayed both as decimals (0.0125) and percentages (1.25%)
- The plot is saved at 300 DPI for high quality
