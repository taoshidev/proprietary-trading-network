# Database Positions Integration for BacktestManager

## Overview

The `backtest_manager.py` now supports loading positions directly from the database using the `taoshi.ts.ptn` module. This feature allows you to backtest using live production data without relying on cached disk files.

## Feature Integration

Based on commit `cfe32a2750a10c5555e57675c9529016273d708f`, I've integrated database position loading as a feature-gated option.

## Configuration

### Enable Database Positions

In `neurons/backtest_manager.py`, set the flag:

```python
use_database_positions = True  # Enable database position loading
```

### Configuration Options

```python
# ============= CONFIGURATION FLAGS =============
use_test_positions = False         # Use hardcoded test positions
use_database_positions = True      # NEW: Use positions from database via taoshi.ts.ptn
run_challenge = False              # Run challenge period logic
run_elimination = False            # Run elimination logic
use_slippage = False              # Apply slippage modeling
build_portfolio_ledgers_only = True  # Whether to build only the portfolio ledgers or per trade pair
parallel_mode = ParallelizationMode.SERIAL  # 1 for pyspark, 2 for multiprocessing

# Time range for database query
start_time_ms = 1735689600000
end_time_ms = 1736035200000
test_single_hotkey = '5HDmzyhrEco9w6Jv8eE3hDMcXSE4AGg1MuezPR4u2covxKwZ'  # Specific miner or None for all
```

## How It Works

### 1. **Database Connection**
- Automatically sets required environment variables:
  ```python
  os.environ["TAOSHI_TS_DEPLOYMENT"] = "DEVELOPMENT"
  os.environ["TAOSHI_TS_PLATFORM"] = "LOCAL"
  ```

### 2. **Position Retrieval**
- Uses `ptn_utils.DatabasePositionOrderSource()` to connect to database
- Queries positions with orders for specified time range and miners
- Supports filtering by miner hotkeys and time periods

### 3. **Data Conversion**
- Converts database format to `Position` objects
- Handles trade pair ID mapping using `TradePair.from_trade_pair_id()`
- Manages position type conversions
- Includes comprehensive error handling

### 4. **Integration**
- Saves converted positions to `PositionManager`
- Initializes `PerfLedgerManager` with database positions
- Maintains compatibility with existing backtest logic

## Usage Examples

### Example 1: Single Miner Backtest
```python
use_database_positions = True
start_time_ms = 1735689600000  # Jan 1, 2025
end_time_ms = 1736035200000    # Jan 5, 2025
test_single_hotkey = '5HDmzyhrEco9w6Jv8eE3hDMcXSE4AGg1MuezPR4u2covxKwZ'
```

### Example 2: Multi-Miner Analysis
```python
use_database_positions = True
start_time_ms = 1735689600000
end_time_ms = 1736035200000
test_single_hotkey = None  # Process all miners in time range
```

### Example 3: Performance Testing
```python
use_database_positions = True
parallel_mode = ParallelizationMode.MULTIPROCESSING
build_portfolio_ledgers_only = False  # Build per-trade-pair ledgers
```

## Requirements

### Dependencies
- `taoshi.ts.ptn` module must be available
- Database connection configured and accessible
- Proper authentication/credentials setup

### Environment Setup
The feature automatically configures required environment variables, but ensure your database access is properly set up.

## Error Handling

The implementation includes comprehensive error handling:

1. **Import Errors**: Clear messages if `taoshi.ts.ptn` is unavailable
2. **Database Errors**: Graceful handling of connection/query failures
3. **Data Conversion**: Validation and error logging for position object creation
4. **Configuration Validation**: Prevents conflicting settings

## Logging

The feature provides detailed logging:
- Database query parameters and results
- Position conversion statistics
- Error details for troubleshooting
- Performance timing information

## Advantages Over Disk-Based Positions

1. **Real-Time Data**: Access to latest positions without cache delays
2. **Flexible Querying**: Filter by time ranges and specific miners
3. **Data Integrity**: Direct access to source of truth
4. **Scalability**: No disk I/O limitations for large datasets

## Migration from Disk Positions

To migrate from disk-based to database positions:

1. Ensure `taoshi.ts.ptn` is properly installed and configured
2. Update configuration flags in `backtest_manager.py`
3. Test with a single miner first
4. Gradually expand to full dataset

## Performance Considerations

- Database queries may be slower than disk cache for small datasets
- Network latency affects query performance
- Consider using `build_portfolio_ledgers_only=True` for faster processing
- Use specific miner hotkeys to limit query scope when possible

## Troubleshooting

### Common Issues

1. **ImportError**: `taoshi.ts.ptn` not available
   - Ensure the module is installed and accessible
   - Check Python path and dependencies

2. **Database Connection**: Connection failures
   - Verify database credentials and connectivity
   - Check environment variable configuration

3. **Position Conversion**: Unknown trade pair IDs
   - Verify trade pair mappings in `TradePair.from_trade_pair_id()`
   - Check for new trade pairs not in the enum

### Debug Tips

- Enable verbose logging with `bt.logging.enable_debug()`
- Start with small time ranges for testing
- Use single miner hotkey for initial validation
- Monitor database query performance