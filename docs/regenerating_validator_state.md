# Regenerating Validator State Manually

When initiating a validator for the first time or recovering from unexpected downtime, it's crucial to synchronize your validator's positions to maintain consensus. Failure to do so may result in missing orders/positions, leading to a lower VTRUST score.

Note: as of 6/11/24, Taoshi added support for automatic validator restoration. Only follow these steps if you are unable to restore your validator using the automatic restoration process.

**Important**: As of the latest version, validator checkpoint files are now compressed (`.gz` format) for improved efficiency and reduced storage requirements. The restoration script automatically detects and handles both compressed and uncompressed checkpoint files.

**File Format Compatibility**: The restore script supports both formats:
- **Compressed files** (`.gz`): Preferred format, ~90% smaller (e.g., 15MB vs 150MB)
- **Uncompressed files** (`.json`): Legacy format, still fully supported

**Download Methods**: Depending on how you download the checkpoint file, it may be automatically decompressed:
- **Browser downloads**: Often automatically decompress `.gz` files to `.json`
- **HTTP clients**: May decompress based on `Accept-Encoding` headers
- **Command line tools**: `curl` and `wget` preserve compression when used correctly

If you receive a decompressed `.json` file instead of `.gz`, simply rename it to `validator_checkpoint.json` (without .gz extension) and the restore script will handle it correctly.
## Purpose

The steps detailed below regenerate the `validation/*` directory by fetching the latest trade data and metrics from the mainnet, ensuring your validator remains in sync with the network.

## What the Restore Overwrites

The restore replaces local state with the checkpoint's contents. It deletes the following first, then writes them back from the checkpoint:

- Per-miner `positions/`, `archived_positions/` and `limit_orders/` directories (including cancelled limit order history, which the checkpoint does not carry)
- Eliminations, perf ledgers (including frozen perf ledgers), challenge period buckets, miner accounts, asset selections, entities and weekly seals

Debt, emissions and penalty ledgers are synced from the checkpoint but their existing files are not deleted first.

**Use a recent checkpoint.** If the checkpoint is missing a section (older checkpoints have no entities, weekly seals or debt/emissions/penalty ledgers), that state is deleted and not replaced. A checkpoint with challenge period records in an older format restores no buckets at all. The script does not check the checkpoint's version.

**No backup is taken.** The script writes straight into `validation/` and does not copy the old state anywhere. Back it up yourself first (step 3) if you may want it back.


## Steps 
1. **Get Restoration File**: Ping a Taoshi team member in the Discord and they will send you a near realtime checkpoint file. (Distribution limited to once a week for verified validators)
   
2. **Prepare for Restoration**: Transfer the checkpoint file to the root level of the `vanta-network` directory on your validator:
   - **If you receive a compressed file**: Name it `validator_checkpoint.json.gz`
   - **If you receive an uncompressed file**: Name it `validator_checkpoint.json`
   - **Auto-detection**: The script will automatically detect which format you have
3. **Stop Validator**: Stop `sn8` and every validator app, then confirm the RPC ports are free, as described in steps 1 and 2 of [Stopping and Restarting Your Validator](validator.md#stopping-and-restarting-your-validator). Every app must be stopped, `vanta-state` included: it holds the position files this procedure replaces.

   Then back up your current state, in case you need to roll back:

    ```bash
    cp -r validation validation_backup_$(date +%Y%m%d_%H%M%S)
    ```
4. **Run Restoration Script**: Within the `vanta-network` directory, execute:

    ```bash
    python3 restore_validator_from_backup.py
    ```

     The script will automatically detect your file format:
    ```
    INFO | Found compressed checkpoint file: /path/to/validator_checkpoint.json.gz
    ```
    or
    ```
    INFO | Found uncompressed checkpoint file: /path/to/validator_checkpoint.json
    ```
    
     Successful restoration is indicated by:
    ```
    INFO | == RESTORE COMPLETED SUCCESSFULLY ==
    INFO | regeneration complete in 19.78 seconds
    ```
     A failed restoration logs `RESTORE FAILED:` followed by a traceback. Partially written state is left in `validation/`; restore your backup from step 3 or fix the problem (see Troubleshooting below) and run the script again.

## Troubleshooting Common Issues

**Problem: File naming mismatches**

If you see an error like:
```
ERROR | File validator_checkpoint.json appears to contain compressed data but lacks .gz extension.
ERROR | Solution: Add .gz extension and rename to validator_checkpoint.json.gz
```
**Solution**: Your file contains compressed data but is named incorrectly. Rename it to `validator_checkpoint.json.gz`.

If you see an error like:
```
ERROR | File validator_checkpoint.json.gz has .gz extension but contains uncompressed data.
ERROR | Solution: Remove the .gz extension and rename to validator_checkpoint.json
```
**Solution**: Your file contains uncompressed data but is named incorrectly. Rename it to `validator_checkpoint.json`.

**Problem: No checkpoint file found**
```
ERROR | No checkpoint file found at validator_checkpoint.json or validator_checkpoint.json.gz
```
**Solution**: Ensure your checkpoint file is in the root directory with the correct name.

5. **Restart Validator**: Run `pm2 start sn8`. It recreates and starts every validator app in the right order (see steps 4 and 5 of [Stopping and Restarting Your Validator](validator.md#stopping-and-restarting-your-validator)).


## Future Improvements

We're committed to refining this process through automation and your feedback. In the future, this process should be fully automated. In the meantime, please report any issues encountered during restoration.
