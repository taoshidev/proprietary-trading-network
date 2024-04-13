# Regenerating Validator State

When initiating a validator for the first time or recovering from unexpected downtime, it's crucial to synchronize your validator's positions to maintain consensus. Failure to do so may result in missing orders/positions, leading to a lower VTRUST score.

## Purpose

The script detailed below regenerates the `validation/*` directory by fetching the latest positions, eliminations, and plagiarism scores from the mainnet, ensuring your validator remains in sync with the network.

## Steps Automatic
1. **Stop Validator**: Temporarily halt your validator with PM2 using `pm2 stop sn8 ptn`.
2. Within the `proprietary-trading-network` directory, execute:

    ```bash
    curl https://dashboard.taoshi.io/api/validator-checkpoint -o validator_checkpoint.json && sed -i 's/^{"checkpoint"://' validator_checkpoint.json && sed -i 's/}$//' validator_checkpoint.json && python3 restore_validator_from_backup.py
    ```
3. **Restart Validator**: Resume your PM2 processes with `pm2 start sn8` (will launch ptn automatically).


## Steps Manual (Use if automatic setup fails)

2. **Stop Validator**: Temporarily halt your validator with PM2 using `pm2 stop sn8 ptn`
3. **Download Positions**: Visit [Taoshi Dashboard](https://dashboard.taoshi.io/) and click the "Download Positions" button to obtain a file named `validator_checkpoint.json`. Rename the file if necessary.
4. **Prepare for Restoration**: Move `validator_checkpoint.json` to the root level of the `proprietary-trading-network` directory.
5. **Run Restoration Script**: Within the `proprietary-trading-network` directory, execute:

    ```bash
    python3 restore_validator_from_backup.py
    ```

     Successful restoration is indicated by:
    ```
    2024-03-25 01:05:36.660 | INFO | regeneration complete in 1.25 seconds
    ```
     If restoration fails, consult the failure log for troubleshooting steps.
     A commonly seen error happens when the regeneration script sees that the disk positions are newer than the checkpoint file's positions. In this case, please re-download form the dashboard (updated every few seconds). Please do NOT delete any local files.



6. **Restart Validator**: Resume your PM2 processes with `pm2 start sn8` (will launch ptn automatically).


## Verify Backups (optional)

Confirm the creation of a `backups/` directory containing previous data (positions, eliminations, plagiarism). For example:
    ```
    (venv) jbonilla@MacBook-Air prop-net % ls -lh backups/20240325_010535


    total 16
    -rw-r--r--  1 jbonilla staff 164B Mar 25 01:05 eliminations.json
    drwxr-xr-x 148 jbonilla staff 4.6K Mar 25 00:36 miners
    -rw-r--r--  1 jbonilla staff 165B Mar 25 01:05 plagiarism.json
    ```

## Future Improvements

We're committed to refining this process through automation and your feedback. In the future, this process should be fully automated. In the meantime, please report any issues encountered during restoration.
