# Regenerating Validator State

When initiating a validator for the first time or recovering from unexpected downtime, it's crucial to synchronize your validator's positions to maintain consensus. Failure to do so may result in missing orders/positions, leading to a lower v_trust score.

## Purpose

The script detailed below regenerates the `validation/*` directory by fetching the latest positions, eliminations, and plagiarism scores from the mainnet, ensuring your validator remains in sync with the network.

## Steps

1. **Update PTN**: Ensure your Proprietary Trading Network (PTN) is up-to-date by executing `git pull origin main` in the `proprietary-trading-network` directory.
2. **Stop Validator**: Temporarily halt your validator with PM2 using `pm2 stop sn8`.
3. **Download Positions**: Visit [Taoshi Dashboard](https://dashboard.taoshi.io/) and click the "Download Positions" button to obtain a file named `validator_checkpoint.json`. Rename the file if necessary.
4. **Prepare for Restoration**: Move `validator_checkpoint.json` to the root level of the `proprietary-trading-network` directory.
5. **Run Restoration Script**: Execute `restore_validator_from_backup.py` within the `proprietary-trading-network` directory. Successful restoration is indicated by:
    ```
    2024-03-25 01:05:36.660 | INFO | regeneration complete in 1.25 seconds
    ```
6. **Verify Backups**: Confirm the creation of a `backups/` directory containing previous data (positions, eliminations, plagiarism). For example:
    ```
    (venv) jbonilla@MacBook-Air prop-net % ls -lh backups/20240325_010535
    total 16
    -rw-r--r--  1 jbonilla staff 164B Mar 25 01:05 eliminations.json
    drwxr-xr-x 148 jbonilla staff 4.6K Mar 25 00:36 miners
    -rw-r--r--  1 jbonilla staff 165B Mar 25 01:05 plagiarism.json
    ```
    If restoration fails, consult the failure log for troubleshooting steps.
7. **Restart Validator**: Resume your PM2 processes with `pm2 start sn8`.

## Future Improvements

We're committed to refining this process through automation and your feedback. Please report any issues encountered during restoration.
