# Regenerating Validator State

When initiating a validator for the first time or recovering from unexpected downtime, it's crucial to synchronize your validator's positions to maintain consensus. Failure to do so may result in missing orders/positions, leading to a lower VTRUST score.

## Purpose

The script detailed below regenerates the `validation/*` directory by fetching the latest trade data and metrics from the mainnet, ensuring your validator remains in sync with the network.


## Steps 
1. **Get Restoration File**: Ping a Taoshi team member in the Discord and they will send you a file `validator_checkpoint.json`.
2. **Prepare for Restoration**: Transfer the `validator_checkpoint.json` file to the root level of the `proprietary-trading-network` directory on your validator.
3. **Stop Validator**: Temporarily halt your validator with PM2 using `pm2 stop sn8 ptn`
4. **Run Restoration Script**: Within the `proprietary-trading-network` directory, execute:

    ```bash
    python3 restore_validator_from_backup.py
    ```

     Successful restoration is indicated by:
    ```
    2024-03-25 01:05:36.660 | INFO | regeneration complete in 1.25 seconds
    ```
     If restoration fails, consult the failure log for troubleshooting steps.


5. **Restart Validator**: Resume your PM2 processes with `pm2 start sn8` (will launch ptn automatically).


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
