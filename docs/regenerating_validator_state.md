# Regenerating Validator State

If your validator goes down and you fall out of consensus due to missing orders/positions, 
you can always regenerate state following this script.

## Purpose

This script will regenerate the `validation/miners` directory, filling it with the latest positions
and orders from mainnet.

## Steps

1. Ensure you are on the latest version of PTN. You can check this by running `git pull origin main` 
inside the `proprietary-trading-network` directory.
2. Stop your validator. You can do this by using the pm2 process. `pm2 stop sn8` and `pm2 stop ptn` if you
are using the standard setup provided in the readme.
3. Move the file to your server and place in the `proprietary-trading-network` directory
4. Backup your existing `validation/miners` dir. You can call it `validation/miners_bkp`.
5. Go to https://dashboard.taoshi.io/
6. Press the "Download Positions" button in the top right corner. This should provide you with
a downloaded file called `miner_positions.json`
7. Move this file to the `proprietary-trading-network` dir
8. Run `regenerate_miner_positions.py`
9. Start your pm2 processes again, `pm2 start sn8` and `pm2 start ptn`

You should see that a new `validation/miners` directory has been created and inside there 
should be all of the positions and orders downloaded from the dashboard from 
the `miner_positions.json` file.