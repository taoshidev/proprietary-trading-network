# Regenerating Validator State

If 
1. You are running a validator for the first time
2. Your validator experiences unexpected downtime.

You will need to synchronize your validator's positions using these steps to stay in consensus. Otherwise your missing orders/positions will lead your validator to have low v_trust.

## Purpose

This script will regenerate the `validation/miners` directory, filling it with the latest positions
and orders from mainnet.

## Steps

1. Ensure you are on the latest version of PTN. You can check this by running `git pull origin main` 
inside the `proprietary-trading-network` directory.
2. Stop your validator. You can do this by using the pm2 process. `pm2 stop sn8` and `pm2 stop ptn` if you
are using the standard setup provided in the readme.
3. Move your existing `validation/miners` dir. You can call it `validation/miners_bkp`. Once moved, you should
have no `validation/miners` directory.
4. Go to https://dashboard.taoshi.io/
5. Press the "Download Positions" button in the top right corner. This should provide you with
a downloaded file called `miner_positions.json`. If the file is not called `miner_positions.json` please
rename it so that it is.
6. Move this file to the `proprietary-trading-network` dir
7. Run `regenerate_miner_positions.py` inside the `proprietary-trading-network` dir

You should see that a new `validation/miners` directory has been created and inside there 
should be all of the positions and orders downloaded from the dashboard from 
the `miner_positions.json` file

8. Start your pm2 processes again, `pm2 start sn8` and `pm2 start ptn`
