# Validator

Your validator receives signals from miners when they have one prepared. Your validator will perform core logic checks to ensure only registered non-eliminated miners can be given weights. Your validator will look to set weights every 30 minutes.

Your validator will store all information related to miners on disk as it doesn't take up much space. All information created is stored in the `validation` directory.

When the validator receives signals, they are converted to orders. These orders make up positions which are stored are safely stored as files in the `validation/miners` directory on a per miner basis.

The core logic looks to detect & eliminate any sort of miner copying from the network. It does this by performing an analysis on every order received. If a miner is detected to be plagiarising off another miner, they will be eliminated from the network. The information on plagiarising miners is held in `validation/miner_copying.json`.

When a miner is eliminated due to exceeding drawdown limits, or being caught plagiarising they will end up in the `validation/eliminations.json` file.

This tutorial shows how to run a PTN Validator.

**IMPORTANT**

Before attempting to register on mainnet, we strongly recommend that you run a validator on the testnet. To do ensure you add the appropriate testnet flags.

| Environment | Netuid |
| ----------- | -----: |
| Mainnet     |      8 |
| Testnet     |    116 |

Your incentive mechanisms running on the mainnet are open to anyone. They emit real TAO. Creating these mechanisms incur a lock_cost in TAO.

**DANGER**

- Do not expose your private keys.
- Only use your testnet wallet.
- Do not reuse the password of your mainnet wallet.
- Make sure your incentive mechanism is resistant to abuse.

# System Requirements

- Requires **Python 3.10 or higher.**
- [Bittensor](https://github.com/opentensor/bittensor#install)

Below are the prerequisites for validators. You may be able to make a validator work off lesser specs but it is not recommended.

- 2 vCPU + 8 GB memory
- 100 GB balanced persistent disk
- A "Grow" Twelvedata API account (https://twelvedata.com/)

# Getting Started

Clone repository

```bash
git clone https://github.com/taoshidev/proprietary-trading-network.git
```

Change directory

```bash
cd proprietary-trading-network
```

Create Virtual Environment

```bash
python3 -m venv venv
```

Activate a Virtual Environment

```bash
. venv/bin/activate
```

Disable pip cache

```bash
export PIP_NO_CACHE_DIR=1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a local and editable installation

```bash
python3 -m pip install -e .
```

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your validator.

The validator will be registered to the subnet specified. This ensures that the validator can run the respective validator scripts.

Create a coldkey and hotkey for your validator wallet.

```bash
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

## 2a. (Optional) Getting faucet tokens

Faucet is disabled on the testnet. Hence, if you don't have sufficient faucet tokens, ask the Bittensor Discord community for faucet tokens.

## 3. Register keys

This step registers your subnet validator keys to the subnet, giving it the first slot on the subnet.

```bash
btcli subnet register --wallet.name validator --wallet.hotkey default
```

To register your validator on the testnet add the `--subtensor.network test` flag.

Follow the below prompts:

```bash
>> Enter netuid (0): # Enter the appropriate netuid for your environment
Your balance is: # Your wallet balance will be shown
The cost to register by recycle is τ0.000000001 # Current registration costs
>> Do you want to continue? [y/n] (n): # Enter y to continue
>> Enter password to unlock key: # Enter your wallet password
>> Recycle τ0.000000001 to register on subnet:8? [y/n]: # Enter y to register
📡 Checking Balance...
Balance:
  τ5.000000000 ➡ τ4.999999999
✅ Registered
```

## 4. Check that your keys have been registered

This step returns information about your registered keys.

Check that your validator key has been registered:

```bash
btcli wallet overview --wallet.name validator
```

To check your validator on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: 8 # or 116 on testnet
COLDKEY    HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
validator  default  197    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                56  none  5GKkQKmDLfsKaumnkD479RBoD5CsbN2yRbMpY88J8YeC5DT4
1          1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                                Wallet balance: τ0.000999999
```

## 6. Running a Validator

### Using Provided Scripts

These validators run and update themselves automatically.

To run a validator, follow these steps:

1. Ensure PTN is [installed](#getting-started).
2. Install [pm2](https://pm2.io) and the [jq](https://jqlang.github.io/jq/) package on your system.
3. Create a `secrets.json` file in the root level of the PTN repo to include your TwelveData API key as shown below:

```json
{
  "twelvedata_apikey": "YOUR_API_KEY_HERE"
}
```

- Replace `YOUR_API_KEY_HERE` with your actual TwelveData API key.
- Obtain an API key by signing up at TwelveData's website. The free tier is sufficient for testnet usage. For mainnet applications, a premium tier subscription is recommended.

4. Run the `run.sh` script, which will run your validator and pull the latest updates as they are issued.

```bash
$ pm2 start run.sh --name sn8 -- --wallet.name <wallet> --wallet.hotkey <hotkey> --netuid 8
```

This will run two PM2 process:

1. A process for the validator, called sn8 by default (you can change this in run.sh)
2. A process for the `run.sh` script. The script will check for updates every 30 minutes, if there is an update, it will pull, install, restart tsps, and restart itself.

### Manually

If there are any issues with the run script or you choose not to use it, run a validator manually.

```bash
python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>

```

To run your validator on the testnet add the `--subtensor.network test` flag and `--netuid 116` flag.

You can also run your script in the background. Logs are stored in `nohup.out`.

```bash
nohup python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey> &
```

To run your validator on the testnet add the `--subtensor.network test` flag and `--netuid 116` flag.

## 7. Get emissions flowing

Register to the root network using the `btcli`:

```bash
btcli root register
```

To register your validator to the root network on testnet use the `--subtensor.network test` flag.

Then set your weights for the subnet:

```bash
btcli root weights
```

To set your weights on testnet `--subtensor.network test` flag.

## 8. Stopping your validator

To stop your validator, press CTRL + C in the terminal where the validator is running.

# Testing

You can begin testing PTN on the testnet with netuid 116. You can do this by using running:

```bash
python neurons/validator.py --netuid 116 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug
```

This will have the validator run and test against historical data instead of live. Don't use this flag if you only want to test against live data.
