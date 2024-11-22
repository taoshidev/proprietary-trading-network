# Validator

Your validator receives trade signals from miners and maintains a portfolio per miner with all their positions on disk in the `validation/miners` directory. 

Your validator will track portfolio returns using live price information. If a portfolio's value declines beyond the drawdown limits, the validator will eliminate that miner. Based on portfolio metrics such as omega score and return, weights get set to reward the best miners. Your validator will look to set weights every 5 minutes.

Validators detect & eliminate any sort of miner copying from the network. It does this by performing an analysis on every order received. If a miner is detected to be plagiarising off another miner, they will be eliminated from the network. The information on plagiarising miners is held in `validation/miner_copying.json`.

When a miner is eliminated due to exceeding drawdown limits, or being caught plagiarising they will end up in the `validation/eliminations.json` file. Only registered non-eliminated miners can be given weights. Once eliminated, a miner can no longer send requests to validators until they are deregistered by the network and then re-register. 



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

- Requires **Python 3.10.**
- [Bittensor](https://github.com/opentensor/bittensor#install)

Below are the prerequisites for validators. 
- 4 vCPU + 16 GB memory
- 1 TB balanced persistent disk
- 1000 TAO staked
- A Twelvedata API account. (https://twelvedata.com/) with "Pro 610". The free tier is sufficient for testnet usage.
<img src="https://github.com/taoshidev/proprietary-trading-network/assets/161871533/ab74b310-7d32-432e-aada-4e0da2cb7a04" width="300">

- A Polygon API account (https://polygon.io/) with "Currencies Starter" as well as "Indicies Advanced"
<img src="https://github.com/taoshidev/proprietary-trading-network/assets/161871533/37e525b9-ab43-48b5-95ab-10fdc0903a5d" width="300">

<img src="https://github.com/taoshidev/proprietary-trading-network/assets/161871533/e726ed33-a3d4-4f58-9270-ce201b51e1b1" width="300">

For mainnet applications, the paid tiers are required to stay in consensus (~$250/month as of the time of writing). For testnet, you may use the free tiers.

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

Note: You should disregard any warnings about updating Bittensor after this. We want to use the version specified in `requirements.txt`.

Create a local and editable installation

```bash
python3 -m pip install -e .
```

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your validator.

The validator will be registered to the subnet specified. This ensures that the validator can run the respective validator scripts.

Create a coldkey and hotkey for your validator wallet. A coldkey can have multiple hotkeys, so if you already have an existing coldkey, you should create a new hotkey only. Be sure to save your mnemonics!

```bash
btcli wallet new_coldkey --wallet.name <wallet>
btcli wallet new_hotkey --wallet.name <wallet> --wallet.hotkey <validator>
```

You can list the local wallets on your machine with the following.

```bash
btcli wallet list
```

## 2a. Getting Testnet TAO

### Discord ###

Please ask the Bittensor Discord community for testnet TAO. This will let you register your validators(s) on Testnet.

Please first join the Bittensor Discord here: https://discord.com/invite/bittensor

Please request testnet TAO here: https://discord.com/channels/799672011265015819/1190048018184011867

Bittensor -> help-forum -> requests for testnet tao

## 3. Register keys

This step registers your subnet validator keys to the subnet, giving it the first slot on the subnet.

```bash
btcli subnet register --wallet.name <wallet> --wallet.hotkey <validator>
```

To register your validator on the testnet add the `--subtensor.network test` and `--netuid 116` flags.

Follow the below prompts:

```bash
>> Enter netuid (0): # Enter the appropriate netuid for your environment (8 for the mainnet)
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
btcli wallet overview --wallet.name <wallet>
```

To check your validator on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: 8 # or 116 on testnet
COLDKEY    HOTKEY    UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
wallet     validator 197    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                56  none  5GKkQKmDLfsKaumnkD479RBoD5CsbN2yRbMpY88J8YeC5DT4
1          1         1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                                Wallet balance: τ0.000999999
```

## 6. Running a Validator

### Overview

This guide provides instructions for running the validator using our automatic updater script, `run.sh`. It also introduces two optional flags

1. The `--start-generate` flag, which enables the generation of JSON files corresponding to trade data. These files can be sold to customers using the Request Network (further instructions pending).
2. The `--autosync` flag, which allows you to synchronize your data with a validator trusted by Taoshi (strong recommend enabling this flag to maintain validator consensus)

### Prerequisites

Before running a validator, follow these steps:

1. Ensure PTN is [installed](#getting-started).
2. Install [pm2](https://pm2.io) and the [jq](https://jqlang.github.io/jq/) package on your system.

```bash
npm install -g pm2
```
```bash
brew install jq
```

3. Create a `secrets.json` file in the root level of the PTN repo to include your TwelveData API key as shown below:

```json
{
  "twelvedata_apikey": "YOUR_API_KEY_HERE",
  "polygon_apikey": "OTHER_API_KEY_HERE",
  "tiingo_apikey": "OTHER_API_KEY_HERE"
}
```

- Obtain API keys by signing up at data providers' websites.
- Be careful to format your file as shown above or errors will be thrown when running your validator. Don't forget the comma!

### Using `run.sh` Script

1. **Mainnet Execution**: Run the validator on the mainnet by executing the following command. Include/exclude the `[--start-generate]` and `[--autosync]` flags as needed:
    ```bash
    $ pm2 start run.sh --name sn8 -- --wallet.name <wallet> --wallet.hotkey <validator> --netuid 8 [--start-generate] [--autosync]
    ```
   
2. **Testnet Execution**: For testnet operations with optional data generation, use this command:
    ```bash
    $ pm2 start run.sh --name sn8 -- --wallet.name <wallet> --wallet.hotkey <validator> --netuid 116 --subtensor.network test [--start-generate]
    ```

These commands initialize two PM2 processes:
   - **Validator Process**: Default name `ptn`
   - **Autoupdate Process**: Named `sn8`, which checks for and applies updates every 30 minutes.



### Pitfalls

- When running on the testnet, it is crucial to include the `--subtensor.network test` and `--netuid 116` flags to ensure proper configuration.
- Details on how to sell the generated trade data via the Request Network will be provided when available.

### Synchronizing your validator

Using the `--autosync` flag will allow your validator to synchronize with a trusted validator automatically.

However, we understand some validators want strict control and the ability to scrutinize all data changes.
In this case, we provide an alternative restore mechanism that essentially does a "nuke and force rebuild". 
 To use this manual restore mechanism, please follow the steps [here](https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/regenerating_validator_state.md) for performing the synchronization.

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


## 8. Relaunching run.sh

You will need to do this if you want to change any runtime configuration to run.sh such as adding or removing the `--start-generate`/ `--autosync` flags. Prepare your new `pm2 start run.sh ...` command before proceeding to minimize downtime.

Login to validator and cd into the PTN repo
```bash
cd proprietary-trading-network/
```
Active venv
```bash
. venv/bin/activate
```
Stop + Delete running pm2 processes
```bash
pm2 stop sn8 ptn
```
```bash
pm2 delete sn8 ptn
```
Run new run.sh command (USE YOUR OWN COMMAND)
```bash
pm2 start run.sh ...
```
Save configs
```bash
pm2 save
```
Verify that the ptn and sn8 pm2 processes have status "online" and are running smoothly
```
pm2 status
```
```
pm2 log
```
# Testing

You can begin testing PTN on the testnet with netuid 116. You can do this by using running:

```bash
python neurons/validator.py --netuid 116 --subtensor.network test --wallet.name <wallet> --wallet.hotkey <validator>
```
Note this won't launch the autoupdater. To launch with the autoupdater, use the run.sh command.

## 9. Pitfall Prevention

1. When running a validator in certain cloud environments such as Runpod, you may not have your Bittensor default port open (8091). This will cause your validator to be unable to communicate with miners and thus have a low VTRUST as your validator isn't receiving the latest orders. In order to correct this issue, explicitly open a tcp port, and pass this as an arugment with `--axon.port <YOUR_OPEN_PORT>`

2. Do not use share API keys across multiple validators/scripts. Each API key corresponds to one allowed websocket connection. Using the API keys across multiple scripts will lead to rate limits and failures on your validator. 

3. If you see an a ```JSONDecodeError``` exception when running your validator, ensure you secrets.json file is correctly formatted with proper commas.  
