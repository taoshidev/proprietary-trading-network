# Miner

Basic Rules:
1. Miner must have a minimum of 10 trades. Before they reach 10 trades, the miner will be in the grace period for the first month. In this month, they will receive a small amount of TAO that will help them avoid getting deregistered.
2. Miner will be penalized if they are not providing consistent predictions to the system. The details of this may be found [here](https://github.com/taoshidev/proprietary-trading-network/blob/main/vali_objects/utils/position_utils.py).
3. Positions must be open for a minimum of one minute.
4. A miner can have a maximum of 200 positions open.
5. A miner's order will be ignored if placing a trade outside of market hours
6. A miner's order will be ignored if they are rate limited (maliciously sending too many requests)
7. A portfolio that falls more than 10% of the maximum value will be eliminated.
8. A portfolio that falls more than 5% in a single day from a daily max will be eliminated.

The goal of the miner is to make a consistently competitive trading strategy. We track the returns for each miner over the past 30 days, and use this information to build a score for them. If you have a few great trades on the morning, you could be pushed to the top of the rankings that same day. We use a historical decay to discourage miners from sitting on a single good trade, that decay dampens prior positive returns but also dampens losses, giving miners a new chance to participate on bad returns. The historical decay function used can be found [here](https://github.com/taoshidev/proprietary-trading-network/blob/main/vali_objects/scoring/historical_scoring.py).

We then rank the miners based on their augmented historical return, and distribute emissions based on an exponential decay function, giving significant priority to the top miners. Details of both scoring functions can be found [here](https://github.com/taoshidev/proprietary-trading-network/tree/main/vali_objects/scoring). The best way to get emissions is to have a consistently great trading strategy, which makes multiple transactions each week (the more the better).

On the mining side we've setup some helpful infrastructure for you to send in signals to the network. The script `mining/run_receive_signals_server.py` will launch a flask server to receive order signals.

We recommend using this flask server to send in signals to the network. To see an example of sending a signal into the server, use `mining/sample_signal_request.py`.

Once a signal is properly sent into the signals server, it is parsed and stored locally in `mining/received_signals` to prepare for processing by `neurons/miner.py`. From there, the core miner logic in `neurons/miner.py` will automatically look to send the signal to validators on the network, retrying on failure. Once the signal is successfully sent into the network and ack'd by validators, the signal is stored in `mining/processed_signals`. otherwise, it gets stored in `mining/failed_signals` with debug information about which validators didn't receive the signal.

The current flow of information is as follows:

1. Run `mining/run_receive_signals_server.py` and `neurons/miner.py` to receive and parse signals
2. Send order signals from your choice of data provider (TradingView, python script, manually running `mining/sample_signal_request.py`)
3. Allow the miner to automatically send in your signals to validators
4. Validators update your existing positions, or create new positions based on your signals
5. Validators track your positions returns
6. Validators review your positions to assess drawdown every few seconds to determine if a miner should be eliminated (see main README for more info)
7. Validators wait for you to send in signals to close out positions (FLAT)
8. Validators set weights based on miner returns every 5 minutes based on portfolio performance. 


Please keep in mind that only one order can be submitted per minute per trade pair. This limitation may interfere with certain HFT strategies. We suggest verifying your miner on testnet before running on mainnet. 

When getting set up, we recommend running `mining/run_receive_signals_server.py` and `mining/sample_signal_request.py` locally to verify that order signals can be created and parsed correctly.

After that, we suggest running `mining/run_receive_signals_server.py` and `mining/sample_signal_request.py` in conjunction with `neurons/miner.py` on testnet. Inspect the log outputs to ensure that validators receive your orders. Ensure you are on your intended enviornment add the appropriate testnet flags.

| Environment | Netuid |
| ----------- | -----: |
| Mainnet     |      8 |
| Testnet     |    116 |

The simplest way to get a miner to submit orders to validators is by manually running `mining/sample_signal_request.py`. However, we expect most top miners to interface their existing trading software with `neurons/miner.py` and `mining/run_receive_signals_server.py` to automatically send trade signals. We will be releasing more detailed guides on how to set up an automated trades soon.

**DANGER**

- Do not expose your private keys.
- Only use your testnet wallet.
- Do not reuse the password of your mainnet wallet.
- Make sure your incentive mechanism is resistant to abuse.
- Your incentive mechanisms are open to anyone. They emit real TAO. Creating these mechanisms incur a lock_cost in TAO.
- Before attempting to register on mainnet, we strongly recommend that you run a miner on the testnet.
- Miners should use real exchange prices directly for training and live data purposes. This should come from MT5 and CB Pro / Binance. They should not rely on the data sources validators are providing for prices, as the data is subject to change based on potential downtime and fallback logic. 

# System Requirements

- Requires **Python 3.10 or higher.**
- [Bittensor](https://github.com/opentensor/bittensor#install)

Below are the prerequisites for miners. You may be able to make a miner work off lesser specs but it is not recommended.

- 2 vCPU + 8 GB memory
- Run the miner using CPU

# Getting Started

## 1. Install PTN

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

Create `mining/miner_secrets.json` and replace xxxx with your API key.

```json
{
	"api_key": "xxxx"
}
```

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your miner.

The miner will be registered to the subnet specified. This ensures that the miner can run the respective miner scripts.

Create a coldkey and hotkey for your miner wallet.

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

## 2a. (Optional) Getting faucet tokens

Faucet is disabled on the testnet. Hence, if you don't have sufficient faucet tokens, ask the Bittensor Discord community for faucet tokens.

## 3. Register keys

This step registers your subnet miner keys to the subnet, giving it the first slot on the subnet.

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default
```

To register your miner on the testnet add the `--subtensor.network test` flag.

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

Check that your miner has been registered:

```bash
btcli wallet overview --wallet.name miner
```

To check your miner on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: 8 # or 116 on testnet
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
miner    default  196    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000        *      134  none  5HRPpSSMD3TKkmgxfF7Bfu67sZRefUMNAcDofqRMb4zpU4S6
1        1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                               Wallet balance: τ4.998999856
```

## 6. Run your Miner

Run the subnet miner:

```bash
python neurons/miner.py --netuid 8  --wallet.name miner --wallet.hotkey default --logging.debug
```

To run your miner on the testnet add the `--subtensor.network test` flag and `--netuid 116` flag.

You will see the below terminal output:

```bash
>> 2023-08-08 16:58:11.223 |       INFO       | Running miner for subnet: 8 on network: ws://127.0.0.1:9946 with config: ...
```

## 7. Get emissions flowing

Register to the root network using the `btcli`:

```bash
btcli root register
```

To register your miner to the root network on testnet use the `--subtensor.network test` flag.

Then set your weights for the subnet:

```bash
btcli root weights
```

To set your weights on testnet `--subtensor.network test` flag.

## 8. Stopping your miner

To stop your miner, press CTRL + C in the terminal where the miner is running.

# Running Multiple Miners

You may use multiple miners when testing if you pass a different port per registered miner.

You can run a second miner using the following example command:

```bash
python neurons/miner.py --netuid 116 --subtensor.network test --wallet.name miner2 --wallet.hotkey default --logging.debug --axon.port 8095
```

# Issues?

If you are running into issues, please run with `--logging.debug` and `--logging.trace` set so you can better analyze why your miner isn't running.
