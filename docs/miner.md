# Miner

Our miners act like traders. To score well and receive incentive, they place **orders** on our system against different trade pairs. The magnitude of each order is determined by its **leverage**, which can be thought of as the percentage of the portfolio used for the transaction. An order with a leverage of 1.0x indicates that the miner is betting against their entire portfolio value.

The first time a miner places an order on a trade pair, they will open a **position** against it. The leverage and directionality of this position determines the miner's expectation of the trade pair's future movement. As long as this position is open, the miner is communicating an expectation of continued trade pair movement in this direction. There are two types of positions: **LONG** and **SHORT**. 

A long position is a bet that the trade pair will increase, while a short position is a bet that the trade pair will decrease. Even if the overall position is LONG, a miner can submit a number of orders within this position to manage their risk exposure by adjusting the leverage. SHORT orders on a long position will reduce the overall leverage of the position, reducing the miner's exposure to the trade pair. LONG orders on a long position will increase the overall leverage of the position, increasing the miner's exposure to the trade pair.

## Basic Rules
1. Your miner will start in the challenge period upon entry. This 60-day period will require your miner to demonstrate consistent performance, after which they will be released from the challenge period, which may happen before 60 days has expired. In this month, they will receive a small amount of TAO that will help them avoid getting deregistered. The minimum requirements to pass the challenge period:
   - 2% Total Return
   - 5% Max Drawdown 
   - No single day’s change in portfolio value should exceed 20% of your total 60-day return.   
   - A single position should not account for more than 25% of your total return.
2. Miner will be penalized if they are not providing consistent predictions to the system or if their drawdown is too high. The details of this may be found [here](https://github.com/taoshidev/proprietary-trading-network/blob/main/vali_objects/utils/position_utils.py).
3. A miner can have a maximum of 200 positions open.
4. A miner's order will be ignored if placing a trade outside of market hours.
5. A miner's order will be ignored if they are rate limited (maliciously sending too many requests)
6. There is a 10-second cooldown period between orders, during which the miner cannot place another order.

## Scoring Details

The primary scoring mechanic in our system is *Risk Adjusted Returns*. We look at all of your positions in the prior lookback period, 90 days, and evaluate the returns from these positions. Notably, to determine the return of a miner, we look at the returns from closed positions and from open positions in loss. In filtering for open positions, we will also filter against any positions which have been open for more than 90 days. If they are in loss and still open, they will count against your score. We do this to avoid the scenario where a losing position is never closed, and the miner is able to avoid the penalty associated with this loss.

While our primary scoring mechanic is returns, consistency plays a substantial role in scoring our miners as we look to prioritize miners with a consistent track record of success. Additionally, we have a layer of costs and penalties baked into PTN, to simulate the real costs of trading.

There are two primary systems which live in parallel to give us a stronger perspective on the quality of our miners: _Positions_ and _Portfolio Value_.

### Scoring Metrics

We use four scoring metrics to evaluate miners based on their mid trade scores: **Short Term Risk Adjusted Returns**, **Long Term Risk Adjusted Returns**, **Sharpe** and **Omega**.

We measure miner risk as their maximum portfolio drawdown, the largest drop in value seen while we have been tracking the behavior of the miner. We use a blend of recently seen max drawdown values and historically likely values to make this determination, with the most recent values having the most weight. Details on this mechanic may be found in [our proposal 9](https://docs.taoshi.io/tips/p9/).

To find the risk adjusted return, we take the product of all positional returns as the current miner return. We then divide this by the drawdown term. If, for example, a miner has a total 90-day return of 7.5% and a drawdown of 2.5%, their long term risk adjusted return would be 3.0.

_Short term returns_ look at positions opened in the prior 90 days, but closed in the last 5 days. Like the long term returns, these use losing positions to calculate the return.

The _sharpe ratio_ will look at the positional return divided by the standard deviation of the returns. To avoid gaming on the bottom, a minimum value of 0.5% is used for the standard deviation.

The _omega ratio_ is a measure of the winning trades versus the losing trades. It serves as a useful proxy for the risk to reward ratio the miner is willing to take with each trade. Like the sharpe ratio, we will use a minimum value of 0.5% for the denominator.

| Metric                     | Scoring Weight |
|----------------------------|----------------|
| Long Term Realized Returns | 100%           |
| Short Term Realized Returns| 25%            |
| Sharpe Ratio               | 25%            |
| Omega Ratio                | 25%            |

### Scoring Penalties

There are four primary penalties in place for each miner:

1. **Max Positional Return**: A single position should not represent more than 15% of total realized return.
2. **Realized Return Distribution**: No more than 30% of the miner's realized returns should be from positions all closed in a single week.
3. **Max Portfolio Value Change - Daily**: A single day of trading should not represent more than 20% of the total unrealized return.
4. **Max Portfolio Value Change - Biweekly**: A single two-week period should not account for more than 35% of total unrealized return.

Portfolio value is tracked in realtime against positions, regardless of if they are closed or open. If the measured volatility on the portfolio value is too high relative to the total returns from the miner, we will flag them as inconsistent, even if their closed positions meet the requirements. This is meant to protect from the scenario where most of a miner's value comes from a single interval, but their positions may close over a longer period. Full details on the logic associated with each proposal may be found in [proposal 9](https://docs.taoshi.io/tips/p9/).

### Fees and Transaction Costs
We want to simulate real costs of trading for our miners, to make signals from PTN more valuable outside our platform. To do this, we have incorporated two primary costs: **Transaction Fees** and **Cost of Carry**. 

Transaction fees are proportional to the leverage used. The higher the leverage, the higher the transaction fee. We use cumulative leverage to determine the transaction fee, so any order placed on a position will increase the fees proportional to the change in leverage.

Cost of carry is reflective of real exchanges, and how they manage the cost of holding a position overnight. This rate changes depending on the asset class, the logic of which may be found in [our proposal 4](https://docs.taoshi.io/tips/p4/).

##### Implementation Details
| Market  | Fee Period     | Times                   | Rates Applied       | Triple Wednesday |
|---------|----------------|-------------------------|---------------------|------------------|
| Forex   | 24h            | 21:00 UTC               | Mon-Fri             | ✓                |
| Crypto  | 8h             | 04:00, 12:00, 20:00 UTC | Daily (Mon-Sun)     |                  |
| Indices | 24h            | 21:00 UTC               | Mon-Fri             | ✓                |

The magnitude of the fees will reflect the following distribution:

| Market  | Base Rate (Annual) | Daily Rate Calculation     |
|---------|--------------------|----------------------------|
| Forex   | 3%                 | 0.008% * Max Seen Leverage |
| Crypto  | 10.95%             | 0.03% * Max Seen Leverage  |
| Indices | 5.25%              | 0.014% * Max Seen Leverage |

### Leverage Limits
We also set limits on leverage usage, to ensure that the network has a level of risk protection and mitigation of naive strategies. The [positional leverage limits](https://docs.taoshi.io/tips/p5/) are as follows:

| Market  | Leverage Limit |
|---------|----------------|
| Forex   | 0.1x - 5x      |
| Crypto  | 0.01x - 0.5x   |
| Indices | 0.1x - 5x      |

We also implement a [portfolio level leverage limit](https://docs.taoshi.io/tips/p10/), which is the sum of all the leverages from each open position. This limit is set at 10x a "typical" position, where a typical position would be 1x leverage for forex/indices and 0.1x leverage for crypto. You can therefore open 10 forex positions at 1x leverage each, 5 forex positions at 2x leverage each, 5 forex positions at 1x and 5 crypto positions at 0.1x, etc.

## Incentive Distribution
The miners are scored in each of the categories above based on their prior positions over the lookback period. Penalties are then applied to these scores, and the miners are ranked based on their total score. Percentiles are determined for each category, with the miner's overall score being reduced by the full scoring weight if they are the worst in a category.

For example, if a miner is last place in the long term realized returns category, they will receive a 0% score for this category. This will effectively reduce their score to 0, and they will be prioritized during the next round of deregistration.

We distribute on an exponential decay, with the top 40% of miners receiving 90% of emissions.

# Mining Infrastructure

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
8. Validators set weights based on miner returns every 5 minutes based on portfolio performance with both open and closed positions. 

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

- Requires **Python 3.10.**
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

Note: You should disregard any warnings about updating Bittensor after this. We want to use the version specified in `requirements.txt`.

Create a local and editable installation

```bash
python3 -m pip install -e .
```

Create `mining/miner_secrets.json` and replace xxxx with your API key. The API key value is determined by you and needs to match the value in `mining/sample_signal_request.py`.

```json
{
	"api_key": "xxxx"
}
```

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your miner.

The miner will be registered to the subnet specified. This ensures that the miner can run the respective miner scripts.

Create a coldkey and hotkey for your miner wallet. A coldkey can have multiple hotkeys, so if you already have an existing coldkey, you should create a new hotkey only. Be sure to save your mnemonics!

```bash
btcli wallet new_coldkey --wallet.name <wallet>
btcli wallet new_hotkey --wallet.name <wallet> --wallet.hotkey <miner>
```

You can list the local wallets on your machine with the following.

```bash
btcli wallet list
```

## 2a. Getting Testnet TAO

### Discord ###

Please ask the Bittensor Discord community for testnet TAO. This will let you register your miner(s) on Testnet.

Please first join the Bittensor Discord here: https://discord.com/invite/bittensor

Then request testnet TAO here: https://discord.com/channels/799672011265015819/1190048018184011867

Bittensor -> help-forum -> requests for testnet tao

## 3. Register keys

This step registers your subnet miner keys to the subnet, giving it the first slot on the subnet.

```bash
btcli subnet register --wallet.name <wallet> --wallet.hotkey <miner>
```

To register your miner on the testnet add the `--subtensor.network test` and `--netuid 116` flags.

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

Check that your miner has been registered:

```bash
btcli wallet overview --wallet.name <wallet>
```

To check your miner on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: 8 # or 116 on testnet
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
wallet   miner    196    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000        *      134  none  5HRPpSSMD3TKkmgxfF7Bfu67sZRefUMNAcDofqRMb4zpU4S6
1        1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                               Wallet balance: τ4.998999856
```

## 6. Run your Miner

Run the subnet miner:

```bash
python neurons/miner.py --netuid 8  --wallet.name <wallet> --wallet.hotkey <miner> --logging.debug
```

To run your miner on the testnet add the `--subtensor.network test` flag and override the netuuid flag to `--netuid 116`.

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
python neurons/miner.py --netuid 116 --subtensor.network test --wallet.name <wallet> --wallet.hotkey <miner2> --logging.debug --axon.port 8095
```

# Issues?

If you are running into issues, please run with `--logging.debug` and `--logging.trace` set so you can better analyze why your miner isn't running.
