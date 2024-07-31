# Miner

Basic Rules:
1. Your miner will start in the challenge period upon entry. This 30 day period will require your miner to demonstrate consistent performance, after which they will be released from the challenge period, which may happen before 30 days has expired. In this month, they will receive a small amount of TAO that will help them avoid getting deregistered. The minimum requirements to pass the challenge period:
  - 2% Total Return
  - -3.08E-8 Time Averaged Sortino
  - 12 Volume Minimum Checkpoints
2. Miner will be penalized if they are not providing consistent predictions to the system or if their drawdown is too high. The details of this may be found [here](https://github.com/taoshidev/proprietary-trading-network/blob/main/vali_objects/utils/position_utils.py).
3. A miner can have a maximum of 200 positions open.
4. A miner's order will be ignored if placing a trade outside of market hours.
5. A miner's order will be ignored if they are rate limited (maliciously sending too many requests)


### Scoring Details

The open positions held by miners will be continuously evaluated based on their value changes. Any measured positive movement on the asset while tracked in a position will count as a gain for the miner. Any negative movements will be tracked as losses. Risk is defined as the sum volume of millisecond negative value change overseen during a position. Given that the price of assets fluctuates so quickly and has some level of noise, it is virtually impossible for an investment strategy to have zero risk. This is normal. An asset with zero return through the course of the day will still carry risk, although the gains and losses result in a product of 1.0. A higher leverage trade will increase the intensity of losses and of gains, but in this scenario the product sum will still be 1.0 as a return. With this increased leverage, there will be a higher volume of losses, and thus risk. You may augment the risk for a position by placing an order on the position, which might increase or decrease the leverage utilization. Please note that there is a 10 second cooldown period between orders. Additionally, we are requiring miners to hold positions for a minimum of 15 minutes on each 6 hour interval to qualify for scoring in that round.

In order to capture information at such a high resolution, we utilize checkpoints which track a miner's behavior over time. Each checkpoint has a target duration of 6 hours, after which the checkpoint is closed and a new checkpoint is opened. The checkpoint contains the aggregate of all gains and losses, as well as information on the duration of open positions held in the checkpoint and number of updates seen.

Each miner is compared to a baseline, the annual return rate of American Treasury Bills. This will consistently add a small amount of loss for the miner every millisecond. If the miner's Omega is less than 1 and log return less than 0, they were unable to beat the growth rate of treasury bills.

#### Scoring Metrics

We will use three scoring metrics to evaluate miners based on their mid trade scores: **Short Term Returns**, **Long Term Returns**, and **Omega**.

Short term Returns measure the pure value change that the miner experienced through the course of their positions. This will be similar to the prior position based system, although open positions will now also be evaluated. For this metric, we only consider the movement of the portfolio value from the prior three days.

Similar to the short term returns, long term returns are also going to measure the historical gains for a miner in determining their quality. Long term returns consider all portfolio value movement from the prior 45 days.

Omega will evaluate the magnitude of the positive asset changes over the magnitude of negative asset changes. Any score above 1 will indicate that the miner experienced a net gain through the course of their position. A higher omega value will result from:

- Higher magnitude positive value change
- Pure positive value change

The total score will result from the product of the Long Term Returns, Short Term Returns, and Omega, so the top miners in our system must perform well in all three metrics to receive substantial incentive. The relative weight of each term in the product sum is Long Term Returns: 1.0, Short Term Returns: 0.25, Omega: 0.05. The terms used to calculate the product are defined by ranking each metric against the other miners. As a simple example, if a miner is first place in both returns and last place in Omega, their total score would start at 1, multiply by 1 due to first place in long term returns and 1 due to short term returns. It would then multiply by (1 - 0.05) as they are the last place in Omega, so their final score would be 0.95.

#### Scoring Penalties

There are two primary penalties in place for each miner: Consistency and Drawdown.

The consistency penalty is meant to discourage miners who cannot deliver consistent performance over each 45 day period. To fully mitigate penalties associated with consistency, your miner should achieve the following metrics:
- Minimum of 18 days of open positions, of any volume.
- Your portfolio value change in any single six-hour checkpoint period should not exceed 50% of the total portfolio value change over the previous 45 days. If your portfolio value increases sharply and decreases back towards a total return of 1.0, the value change in one period may exceed your total portfolio value change and you are likely to experience a large penalty due to consistency.

The drawdown penalty is meant to both discourage miners from taking too much drawdown and benefit miners with low drawdown. To do this, the drawdown penalty is defined as 1 / MDD of the miner. This enables miners with low risk tolerance to be competitive with higher risk miners. Between 0.25% MDD and 1.5% MDD the drawdown penalty is only designed to normalize the returns of your miner relative to the risk. We apply a penalty below 0.25% MDD and above 1.5%, with the upper penalty linearly tapering to 0 as it gets closer to 5% MDD.

### Challenge Period Details

There are four primary requirements for a miner to pass the challenge period: Returns, Sortino and Volume Minimum Checkpoints. All of these metrics were set to be reasonably competitive with our currently successful miners' median values, such that by passing the challenge period the miner will be in a decently competitive stance. The checkpoint files used for the challenge period will also be used to score the miner against other successful miners after passing. The first three metrics are described above in the scoring details section.

The volume minimum checkpoint is defined as a checkpoint which meets a certain threshold of raw gains and losses. The threshold value for inclusion of the checkpoint as valid is 0.1. This means that a checkpoint with a gain of 0.05 and a loss of -0.05 would have an absolute sum of 0.1 and qualify. We are requiring 12 of these valid checkpoints to have been observed in order for the miner to pass the checkpoint qualifications.

### Distribution and Rankings
We then rank miners based on their historical performance, by determining their percentiles against their peers for each scoring category. We then distribute emissions based on an exponential decay function, giving significant priority to the top miners. Details of both scoring functions can be found [here](https://github.com/taoshidev/proprietary-trading-network/tree/main/vali_objects/scoring). The best way to get emissions is to have a consistently great trading strategy, which makes multiple transactions each week (the more the better). Capturing upside through timing and proper leverage utilization will yield the highest score in our system.

## Mining Infrastructure

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
