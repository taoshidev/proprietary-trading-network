<p align="center">
  <a href="https://taoshi.io">Website</a>
  ·
  <a href="#installation">Installation</a>
  ·  
  <a href="https://dashboard.taoshi.io/">Dashboard</a>
  ·
  <a href="https://twitter.com/taoshiio">Twitter</a>
    ·
  <a href="https://twitter.com/taoshiio">Bittensor</a>
</p>

---

<details>
  <summary>Table of contents</summary>
  <ol>
    <li>
      <a href="#bittensor">Bittensor</a>
      <ol>
        <li>
          <a href="#subnets">Subnets</a>
        </li>
        <li>
          <a href="#miners">Miners</a>
        </li>
        <li>
          <a href="#validators">Validators</a>
        </li>
      </ol>
    </li>
    <li><a href="#prop-subnet">Proprietary Trading Network</a></li>
    <li><a href="#features">Featuers</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li>
          <a href="#running-a-validator">Running a Validator</a>
        </li>
        <li>
          <a href="#running-a-miner">Running a Miner</a>
        </li>
      </ol>
    </li>
    <li><a href="#building-a-model">Building A Model</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

---

# Bittensor

Bittensor is a mining network, similar to Bitcoin, that includes built-in incentives designed to encourage
computers to provide access to machine learning models in an efficient and censorship-resistant manner.
Bittensor is compromised of Subnets, Miners, and Validators.

**Explain Like I'm Five**

Bittensor is an API that connects machine learning models and incentivizes correctness through the power of the
blockchain.

### Subnets

Subnets are decentralized networks of machines that collaborate to train and serve machine learning models.

### Miners

Miners run machine learning models. They fulfill requests from the Validators.

### Validators

Validators query and prompt the Miners. Validators also validate miner requests. Validators are also storefronts for
data.

# Proprietary Trading Subnet

This repository contains the code for the Proprietary Trading Network (PTN) developed by Taoshi.

PTN receives signals from quant and deep learning machine learning trading systems to deliver the world's
most complete trading signals across a variety of asset classes.

## How does it work?

PTN is the most challenging & competitive network in the world. Our miners need to provide futures based signals (
long/short)
that are highly efficient and effective across various markets to compete (forex, crypto, indices). The top miners are
those that provide the most returns, while never exceeding certain drawdown limits.

### Rules

1. Miners can submit LONG, SHORT, or FLAT signal into the network 
2. Miners can only open 1 position per trade pair/symbol at a time 
3. Positions are uni-directional. Meaning, if a position starts LONG (the first order it receives is LONG), 
it can't flip SHORT. If you try and have it flip SHORT (using more leverage SHORT than exists LONG) it will close out 
the position. You'll then need to open a second position which is SHORT with the difference. 
4. You can take profit on an open position using LONG and SHORT. Say you have an open LONG position with .75x 
leverage and you want to reduce it to a .5x leverage position to start taking profit on it. You would send in a SHORT signal
of size .25x leverage to reduce the size of the position. LONG and SHORT signals can be thought of working in opposite 
directions in this way.
5. You can close out a position by sending in a FLAT signal. 
6. max drawdown is determined every minute. If you go beyond **5% max drawdown on daily close**, or **10% at any point in time** your miner will be eliminated. 
Eliminated miners won't necessarily be immediately deregistered, they'll need to wait to be deregistered based on registrations & immunity period. 
7. If a miner copies another miner's order repeatedly they will be eliminated. When any order is submitted, analysis
on the integrity of the order is performed. If the order is deemed to be plagiarising it is flagged by the network. Repeated
occurrence leads to removal from the network.
8. There is a fee per trade pair position. Crypto has a 0.3% per position, forex has 0.03%, indices have 0.05%.
9. There is a minimum registration fee of 5 TAO on the mainnet subnet.
10. There is an immunity period of 9 days to help miners submit orders to become competitive with existing miners.
11. The miners who can provide the most returns over a 30 day rolling lookback period are provided the most incentive.

With this system only the world's best traders & deep learning / quant based trading systems can compete.

# Features

🛠️&nbsp;Open Source Strategy Building Techniques (In Our Taoshi Community)<br>
🫰&nbsp;Signals From a Variety of Asset Classes - Forex, Indices, Crypto<br>
📈&nbsp;Higher Payouts<br>
📉&nbsp;Lower Registration Fees<br>
💪&nbsp;Superior Cryptocurrency Infrastructure<br>

# Prerequisites

Below are the prerequisites for validators and miners, you may be able to make miner and validator work off lesser
specs.

Requires **Python 3.10.**

**Validator**

- 2 vCPU + 8 GB memory
- 100 GB balanced persistent disk
- A Twelvedata API account to allow your validator to fetch live prices (https://twelvedata.com/)

**Miner**

- 2 vCPU + 8 GB memory
- Run the miner using CPU

# Installation

On Linux

```bash
# install git and subpackages
$ sudo apt install git-all

# install pip package manager for python 3
$ sudo apt install python3-pip

# install venv virtual environment package for python 3
$ sudo apt-get install python3-venv

# clone repo
$ git clone https://github.com/taoshidev/proprietary-trading-network.git

# change directory
$ cd proprietary-trading-network

# create virtual environment
$ python3 -m venv venv

# activate the virtual environment
$ . venv/bin/activate

# disable pip cache
$ export PIP_NO_CACHE_DIR=1

# install dependencies
$ pip install -r requirements.txt

# create a local and editable installation
$ python -m pip install -e .

```

# Usage

## Understanding Core Validator Logic & Files

Your validator receives signals from miners when they have one prepared. Your validator will perform core logic checks
to ensure only registered non-eliminated miners can be given weights. Your validator will look to set weights
every 30 minutes.

Your validator will store all information related to miners on disk as it doesn't take up much space. 
All information created is stored in the `validation` directory.

When the validator receives signals, they are converted to orders. These orders make up positions which are stored
are safely stored as files in the `validation/miners` directory on a per miner basis.

The core logic looks to detect & eliminate any sort of miner copying from the network. It does this by performing
an analysis on every order received. If a miner is detected to be plagiarising off another miner, they will be eliminated
from the network. The information on plagiarising miners is held in `validation/miner_copying.json`.

When a miner is eliminated due to exceeding drawdown limits, or being caught plagiarising 
they will end up in the `validation/eliminations.json` file.

## Running a Validator

### Using Provided Scripts

These validators run and update themselves automatically.

To run a validator, follow these steps:

1. [Install Prop Subnet.](#installation)
2. Install [PM2](https://pm2.io) and the (jq)[https://jqlang.github.io/jq/] package on your system.
3. Create a secrets.json file in the root level of the PTN repo to include your TwelveData API key as shown below:

```json
{
  "twelvedata_apikey": "YOUR_API_KEY_HERE"
}
```
- Replace YOUR_API_KEY_HERE with your actual TwelveData API key.
- Obtain an API key by signing up at TwelveData's website. The free tier is sufficient for testnet usage. For mainnet applications, a "Grow" tier subscription is required.

On Linux:

```bash
# update lists
$ sudo apt update

# JSON-processor
$ sudo apt install jq

# install npm
$ sudo apt install npm

# install pm2 globally
$ sudo npm install pm2 -g

# update pm2 process list
$ pm2 update
```

On MacOS:

```bash
# update lists
$ brew update

# JSON-processor
$ brew install jq

# install npm
$ brew install npm

# install pm2 globally
$ sudo npm install pm2 -g

# update pm2 process list
$ pm2 update
```

3. Be sure to install venv for the repo.

```bash
# /proprietary-trading-network

# create virtual environment
$ python3 -m venv venv

# activate virtual environment
$ source venv/bin/activate

# install packages
$ pip install -r requirements.txt
```

4. Run the `run.sh` script, which will run your validator and pull the latest updates as they are issued.

```bash
$ pm2 start run.sh --name sn8 -- --wallet.name <wallet> --wallet.hotkey <hotkey> --netuid 8
```

This will run two PM2 process:

1. A process for the validator, called sn8 by default (you can change this in run.sh)
2. And a process for the run.sh script (in step 4, we named it ptn). The script will check for updates every 30 minutes,
   if there is an update, it will pull, install, restart ptn, and restart itself.

### Manually

If there are any issues with the run script or you choose not to use it, run a validator manually.

```bash
$ python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>
```

You can also run your script in the background. Logs are stored in `nohup.out`.

```bash
$ nohup python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey> &
```

## Running a Miner

On the mining side we've setup some helpful infrastructure for you to send in signals to the network. You can run
`mining/run_receive_signals_server.py` which will launch a flask server. You can use this flask server to send in
signals to the network. To see an example of sending a signal into the server, checkout `mining/sample_signal_request.py`.

Once a signal is properly sent into the signals server, it is stored locally in `mining/received_signals` to 
prepare for processing. From there, the core miner logic will automatically look to send the signal into the network, 
retrying on failure. Once the signal is attempted to send into the network, the signal is stored in `mining/processed_signals`.

# Running on mainnet

You can run on mainnet by following the instructions in `docs/running_on_mainnet.md`.

If you are running into issues, please run with `--logging.debug` and `--logging.trace` set so you can better
analyze why your miner isn't running.

The current flow of information is as follows:

1. Send in your signals to validators
2. Validators update your existing positions, or create new positions based on your signals
3. Validators track your positions returns
4. Validators review your positions to assess drawdown every minute
4. Validators wait for you to send in signals to close out positions (FLAT)
5. Validators set weights based on miner returns every 30 minutes

# Building a strategy

We recommend joining our community hub via Discord to get assistance in building a trading strategy. We have partnerships
with both glassnode and LunarCrush who provide valuable data to be able to create an effective strategy. Analysis and information
on how to build a deep learning ML based strategy will continue to be discussed in an open manner by team Taoshi to help
guide miners to compete.

# Testing

You can begin testing on testnet netuid 116. You can follow the `docs/running_on_testnet.md` file inside the repo
to run on testnet.


---

# Contributing

For instructions on how to contribute to Taoshi, see CONTRIBUTING.md and Taoshi's code of conduct.

# License

Refer to the <a href='?tab=MIT-1-ov-file'>License</a> page for information about Taoshi's licensing.

Bittensor's source code in this repository is licensed under the MIT License.
