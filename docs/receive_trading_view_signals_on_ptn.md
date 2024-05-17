# How to Compete on PTN Using TradingView Signals (For By Hand Traders or Strategies)

PTN is a data science competition in which you contribute algorithmic or manual trading signals to predict various financial markets, creating actionable signals for third parties.

We have a comprehensive [README](https://github.com/taoshidev/proprietary-trading-network) and a user-friendly [home page](https://www.taoshi.io/ptn) that provides all the necessary information. Additionally, you can join our active and supportive Taoshi Community Hub on Discord, where our team and fellow participants are always ready to guide and assist you.

## Quick Start Guide

Ready to dive in? Our quick start guide is designed to get you up and running with the Trading View signals in no time. Follow these simple steps, and you'll receive your first signal before you know it.

1. **Sign up for a Digital Ocean account.** Though no payment will be taken you will need to add a payment method.
2. **Deploy a Virtual Machine (Droplet or VM).** Once you sign up, you'll land on your Control Panel. You'll want to select "Deploy a virtual machine."
3. Once deployed, you can **Install Taoshi's Proprietary Trading Network** (PTN). Be sure to create a `miner_secrets.json` file with a specified API key.

   **Note:** If you are familiar with the command line, you can install PTN through your terminal or by launching a Droplet Console.

4. After installation, you'll want to **Run a Signals Server** and test that your miner can receive signals by sending a **Sample Signal**.
5. Now that your miner is ready to receive signals, you can send **Trading View signals** to your server using your server's endpoint and custom API key.

Need more? Check below for more in-depth steps.

---

## Getting Started

### 1. Sign up for a Digital Ocean account.

Though no payment will be taken you will need to add a payment method.

### 2. Deploy a virtual machine (Droplet or VM).

Once you sign up, you'll land on your Control Panel. You'll want to select "Deploy a virtual machine."

Next, configure your VM. As always, set it up as you'd like; however, the recommended settings are as follows.

1. Select a location closest to you
2. Select **Ubuntu image**.
3. Select a **Basic Shared CPU**
4. Select "**Password**" as your authentication method

[![](https://i.imgur.com/C1CUYXN.png)](https://www.youtube.com/watch?v=2udT5aRMCZ8)

### 3. Install Taoshi's Proprietary Trading Network (PTN).

**Note:** Complete documentation is located [here](https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/miner.md).

1. **Go to your droplet click access and Launch Droplet Console.**
   A new window will pop up. This is direct access to your VM machine through the command line.

   [![](https://i.imgur.com/K8z8wTr.png)](https://youtu.be/o2gYL9gML3I?si=WnKXMTDISXezsd53)

   **NOTE**: This can also be done in the terminal on your machine.

2. Update your system and add development dependancies. Update the local package index

```bash
apt update
```

3. Upgrade installed packages to their latest available versions.

```
apt upgrade
```

4. Install necessary dependencies

```
apt install build-essential python3.12-venv
```

[![](https://i.imgur.com/MVKZEuX.png)](https://www.youtube.com/watch?v=HsdN92LyawQ)

5. Clone the PTN repository to your virtual machine.

```
git clone https://github.com/taoshidev/proprietary-trading-network.git
```

4. Change directory

```
cd proprietary-trading-network
```

5. Create Virtual Environment

```
python3 -m venv venv
```

6. Activate the Virtual Environment

```
. venv/bin/activate
```

7.  Disable pip cache

```
export PIP_NO_CACHE_DIR=1
```

8. Install dependencies

```
pip install -r requirements.txt
```

9. Create a local and editable installation

```
python3 -m pip install -e .
```

10. Create Miner Secrets
    Create `mining/miner_secrets.json` and replace xxxx with your API key. This API key is created by you and used for interaction verification between systems.

```js
{
	"api_key": "xxxx"
}
```

[![](https://i.imgur.com/gchGxvg.png)](https://youtu.be/tujYrGgMuK4)

Congrats, you've configured your miner. Let's get your Bittensor wallets set up.

### 4. Create your Miner Wallet

**Note:** This step can be skipped if you've already set up your miner wallets.

Your previously configured miner must be paired with a Bittensor address. To do so, follow these steps.

While still in your active virtual environment `(venv)` and in the `/proprietary-trading-network directory`, create a coldkey and hotkey for your miner wallet.

```bash
btcli wallet new_coldkey --wallet.name miner
```

Next create a hotkey.

```bash
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

[![](https://i.imgur.com/6FuNY5G.png)](https://youtu.be/lsAzEBmAn_8?si=Jb4hkFPGEq8qXMWJ)

**Important:** Attach the flag `--subtensor.network test` and `--netuid 116` on your `btcli` commands to target the testnet.

### 5. Register with PTN

Wallah, next, register your miner with PTN. You will be required to pay registration costs. Be sure to have a funded Bittensor wallet. To do so…

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default
```

Finally, check to see that your keys are successfully registered.

```bash
btcli wallet overview --wallet.name miner
```

[![](https://i.imgur.com/nDMe4vj.png)](https://youtu.be/jxMi76ME-R4?si=YgmXcbAlvyXF-p0_)

**Important:** Attach the flag `--subtensor.network test` and `--netuid 116` on your `btcli` commands to target the testnet.

### 6. Run your Miner

Run your newly created miner

```bash
python neurons/miner.py  --wallet.name miner --wallet.hotkey default --logging.debug
```

[![](https://i.imgur.com/54T0mL3.png)](https://youtu.be/cqJJmT9zszA?si=sbIFSE4Axph1eVl0)

**Important:** Attach the flag `--subtensor.network test` and `--netuid 116` on your `btcli` commands to target the testnet.

### 7. Run your signals server and send sample signal

While still in your **Droplet Console**, within the PTN directory and with a virtual environment activated, let’s set up a signals server to receive signals.

Use our signal server script.

```bash
sh run_receive_signals_server.sh
```

[![](https://i.imgur.com/JK3VHfl.png)](https://youtu.be/rNLHZLz6M-8?si=lkVrdxRvnjmUAuXN)

Test your set up by sending a sample signal. Launch a new **Droplet Console**, and within the PTN directory let’s ensure connect a test server to your recently created signals server.

Change to `mining` directory.

```bash
cd mining
```

Run the following command

```bash
python3 sample_signal_request.py
```

[![](https://i.imgur.com/PgwaSaj.png)](https://youtu.be/FgWi0q0lMcg?si=vSwfnbFgr9_vIYKg)

### 8. Create a TradingView Signal Alert

To send a signal from Trading View to PTN you will first need to get your Droplet public IP address.

1. Within Digital Ocean to go to your newly created Droplet.
2. Click on “Networking” located in the menu on the left.
3. Copy your public IP address.

Next, sign into TradingView. You will need a premium TradingView account send a signal via a webhook.

Once signed in, create a new alert. Paste in the following data, remembering to add your custom API Key

```json
{
  "api_key": "MY_CUSTOM_API_KEY_123",
  "order_type": "{{strategy.market_position}}",
  "trade_pair": "BTCUSD",
  "leverage": "{{strategy.order.comment}}"
}
```

In the Notifications tab, create a webhook and paste in your Public IP address. Ensure you prepend http and prepend port `80`.

[![](https://i.imgur.com/3Cvk8PW.png)](https://youtu.be/TPEjATuDMEg?si=uxC50wnG8uMf0P0H)

Congratulations, you’ve successfully set up a TradingView alert to PTN

## Live Example

Once your criteria is met, Trading View will send a signal to your webhook.

[![](https://i.imgur.com/ccc5FD6.png)](https://youtu.be/ZEDE_oKFH44?si=sQ6NcyD_xO2R4ymh)
