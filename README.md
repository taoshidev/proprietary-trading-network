<p align="center">
  <a href="https://taoshi.io">
    <img width="500" alt="taoshi - ptn repo logo" src="https://i.imgur.com/5hTsp97.png">
  </a>
</p>

<div align='center'>

[![Discord Chat](https://img.shields.io/discord/1163496128499683389.svg)](https://discord.gg/2XSw62p9Fj)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

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
    <li><a href="#proprietary-trading-network">Proprietary Trading Network</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#how-does-it-work">How does it work?</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#building-a-model">Building A Model</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>

  </ol>
</details>

---

<details id='bittensor'>
  <summary>What is Bittensor?</summary>

Bittensor is a mining network, similar to Bitcoin, that includes built-in incentives designed to encourage computers to provide access to machine learning models in an efficient and censorship-resistant manner. Bittensor is comprised of Subnets, Miners, and Validators.

> Explain Like I'm Five

Bittensor is an API that connects machine learning models and incentivizes correctness through the power of the blockchain.

### Subnets

Subnets are decentralized networks of machines that collaborate to train and serve machine learning models.

### Miners

Miners run machine learning models. They send signals to the Validators.

### Validators

Validators recieve trade signals from Miners. Validators ensure trades are valid, store them, and track portfolio returns. 

</details>

<br />
<br />

# Proprietary Trading Subnet

This repository contains the code for the Proprietary Trading Network (PTN) developed by Taoshi.

PTN receives signals from quant and deep learning machine learning trading systems to deliver the world's
most complete trading signals across a variety of asset classes.

# Features

🛠️&nbsp;Open Source Strategy Building Techniques (In Our Taoshi Community)<br>
🫰&nbsp;Signals From a <a href="https://github.com/taoshidev/proprietary-trading-network/blob/main/vali_objects/vali_config.py#L19"> Variety of Asset Classes</a> - Forex and Crypto<br>
📈&nbsp;<a href="https://taomarketcap.com/subnet/8?subpage=miners&metagraph_type=miners">Millions of $ Payouts</a> to Top Traders<br>
💪&nbsp;Innovative Trader Performance Metrics that Identify the Best Traders<br>
🔎&nbsp;<a href="https://dashboard.taoshi.io/">Trading + Metrics Visualization Dashboard</a>

## How does it work?

PTN is the most challenging & competitive network in the world. Our miners need to provide futures based signals (long/short)
that are highly efficient and effective across various markets to compete (forex, crypto). The top miners are
those that provide the most returns, while never exceeding certain drawdown limits.

### Rules

1. Miners can submit LONG, SHORT, or FLAT signal for Forex and Crypto trade pairs into the network during market hours. <a href="https://github.com/taoshidev/proprietary-trading-network/blob/main/vali_objects/vali_config.py#L173">Currently supported trade pairs</a>
2. Miners are eliminated if they are detected as plagiarising other miners, or if they exceed 10% max drawdown (more info in  the "Eliminations" section).
3. There is a fee for leaving positions open "carry fee". The fee is equal to 10.95/3% per year for a 1x leverage position (crypto/forex) <a href="https://docs.taoshi.io/tips/p4/">More info</a>
4. There is a slippage assessed per order. The slippage cost is is greater for orders with higher leverages, and in assets with lower liquidity.
5. Based on portfolio metrics such as omega score and total portfolio return, weights/incentive get set to reward the best miners <a href="https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/miner.md">More info</a>

With this system only the world's best traders & deep learning / quant based trading systems can compete.


# Eliminations

In the Proprietary Trading Network, Eliminations occur for miners that commit Plagiarism, or exceed 10% Max Drawdown.


### Plagiarism Eliminations

Miners who repeatedly copy another miner's trades will be eliminated. Our system analyzes the uniqueness of each submitted order. If an order is found to be a copy (plagiarized), it triggers the miner's elimination.

### Max Drawdown Elimination

Miners who exceed 10% max drawdowns will be eliminated. Our system continuously tracks each miner’s performance, measuring the maximum drop from peak portfolio value. If a miner’s drawdown exceeds the allowed threshold, they will be eliminated to maintain risk control.

### Post-Elimination

After elimination, miners are not immediately deregistered from the network. They will undergo a waiting period, determined by registration timelines and the network's immunity policy, before official deregistration. Upon official deregistration, the miner forfeits registration fees paid.



# Get Started

### Mainnet Trade Dashboard
Take a look at the top traders on PTN <a href="https://dashboard.taoshi.io/">Dashboard</a>

### Auto Trade with PTN data 
https://x.com/glitchfinancial

### Subscribe to Realtime Trade Data from PTN
https://request.taoshi.io/login 

### Theta Token
https://www.taoshi.io/theta

### Validator Installation

Please see our [Validator Installation](https://github.com/taoshidev//proprietary-trading-network/blob/main/docs/validator.md) guide.

### Miner Installation

Please see our [Miner Installation](https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/miner.md) guide.

# Building a strategy

We recommend joining our community hub via Discord to get assistance in building a trading strategy. We have partnerships with both Glassnode and LunarCrush who provide valuable data to be able to create an effective strategy. Analysis and information
on how to build a deep learning ML based strategy will continue to be discussed in an open manner by team Taoshi to help
guide miners to compete.

# Contributing

For instructions on how to contribute to Taoshi, see CONTRIBUTING.md and Taoshi's code of conduct.

# License

Bittensor's source code in this repository is licensed under the MIT License.
Taoshi Inc's source code in this repository is licensed under the MIT License.
