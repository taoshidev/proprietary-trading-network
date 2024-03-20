# Running on the Testing Network
This tutorial shows how to use the bittensor testnetwork to create a subnetwork and connect your mechanism to it. 
Mechanisms running on the testnet are open to anyone, and although they do not emit real TAO, they cost test TAO 
to create and you should be careful not to expose your private keys, to only use your testnet wallet, not use the 
same passwords as your mainnet wallet, and make sure your mechanism is resistant to abuse. 

## Steps

1. Clone and Install Prop Net Source Code
This clones and installs the template if you dont already have it (if you do, skip this step)
```bash
cd .. # back out of the subtensor repo
git clone git@github.com:taoshidev/proprietary-trading-network.git # Clone the prop net subnet repo
cd proprietary-trading-network # Enter the prop net subnet repo
python -m pip install -e . # Install the bittensor-subnet-template package
```

2. Create wallets for your subnet owner, for your validator and for your miner.
This creates local coldkey and hotkey pairs for your 3 identities. The owner will create and control the subnet and 
validator/miner scripts and be registered to the subnetwork created by the owner.
```bash
# Create a coldkey for your owner wallet.
btcli wallet new_coldkey --wallet.name owner

# Create a coldkey and hotkey for your miner wallet.
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# Create a coldkey and hotkey for your validator wallet.
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

3. (Optional) Getting testnet tokens
If you don't have enough tokens to register with testnet, please go to the bittensor discord and request testnet tao
from the general discussion.

4. Register your validator and miner keys to the TPN testnet.
This registers your validator and miner keys to the network.
```bash
# Register your miner key to the network.
btcli subnet register --wallet.name miner --wallet.hotkey default  --subtensor.network test --netuid 116
>> Continue Registration?
  hotkey:     ...
  coldkey:    ...
  network:    finney [y/n]: # Select yes (y)
>> ⠦ 📡 Submitting POW...
>> ✅ Registered

# Register your validator key to the network.
btcli subnet register --wallet.name validator --wallet.hotkey default --subtensor.network test --netuid 116
>> Continue Registration?
  hotkey:     ...
  coldkey:    ...
  network:    finney [y/n]: # Select yes (y)
>> ⠦ 📡 Submitting POW...
>> ✅ Registered
```

5. Check that your keys have been registered.
This returns information about your registered keys.
```bash
# Check that your validator key has been registered.
btcli wallet overview --wallet.name validator --subtensor.network test
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  0      True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0         

# Check that your miner has been registered.
btcli wallet overview --wallet.name miner --subtensor.network test
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  1      True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0   
```

6. Run the miner and validator directly with the netuid and chain_endpoint arguments.
```bash
# Run the miner with the netuid and chain_endpoint arguments.
python neurons/miner.py --netuid 116 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug
>> 2023-08-08 16:58:11.223 |        INFO       | Connected to test network and wss://test.finney.opentensor.ai:443/...

# Run the validator with the netuid and chain_endpoint arguments.
python neurons/validator.py --netuid 116 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug
>> 2023-08-08 16:58:11.223 |       INFO       | Running validator for subnet: 116 on network: wss://entrypoint-finney.opentensor.ai:443 with config:...
```

You can also use two miners/validators when testing.

Example running a second miner:

```bash
$ python neurons/miner.py --netuid 116 --subtensor.network test --wallet.name miner2 --wallet.hotkey default --logging.debug --axon.port 8095
```

7. Stopping Your Nodes:
If you want to stop your nodes, you can do so by pressing CTRL + C in the terminal where the nodes are running.

