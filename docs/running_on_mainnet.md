# Running on the Main Network
This tutorial shows how to use the bittensor `btcli` to create a subnetwork and connect 
your mechanism to it. It is highly recommended that you run `running_on_testnet` first 
before attempting to register on mainnet. Mechanisms running on the main are open to anyone, 
emit real TAO, and incurs a `lock_cost` in TAO to create, and you should be careful not 
to expose your private keys, to only use your testnet wallet, not use the same 
passwords as your mainnet wallet, and make sure your mechanism is resistant to abuse. 


## Steps

1. Clone and Install Prop Net Source Code
This clones and installs the template if you dont already have it (if you do, skip this step)
```bash
cd .. # back out of the subtensor repo
git clone git@github.com:taoshidev/proprietary-trading-network.git # Clone the prop net subnet repo
cd proprietary-trading-network # Enter the prop net repo
python -m pip install -e . # Install the bittensor-subnet-template package
```

2. (Optional) Register your validator and miner keys to the networks.

> Note: While not enforced, it is recommended for owners to run a validator and miner on the network to display proper use to the community

This registers your validator and miner keys to the network giving them slots on the network.
```bash
# Register your miner key to the network.
btcli subnet register --wallet.name miner --wallet.hotkey default  
>> Enter netuid [1] (8): # Enter netuid 1 to specify the network you just created.
>> Continue Registration?
  hotkey:     ...
  coldkey:    ...
  network:    finney [y/n]: # Select yes (y)
>> ⠦ 📡 Submitting POW...
>> ✅ Registered

# Register your validator key to the network.
btcli subnet register --wallet.name validator --wallet.hotkey default 
>> Enter netuid [1] (8): # Enter netuid 1 to specify the network you just created.
>> Continue Registration?
  hotkey:     ...
  coldkey:    ...
  network:    finney [y/n]: # Select yes (y)
>> ⠦ 📡 Submitting POW...
>> ✅ Registered
```

3. Check that your keys have been registered.
This returns information about your registered keys.
```bash
# Check that your validator key has been registered.
btcli wallet overview --wallet.name validator 
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  0      True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0         

# Check that your miner has been registered.
btcli wallet overview --wallet.name miner 
Subnet: 1                                                                                                                                                                
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58                    
miner    default  1      True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                14  none  5GTFrsEQfvTsh3WjiEVFeKzFTc2xcf…
1        1        2            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000                                                         
                                                                          Wallet balance: τ0.0   
```

4. Run the miner and validator directly with the netuid and chain_endpoint arguments.

**NOTE - we strongly recommend using a local subtensor to run on mainnet for reliability. As a validator, we highly
recommend using our run.sh script. Please read the README in order to see how to run the run.sh script.**

```bash
# Run the miner with the netuid and chain_endpoint arguments.
python neurons/miner.py --netuid 8  --wallet.name miner --wallet.hotkey default --logging.debug
>> 2023-08-08 16:58:11.223 |       INFO       | Running miner for subnet: 1 on network: wss://entrypoint-finney.opentensor.ai:443 with config: ...

# Run the validator with the netuid and chain_endpoint arguments.
python neurons/validator.py --netuid 8  --wallet.name validator --wallet.hotkey default --logging.debug
>> 2023-08-08 16:58:11.223 |       INFO       | Running validator for subnet: 1 on network: wss://entrypoint-finney.opentensor.ai:443 with config: ...
```

5. Stopping Your Nodes:
If you want to stop your nodes, you can do so by pressing CTRL + C in the terminal where the nodes are running.
