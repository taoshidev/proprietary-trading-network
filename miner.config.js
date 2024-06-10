

module.exports = {
    apps: [
      {
        name: 'miner',
        script: 'python3',
        args: './neurons/miner.py --subtensor.network test --netuid 116 --wallet.name miner --wallet.hotkey default --logging.debug'
      }
    ],
  };