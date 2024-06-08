module.exports = {
    apps: [
      {
        name: 'server',
        script: 'python3',
        args: './mining/run_receive_signals_server.py'
      },
      {
        name: 'signals',
        script: 'python3',
        args: './mining/generate_signals.py'
      },
    ],
  };