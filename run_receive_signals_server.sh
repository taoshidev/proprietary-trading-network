#!/bin/bash

while true; do
    echo "Starting Python script..."
    source venv/bin/activate
    python -m pip install -e .
    nohup venv/bin/python mining/run_receive_signals_server.py &
    PID=$!
    echo "Python script started with PID: $PID"

    # Wait for the process to finish
    wait $PID

    # Check if the process is still running
    if ps -p $PID > /dev/null; then
        echo "Python script is still running, not restarting."
    else
        echo "Python script stopped, restarting in 5 seconds..."
        sleep 5
    fi
done