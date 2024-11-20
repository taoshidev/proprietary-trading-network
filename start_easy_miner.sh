#!/bin/bash

# Define the required Node.js version
REQUIRED_NODE_VERSION="20.9.0"

PROJECT_ROOT=$(pwd)

# Check for Python executable
PYTHON_EXEC=$(command -v python3 || command -v python)

#!/bin/bash

# Define the required Node.js version
REQUIRED_NODE_VERSION="20.9.0"

# Check if nvm is installed
if ! command -v nvm &> /dev/null; then
    echo "NVM is not installed. Installing NVM..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
    # Load NVM
    export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

# Check if the required Node.js version is installed
if ! nvm ls "$REQUIRED_NODE_VERSION" &> /dev/null; then
    echo "Node.js $REQUIRED_NODE_VERSION is not installed. Installing..."
    nvm install "$REQUIRED_NODE_VERSION"
fi

# Use the required Node.js version
echo "Switching to Node.js $REQUIRED_NODE_VERSION..."
nvm use "$REQUIRED_NODE_VERSION"

# Confirm Node.js version
NODE_VERSION=$(node -v)
echo "Using Node.js $NODE_VERSION"

# Install the npm package
echo "Installing npm package..."
npm install "$1" --engine-strict

if [ $? -eq 0 ]; then
    echo "Package installed successfully!"
else
    echo "Error installing package. Please check the output above for details."
    exit 1
fi

if [ -z "$PYTHON_EXEC" ]; then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

echo "Installing PTN..."

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_EXEC -m venv venv
fi

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    . venv/bin/activate
else
    echo "Error: Unable to find venv/bin/activate. Exiting..."
    exit 1
fi

# Install dependencies
pip install -r requirements.txt
python -m pip install -e .

echo "Starting Signals Server..."
python mining/run_receive_signals_server.py &  # Run Python script in the background
PID=$!  # Capture the PID of the Python process
echo "Signals Server started with PID: $PID"

# Start Order Watcher...
echo "Starting Order Watcher..."
cd $PROJECT_ROOT/miner_objects/order_watcher
npm install
npm run dev &  # Run Order Watcher in the background

sleep 5

# Start Easy Miner...
echo "Starting Easy Miner..."
cd $PROJECT_ROOT/miner_objects/easy_miner
npm install
npm run dev &  # Run Easy Miner in the background

# Wait for background processes to finish
wait $PID