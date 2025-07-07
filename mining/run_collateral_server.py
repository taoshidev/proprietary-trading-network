import asyncio
import json
import os
import traceback
from flask import Flask, request, jsonify
import waitress
import bittensor as bt
from bittensor_wallet import Wallet

from miner_config import MinerConfig
from miner_objects.miner_contract_manager import MinerContractManager
from vali_objects.vali_config import ValiConfig

app = Flask(__name__)

# Load API key from existing miner_secrets.json
secrets_json_path = ValiConfig.BASE_DIR + "/mining/miner_secrets.json"
if os.path.exists(secrets_json_path):
    with open(secrets_json_path, "r") as file:
        data = file.read()
    API_KEY = json.loads(data)["api_key"]
else:
    raise Exception(f"{secrets_json_path} not found", 404)

# Initialize wallet and contract manager (will be set when server starts)
wallet = None
contract_manager = None
config = None

def validate_api_key(request):
    """Validate API key from request headers or JSON body"""
    # Try header first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        return token == API_KEY
    
    # Try JSON body
    if request.is_json and request.json and "api_key" in request.json:
        return request.json["api_key"] == API_KEY
    
    return False

def run_async(coro):
    """Helper to run async functions in Flask routes"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()
        return result
    except Exception as e:
        bt.logging.error(f"Error running async operation: {e}")
        raise

@app.route("/api/collateral/balance", methods=["GET"])
def get_collateral_balance():
    """Get current collateral balance in alpha tokens"""
    if not validate_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401
    
    if not contract_manager:
        return jsonify({"error": "Contract manager not initialized"}), 500
    
    try:
        alpha_balance = contract_manager.get_collateral_balance()
        
        return jsonify({
            "balance_alpha": alpha_balance,
            "success": True
        })
    except Exception as e:
        bt.logging.error(f"Error getting collateral balance: {e}")
        bt.logging.error(traceback.format_exc())
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/collateral/deposit", methods=["POST"])
def deposit_collateral():
    """Initiate collateral deposit with alpha tokens"""
    if not validate_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401
    
    if not contract_manager:
        return jsonify({"error": "Contract manager not initialized"}), 500
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    
    # Get amount (required) - now in alpha tokens
    if "amount" not in data:
        return jsonify({"error": "Missing required field: amount (in alpha tokens)"}), 400
    
    try:
        alpha_amount = float(data["amount"])
        if alpha_amount <= 0:
            return jsonify({"error": "Amount must be greater than 0"}), 400
    except ValueError:
        return jsonify({"error": "Amount must be a valid number"}), 400
    
    # Get validator info (use default if not specified)
    network = config.subtensor.network if config and hasattr(config, 'subtensor') else "test"
    validator_hotkey = data.get("validator_hotkey", MinerConfig.get_primary_validator_hotkey(network))
    validator_vault = data.get("validator_vault", MinerConfig.get_primary_validator_vault(network))
    
    try:
        bt.logging.info(f"Processing deposit request: {alpha_amount} alpha tokens to validator {validator_hotkey}")
        
        result = run_async(
            contract_manager.send_deposit_request(
                validator_hotkey=validator_hotkey,
                theta_amount=alpha_amount,
                validator_vault_address=validator_vault
            )
        )
        
        # Add additional info to response
        network_info = contract_manager.get_network_info()
        result["alpha_token_price"] = network_info["alpha_token_price"]
        result["validator_hotkey"] = validator_hotkey
        result["validator_vault"] = validator_vault
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error processing deposit: {e}"
        bt.logging.error(error_msg)
        bt.logging.error(traceback.format_exc())
        return jsonify({"error": error_msg, "success": False}), 500

@app.route("/api/collateral/withdraw", methods=["POST"])
def withdraw_collateral():
    """Initiate collateral withdrawal with alpha tokens"""
    if not validate_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401
    
    if not contract_manager:
        return jsonify({"error": "Contract manager not initialized"}), 500
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    
    # Get amount (required) - now in alpha tokens
    if "amount" not in data:
        return jsonify({"error": "Missing required field: amount (in alpha tokens)"}), 400
    
    try:
        alpha_amount = float(data["amount"])
        if alpha_amount <= 0:
            return jsonify({"error": "Amount must be greater than 0"}), 400
    except ValueError:
        return jsonify({"error": "Amount must be a valid number"}), 400
    
    # Get validator info (use default if not specified)
    network = config.subtensor.network if config and hasattr(config, 'subtensor') else "test"
    validator_hotkey = data.get("validator_hotkey", MinerConfig.get_primary_validator_hotkey(network))
    
    try:
        bt.logging.info(f"Processing withdrawal request: {alpha_amount} alpha tokens from validator {validator_hotkey}")
        
        result = run_async(
            contract_manager.send_withdrawal_request(
                validator_hotkey=validator_hotkey,
                theta_amount=alpha_amount
            )
        )
        
        # Add additional info to response
        network_info = contract_manager.get_network_info()
        result["alpha_token_price"] = network_info["alpha_token_price"]
        result["validator_hotkey"] = validator_hotkey
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error processing withdrawal: {e}"
        bt.logging.error(error_msg)
        bt.logging.error(traceback.format_exc())
        return jsonify({"error": error_msg, "success": False}), 500

def initialize_contract_manager(miner_config):
    """Initialize the contract manager with miner configuration"""
    global wallet, contract_manager, config
    
    try:
        config = miner_config
        wallet = Wallet(config=config)
        
        # Create dendrite for validator communication
        dendrite = bt.dendrite(wallet=wallet)
        
        contract_manager = MinerContractManager(
            wallet=wallet,
            config=config,
            dendrite=dendrite
        )
        
        bt.logging.info("Contract manager initialized successfully")
        return True
        
    except Exception as e:
        bt.logging.error(f"Failed to initialize contract manager: {e}")
        bt.logging.error(traceback.format_exc())
        return False
