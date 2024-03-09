import json
import os
import traceback
import uuid

from flask import Flask, request, jsonify

import waitress

from miner_config import MinerConfig
from vali_config import TradePair
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.vali_dataclasses.signal import Signal

"""
for production: 

1. brew install nginx
2. update your receive_signals.conf file in mining/nginx_conf
3. create symbolic link for nginx conf sudo ln -s /path/to/receive_signals.conf /etc/nginx/conf.d/myapp.conf
4. brew services start nginx
5. brew services stop nginx
"""

app = Flask(__name__)

secrets_json_path = "miner_secrets.json"
# Define your API key
if os.path.exists(secrets_json_path):
    with open(secrets_json_path, "r") as file:
        data = file.read()
    API_KEY = json.loads(data)["api_key"]
else:
    raise Exception(f"{secrets_json_path} not found", 404)


# Endpoint to handle JSON POST requests
@app.route("/api/receive-signal", methods=["POST"])
def handle_data():
    # Check if 'Authorization' header is provided
    data = request.json

    if "api_key" in data:
        token = data["api_key"]
    else:
        return jsonify({"error": "Missing or invalid Authorization header"}), 401

    # Validate the API key
    if token != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401

    # Check if request is JSON
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        # ensure to fits rules for a Signal
        signal = Signal(
            trade_pair=TradePair.get_trade_pair(data["trade_pair"]),
            leverage=data["leverage"],
            order_type=OrderType.get_order_type(data["order_type"]),
        )
        # make miner received signals dir if doesnt exist
        ValiBkpUtils.make_dir(MinerConfig.get_miner_received_signals_dir())
        # store miner signal
        signal_file_uuid = str(uuid.uuid4())
        ValiBkpUtils.write_file(
            MinerConfig.get_miner_received_signals_dir() + signal_file_uuid, str(signal)
        )
    except ValueError:
        print(traceback.format_exc())
        return jsonify({"error": "improperly formatted signal received"}), 400
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "error storing signal on miner"}), 400


    return (
        jsonify({"message": "Signal {} received successfully".format(str(signal))}),
        200,
    )


if __name__ == "__main__":
    waitress.serve(app, host="0.0.0.0", port=80)
