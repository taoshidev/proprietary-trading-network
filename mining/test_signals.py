import sys

import requests
import json
import time

from vali_objects.enums.order_type_enum import OrderType
from vali_config import TradePair
import mining_utils 
import pandas as pd
from vali_objects.enums.order_type_enum import OrderType
from vali_config import TradePair, TradePairCategory, ValiConfig
from get_data import fetch_binance_data
from datetime import datetime 
import pickle
import os
from signals import process_data_for_predictions,LONG_ENTRY
import bittensor as bt


secrets_json_path = ValiConfig.BASE_DIR + "/mining/miner_secrets.json"
# Define your API key
if os.path.exists(secrets_json_path):
    with open(secrets_json_path, "r") as file:
        data = file.read()
    API_KEY = json.loads(data)["api_key"]
else:
    raise Exception(f"{secrets_json_path} not found", 404)


    

str_to_ordertype= { 
             'LONG' : OrderType.LONG, 
             'SHORT': OrderType.SHORT ,
             'FLAT' : OrderType.FLAT      
                   }
     
str_to_tradepair= { 
             'btcusd' : TradePair.BTCUSD, 
                   }


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TradePair) or isinstance(obj, OrderType):
            return obj.__json__()  # Use the to_dict method to serialize TradePair

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    # Set the default URL endpoint
    default_base_url = 'http://127.0.0.1:80'

    # Check if the URL argument is provided
    if len(sys.argv) == 2:
        # Extract the URL from the command line argument
        base_url = sys.argv[1]
    else:
        # Use the default URL if no argument is provided
        base_url = default_base_url

    print("base URL endpoint:", base_url)

    url = f'{base_url}/api/receive-signal'
    
    last = None
    
    order = 'LONG'


    for i in range(0,8):  
        
        print(i)
        
    
        # feed into model to predict 
        if i in [0,3,6,9,12]: 
            order ='LONG'
        else : 
            order = 'FLAT'

        
            print('Order Triggered.')
            bt.logging.info(f"Order Triggered.")

                
            order_type = str_to_ordertype[order]
            
            trade_pair = str_to_tradepair['btcusd']       
            
            
            # Define the JSON data to be sent in the request
                    
            data = {
                'trade_pair':trade_pair ,
                'order_type': order_type,
                'leverage': 1.0,
                'api_key':API_KEY,
                } 
            
            print(f"order type: {order_type}")
            
    
            # Convert the Python dictionary to JSON format
            json_data = json.dumps(data, cls=CustomEncoder)
            print(json_data)
            # Set the headers to specify that the content is in JSON format
            headers = {
                'Content-Type': 'application/json',
            }

            # Make the POST request with JSON data
            response = requests.post(url, data=json_data, headers=headers)
            print('Order Posted')
            bt.logging.info(f"Order Posted")

        

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                print("POST request was successful.")
                print("Response:")
                print(response.json())  # Print the response data
            else:
                print(response.__dict__)
                print("POST request failed with status code:", response.status_code)
            
            time.sleep(10)
            
