import sys

import requests
import json
import time

from vali_objects.enums.order_type_enum import OrderType
from vali_config import TradePair
import mining_utils 
import pandas as pd
from secrets import apikey as key 
from vali_objects.enums.order_type_enum import OrderType
from vali_config import TradePair, TradePairCategory
from datetime import datetime 
import pickle
import os

def round_time_to_nearest_five_minutes(dt):
    # Convert the datetime to seconds since epoch
    dt = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%S.%f')
    timestamp = dt.timestamp()
    # Number of seconds in 5 minutes
    round_to = 5 * 60
    # Perform the rounding
    rounded_timestamp = round(timestamp / round_to) * round_to 
    # Convert the timestamp back to a datetime object
    rounded_dt = datetime.fromtimestamp(rounded_timestamp)
    
    return rounded_dt

str_to_ordertype= { 
             'long' : OrderType.LONG, 
             'short': OrderType.SHORT ,
             'flat' : OrderType.FLAT      
                   }
     
str_to_tradepair= { 
             'btcusd' : TradePair.BTCUSD, 
                   }


class TradeHandler:
    def __init__(self, signal=None, last_update=None, pair=None, current_position=None, trade_opened=None, position_open=False, filename='trade_handler_state.pkl'):
        self.filename = filename
        if os.path.exists(self.filename):
            try:
                loaded_instance = self.load_from_file(self.filename)
                self.__dict__.update(loaded_instance.__dict__)
                print(f"State loaded from {self.filename}")
            except Exception as e:
                print(f"Error loading state: {e}")
                self.initialize_attributes(signal, last_update, pair, current_position, trade_opened, position_open)
        else:
            self.initialize_attributes(signal, last_update, pair, current_position, trade_opened, position_open)

    def initialize_attributes(self, signal, last_update, pair, current_position, trade_opened, position_open):
        self.pair = pair
        self.current_position = current_position
        self.trade_opened = trade_opened
        self.position_open = position_open
        self.last_update = last_update
        self.signal = signal

    def clear_trade(self):
        self.current_position = None
        self.trade_opened = None
        self.position_open = False
        self.last_update = None 
        self.signal = None
        self.save_to_file(self.filename)
        print('Trade cleared.')

    def check_position(self): 
        print(self.current_position)

    def set_position(self, new_position):
        if self.current_position == 'SHORT' and new_position == 'LONG':
            print('Position changed from short to long, closing current position.')
            self.clear_trade()
            self.current_position = 'FLAT'
            self.last_update = datetime.now().isoformat()

        elif self.current_position == 'LONG' and new_position == 'SHORT':
            print('Position changed from long to short, closing current position.')
            self.clear_trade()
            self.current_position = 'FLAT'
            self.last_update = datetime.now().isoformat()
  
        elif self.current_position in ['SHORT', 'LONG'] and new_position == 'FLAT':
            print('Trade closed.')
            self.clear_trade()
            self.current_position = 'FLAT'
            self.last_update = datetime.now().isoformat()


        else:
            if not self.position_open and new_position in ['LONG', 'SHORT']:
                self.trade_opened = datetime.now().isoformat()
                self.last_update = self.trade_opened 
                print(f'Trade opened at: {self.trade_opened}')
                self.position_open = True
                self.current_position = new_position
            else:  
                self.last_update = datetime.now().isoformat()
        
        # Save state to file after updating the position
        self.save_to_file(self.filename)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f'State saved to {filename}')

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print(f'State loaded from {filename}')
        return obj

     
    

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
    
    btc =  TradeHandler(pair='btcusd')
    
    while True:  
            
        
        # load live data
        input = mining_utils.fetch_binance_data()
        
        if (btc.last_update is None) or (round_time_to_nearest_five_minutes(btc.last_update) < pd.to_datetime(input['ds'].tail(1).values[0])):            
            # feed into model to predict 
            
            model = mining_utils.load_model()
            preds = mining_utils.multi_predict(model,input,2)
            modelname = str(model.models[0])
            output = mining_utils.gen_signals_from_predictions(predictions= preds, hist = input ,modelname=modelname ) 
            signals = mining_utils.assess_signals(output)
            order= mining_utils.map_signals(signals)
            
            old_position = btc.position_open
                   
            btc.set_position(order)
            
            new_position = btc.position_open 
            

            
            if sum([old_position,new_position]) == 1 :  
                
                    
                order_type = str_to_ordertype(btc.current_position)
                
                trade_pair = str_to_tradepair(btc.pair)       
                
                
                # Define the JSON data to be sent in the request
                        
                data = {
                    'trade_pair':trade_pair ,
                    'order_type': order_type,
                    'leverage': 1.0,
                    'api_key': key ,
                    } 
                
        
                # Convert the Python dictionary to JSON format
                json_data = json.dumps(data, cls=CustomEncoder)
                print(json_data)
                # Set the headers to specify that the content is in JSON format
                headers = {
                    'Content-Type': 'application/json',
                }

                # Make the POST request with JSON data
                response = requests.post(url, data=json_data, headers=headers)
                
            

                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    print("POST request was successful.")
                    print("Response:")
                    print(response.json())  # Print the response data
                else:
                    print(response.__dict__)
                    print("POST request failed with status code:", response.status_code)
                
                time.sleep(60)
                
            else: 
                print('No Change In Position')
                time.sleep(60)