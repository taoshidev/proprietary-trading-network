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
import duckdb
import numpy as np

model = mining_utils.load_model()
TP = 0.05 
secrets_json_path = ValiConfig.BASE_DIR + "/mining/miner_secrets.json"
# Define your API key
if os.path.exists(secrets_json_path):
    with open(secrets_json_path, "r") as file:
        data = file.read()
    API_KEY = json.loads(data)["api_key"]
else:
    raise Exception(f"{secrets_json_path} not found", 404)

def round_time_to_nearest_five_minutes(dt):
    
    try: 
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
    except: 
        return None

str_to_ordertype= { 
             'LONG' : OrderType.LONG, 
             'SHORT': OrderType.SHORT ,
             'FLAT' : OrderType.FLAT      
                   }
     
str_to_tradepair= { 
             'btcusd' : TradePair.BTCUSD, 
                   }


class TradeHandler:
    def __init__(self,price=None, signal=None, last_update=None, pair=None, current_position=None, trade_opened=None, position_open=False, filename='trade_handler_state.pkl'):
        self.filename = filename
        if os.path.exists(self.filename):
            try:
                loaded_instance = self.load_from_file(self.filename)
                self.__dict__.update(loaded_instance.__dict__)
                print(f"State loaded from {self.filename}")
            except Exception as e:
                print(f"Error loading state: {e}")
                self.initialize_attributes(signal, last_update, pair, current_position, trade_opened, position_open)
                self.init_table()
        else:
            self.initialize_attributes(signal, last_update, pair, current_position, trade_opened, position_open)
            self.init_table()
            
    def initialize_attributes(self, signal: str, last_update: datetime, pair: str, current_position: str, trade_opened: datetime, position_open:str):
        self.pair = pair
        self.current_position = current_position
        self.trade_opened = trade_opened
        self.position_open = position_open
        self.last_update = last_update
        self.signal = signal
        self.price = price 

    def clear_trade(self):
        self.current_position = None
        self.trade_opened = None
        self.position_open = False
        self.last_update = None 
        self.signal = None
        self.price = None
        self.save_to_file(self.filename)
        print('Trade cleared.')

    def check_position(self): 
        print(self.current_position)

    def set_position(self,price:float, new_position: str):
        if self.current_position == 'SHORT' and new_position == 'LONG':
            self.last_update = datetime.now().isoformat()
            self.close_trade_to_duckdb(close_price=self.price, trade_closed= self.last_update,signal=self.signal, pair=self.pair)
            print('Position changed from short to long, closing current position.')
            self.clear_trade(close_price = price)
            self.current_position = 'FLAT'
            self.last_update = datetime.now().isoformat()

        elif self.current_position == 'LONG' and new_position == 'SHORT':
            self.last_update = datetime.now().isoformat()
            self.close_trade_to_duckdb(close_price=self.price, trade_closed= self.last_update,signal=self.signal, pair=self.pair)
            print('Position changed from long to short, closing current position.')
            self.clear_trade(close_price = price)
            self.current_position = 'FLAT'
            self.last_update = datetime.now().isoformat()
  
        elif self.current_position in ['SHORT', 'LONG'] and new_position == 'FLAT':
            self.last_update = datetime.now().isoformat()
            self.close_trade_to_duckdb(close_price=self.price, trade_closed= self.last_update,signal=self.signal, pair=self.pair)
            print('Trade closed.')
            self.clear_trade(close_price = price)
            self.current_position = 'FLAT'
            self.last_update = datetime.now().isoformat()
            
        else:
            if not self.position_open and new_position in ['LONG', 'SHORT']:
                self.trade_opened = datetime.now().isoformat()
                self.last_update = self.trade_opened 
                self.price = price
                print(f'Trade opened at: {self.trade_opened}')
                self.position_open = True
                self.current_position = new_position
                self.open_trade_to_duckdb()
            else:  
                self.last_update = datetime.now().isoformat()
        
        # Save state to file after updating the position
        self.save_to_file(self.filename)

    def save_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f'State saved to {filename}')

    @classmethod
    def load_from_file(cls, filename: str):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        print(f'State loaded from {filename}')
        return obj

    def init_table(self,db_filename: str = 'trades.duckdb', table_name: str = 'trades') -> None:
        conn = duckdb.connect(db_filename)
        try:
            # Create the table if it does not exist
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    signal VARCHAR,
                    pair VARCHAR,
                    trade_opened TIMESTAMP,
                    open_price FLOAT,
                    trade_closed TIMESTAMP,
                    close_price FLOAT
                )
            """)
            print(f"Table '{table_name}' initialized in database '{db_filename}'")
        finally:
            conn.close()

    def open_trade_to_duckdb(self, db_filename: str = 'trades.duckdb', table_name: str = 'trades') -> None:
            conn = duckdb.connect(db_filename)
            try:
                # Create the table if it does not exist
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        signal VARCHAR,
                        pair VARCHAR,
                        trade_opened TIMESTAMP,
                        open_price FLOAT,
                        trade_closed TIMESTAMP,
                        close_price FLOAT
                    )
                """)

                # Check if the last trade is closed
                result = conn.execute(f"""
                    SELECT trade_closed
                    FROM {table_name}
                    WHERE signal = ? AND pair = ?
                    ORDER BY trade_opened DESC
                    LIMIT 1
                """, (self.signal, self.pair)).fetchone()
                
                if result is not None and result[0] is None:
                    print("Warning: The last trade is not closed yet.")

                # Insert the current trade data into the table
                conn.execute(f"""
                    INSERT INTO {table_name} (signal, pair, trade_opened, open_price)
                    VALUES (?, ?, ?, ?)
                """, (self.signal, self.pair, self.trade_opened, self.price))
                
                print(f"Trade opened and saved to DuckDB table '{table_name}' in database '{db_filename}'")
            finally:
                conn.close()

    def close_trade_to_duckdb(signal: str, pair: str, close_price: float, trade_closed: datetime, db_filename: str = 'trades.duckdb', table_name: str = 'trades') -> None:
            conn = duckdb.connect(db_filename)
            try:
                # Check if there is an open trade that has not been closed yet
                result = conn.execute(f"""
                    SELECT close_price, trade_closed
                    FROM {table_name}
                    WHERE signal = ? AND pair = ? AND trade_closed IS NULL
                    ORDER BY trade_opened DESC
                    LIMIT 1
                """, (signal, pair)).fetchone()
                
                if result is None:
                    raise Exception("No open trade found to close.")
                
                if result[0] is not None or result[1] is not None:
                    raise Exception("The trade has already been closed.")
                
                # Update the close price and trade closed time for the last open trade
                conn.execute(f"""
                    UPDATE {table_name}
                    SET close_price = ?, trade_closed = ?
                    WHERE signal = ? AND pair = ? AND trade_closed IS NULL
                    ORDER BY trade_opened DESC
                    LIMIT 1
                """, (close_price, trade_closed, signal, pair))
                
                print(f"Trade closed and updated in DuckDB table '{table_name}' in database '{db_filename}'")
            finally:
                conn.close()

    @staticmethod
    def check_last_trade(db_filename: str = 'trades.duckdb', table_name: str = 'trades') -> None:
        conn = duckdb.connect(db_filename)
        try:
            # Retrieve the last row of the table
            result = conn.execute(f"""
                SELECT *
                FROM {table_name}
                ORDER BY trade_opened DESC
                LIMIT 1
            """).df()

            if result:
                conn.close()    
                return result
               #print("Last row in the table:")
               # print(result)
            else:
                print("The table is empty.")
                conn.close()  
                return False

        except: 
                print("The table is empty.")
                conn.close()  
                return False

            
    
            

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
    bt.logging.info(f"Initialised trade handler.")
    bt.logging.info(f"Beginning loop.")

    while True:  
            

        # load live data
        input = fetch_binance_data()
        bt.logging.info(f"Latest candle: {input['ds'].tail(1).values[0]}")
        
        bt.logging.info(f"Last Trade: {btc.check_last_trade()}")


        input = process_data_for_predictions(input)
        
        if (btc.last_update is None) or (round_time_to_nearest_five_minutes(btc.last_update) < pd.to_datetime(input['ds'].tail(1).values[0])):            
            # feed into model to predict 
            
            price = input['close'].tail(1).values[0]
            lasttrade = btc.check_last_trade()
            
            if  isinstance(lasttrade, pd.DataFrame): 
                
                 if lasttrade['trade_closed'].tail(1).isnull():

                    current_pnl = None
                    exit_long = False 
                    
                        
                    current_pnl = input['close'].tail(1).values[0] / lasttrade['open_price'].tail(1) - 1 
                        
                    if current_pnl > TP :  
                        exit_long = True
                        
                    
                    if (current_pnl is not None)  or (exit_long is True) :
                        
                        order = 'FLAT'
                
            else: 
                preds = mining_utils.multi_predict(model,input,2)
                modelname = str(model.models[0])
                output = mining_utils.gen_signals_from_predictions(predictions= preds, hist = input ,modelname=modelname ) 
            #  signals = mining_utils.assess_signals(output)
                order= mining_utils.map_signals(output)
                
                          
            if order != 'PASS' : 
            
                old_position = btc.position_open
                    
                btc.set_position(order,price=price )
                
                new_position = btc.position_open 
                
                
                if sum([old_position,new_position]) == 1 :  
                    
                    print('Order Triggered.')
                    bt.logging.info(f"Order Triggered.")

                        
                    order_type = str_to_ordertype[btc.current_position]
                    
                    trade_pair = str_to_tradepair[btc.pair]       
                    
                    
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
                    bt.logging.info(f"Order Posted")
                    bt.logging.info(f"Status: { response.status_code }")

                

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
                bt.logging.info(f"No Change In Position")

                time.sleep(60)
                
        time.sleep(60)