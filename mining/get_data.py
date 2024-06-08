import requests
import pandas as pd
from datetime import datetime

def fetch_binance_data(symbol="BTCUSDT", interval='5m', start=None, end=None, limit=1000):
    # Set default start and end times if none are provided
    if start is None:
        # Set start to two weeks before the current time
        start = str(int(datetime.now().timestamp() * 1000) - 60000 * 60 * 24 * 7 * 2)
    if end is None:
        # Set end to the current time
        end = str(int(datetime.now().timestamp() * 1000))

    # Construct the URL with the provided parameters
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start}&endTime={end}&limit={limit}'

    # Fetch the data
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Define column names
        columns = ['ds', 'open', 'high', 'low', 'close', 'volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']
        
        # Load the data into a pandas DataFrame
        data = pd.DataFrame(response.json(), columns=columns)
        
        # Convert 'Open Time' and 'Close Time' to datetime format
        data['ds'] = pd.to_datetime(data['ds'], unit='ms')
        #data['Close Time'] = pd.to_datetime(data['Close Time'], unit='ms')
        data['unique_id'] = 'BTCUSD'
        #data['y'] = data['close']
        
        return data
    else:
        print("Failed to fetch data:", response.status_code)
        return None
