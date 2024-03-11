import requests
import json
from vali_objects.enums.order_type_enum import OrderType
from vali_config import TradePair

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TradePair) or isinstance(obj, OrderType):
            return obj.__json__()  # Use the to_dict method to serialize TradePair

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":

    # Define the URL endpoint
    url = 'http://127.0.0.1:8080/api/receive-signal'

    # Define the JSON data to be sent in the request
    data = {
        'trade_pair': TradePair.BTCUSD,
        'order_type': OrderType.LONG,
        'leverage': .5,
        'api_key': 'xxxx'
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