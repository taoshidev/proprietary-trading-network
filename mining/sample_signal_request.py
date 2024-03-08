import requests
import json

if __name__ == "__main__":

    # Define the URL endpoint
    url = 'http://127.0.0.1:5000/api/receive-signal'

    # Define the JSON data to be sent in the request
    data = {
        'trade_pair': 'BTCUSD',
        'order_type': 'LONG',
        'leverage': .5,
        'api_key': 'xxxx'
    }

    # Convert the Python dictionary to JSON format
    json_data = json.dumps(data)

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