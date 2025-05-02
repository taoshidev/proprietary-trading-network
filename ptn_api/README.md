# PTN API and Websocket Data

## Overview

This repository provides a comprehensive API server and client for the [Proprietary Trading Network (PTN) - Bittensor subnet 8](https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/validator.md). It features both REST and WebSocket endpoints that enable real-time access to trading data, positions, statistics, and other critical information.

The REST API server is designed for validators to efficiently provide PTN data to consumers, with support for different data freshness tiers, real-time updates, and secure authentication.

The Websocket server allows for real-time streaming of trading data, enabling clients to receive updates as they happen.

The Websocket client is built to interact with the Websocket server, allowing for real-time data streaming and processing. 

## Features

- **Dual API Support**:
  - REST API for standard HTTP requests
  - WebSocket API for real-time data streaming
- **Secure Authentication**:
  - Dynamic API key management with automatic refresh
  - Token-based authentication for all endpoints
- **Data Tiering**:
  - Support for different data freshness tiers (0%, 30%, 50%, 100%)
  - Customizable pricing models for data access
- **Performance Optimizations**:
  - GZIP compression for large payloads
  - Batch message processing for WebSocket communications
  - Efficient sequence tracking for message reliability
- **Fault Tolerance**:
  - Automatic reconnection with exponential backoff
  - Process isolation for stability

## Architecture

The system consists of three main components:

1. **API Manager**: Coordinates the REST and WebSocket services, handles process management, and maintains shared state.
2. **REST Server**: Provides HTTP endpoints for querying historical data and statistics.
3. **WebSocket Server**: Enables real-time data streaming with sequence tracking.

## Configuration

### API Keys Management

API keys are stored in a JSON file with the following format:

```json
    {
      "user_id": {
        "key": "the_api_key_string",
        "tier": <0,30,50,100>
      },
      ...
    }

    Where tier values represent access levels:
    - 0: Basic access (24-hour lagged data, no real-time access)
    - 30: Standard access (30% real-time data)
    - 50: Enhanced access (50% real-time data)
    - 100: Premium access (100% data freshness + WebSocket support)
```

By default, the system looks for this file at the relative path `ptn_api/api_keys.json`. The API keys are automatically refreshed from disk, allowing you to add or remove keys without restarting the server.


The [Request Network](https://request.taoshi.io/) is a Taoshi product which serves subnet data while handling security, rate limiting, data customization, and provides a polished customer-facing and validator setup UI. Running this repo's APIManager is a prerequisite to serving data on the Request Network

For end users who want to access PTN data, you will need a Request Network API key. If you have any issues or questions, please reach out to the Taoshi team on Discord.

### Command Line Options

```
standalone usage: python api_manager.py [-h] [--serve] [--api-host API_HOST] [--api-rest-port API_REST_PORT]
                     [--api-ws-port API_WS_PORT] [--netuid NETUID] ...

optional arguments:
  -h, --help                       Show this help message and exit
  --serve                          Start the API server for REST and WebSocket endpoints
  --api-host API_HOST              Host address for the API server (default: 127.0.0.1)
  --api-rest-port API_REST_PORT    Port for the REST API server (default: 48888)
  --api-ws-port API_WS_PORT        Port for the WebSocket server (default: 8765)
  
 Provide the same args when running as a part of your SN8 validator.
```

### Making Server Accessible

By default, the server binds to `127.0.0.1` which only allows local requests. To allow access from any IP address:

```bash
python main.py --serve --api-host 0.0.0.0
```

## REST API Endpoints

### All Miners Positions 

`GET /miner-positions`

This endpoint retrieves the positions of all miners, optionally filtered by a specified data freshness tier.

**Tier Parameter**:
- `tier` (optional): Specifies the data freshness tier ['0', '30', '50', '100']
- 0: 100% of positions show data with a 24-hour delay.
- 30: 30% of positions show real-time data, 70% show data with a 24-hour delay. (superset of tier 0)
- 50: 50% of positions show real-time data, 50% show data with a 24-hour delay. (superset of tier 30)
- 100: 100% of positions show real-time data. Equivalent to not providing a tier. (superset of tier 50)

**Response**:
A JSON file containing all miner positions at the specified tier.

e.x:

```json
{
    "5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ": {
        "all_time_returns": 1.046956054826957,
        "n_positions": 15,
        "percentage_profitable": 0.8666666666666667,
        "positions": [
            {
                "average_entry_price": 0.59559,
                "close_ms": 1714156813363,
                "current_return": 1.0002165919508386,
                "initial_entry_price": 0.59559,
                "is_closed_position": true,
                "miner_hotkey": "5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ",
                "net_leverage": 0.0,
                "open_ms": 1714139478990,
                "orders": [
                    {
                        "leverage": -0.1,
                        "order_type": "SHORT",
                        "order_uuid": "18ca3cdf-b785-4f88-90a9-d2c06e8653b1",
                        "price": 0.59559,
                        "price_sources": [],
                        "processed_ms": 1714139478990
                    },
                    {
                        "leverage": 0.0,
                        "order_type": "FLAT",
                        "order_uuid": "c902c428-fcfb-43ca-ab79-117c957dbbfa",
                        "price": 0.5943,
                        "price_sources": [],
                        "processed_ms": 1714156813363
                    }
                ],
                "position_type": "FLAT",
                "position_uuid": "1f3f427f-6cbe-497c-af11-2fbef2ca3c10",
                "return_at_close": 1.0002095904346948,
                "trade_pair": [
                    "NZDUSD",
                    "NZD/USD",
                    7e-05,
                    0.001,
                    500
                ]
            },
...
```

Hotkeys are mapped to a data dict. The data dict contains positions which contain orders.

**Explanation of Schema:**
* miner_hotkey: A unique identifier for a miner. This is the same as the Bittensor metagraph hotkey value.
* all_time_returns: The miner's total return on investment across all positions over all time.
* n_positions: The number of positions held by the miner.
* percentage_profitable: The proportion of the miner's positions that have been profitable.
* positions: A list of individual trading positions held by the miner.
* average_entry_price: The average price at which the miner entered the position.
* current_return: The current return on the position with no fees.
* return_at_close: The return on the position at the time it was closed with all fees applied.
* initial_entry_price: The price at which the position was first opened.
* is_closed_position: Indicates if the position is closed.
* net_leverage: The leverage currently used in the position. 0 if the position is closed.
* open_ms: The timestamp (in milliseconds) when the position was opened.
* close_ms: The timestamp (in milliseconds) when the position was closed. 0 if not closed.
* orders: A list of orders executed within the position.
    - leverage: The leverage applied to the order.
    - order_type: The type of order (e.g., SHORT, LONG, FLAT).
    - order_uuid: A unique identifier for the order. Must be unique across all miners
    - price: The price at which the order was executed (filled).
    - price_sources: Used for debugging. Info about the price sources used to determine the price. At the time of this writing the sources are Polygon and Tiingo.
    - src: 0 if the order was placed by the miner, 1 if it was an automatically generated FLAT order due to elimination, 2 for automatically generated FLAT order to due trade pair deprecation. 
    - bid: The bid price at the time of order execution.
    - ask: The ask price at the time of order execution.
* processed_ms: The timestamp (in milliseconds) when the order was processed.
* position_type: The current status of the position (e.g., FLAT, SHORT, LONG).
* position_uuid: A unique identifier for the position.
* trade_pair: Information about the trade pair (e.g., currency pair BTCUSD).

### Single Miner Positions

`GET /miner-positions/<minerid>`

Returns position data for a specific miner identified by their hotkey.

### Miner Hotkeys

`GET /miner-hotkeys`

Returns all the hotkeys as seen in the metagraph from the validator's perspective.

### All Miners Statistics 

`GET /statistics`

Returns statistics relevant for scoring miners, consumed by the Taoshi dashboard.

**Parameters**:
- `checkpoints` (optional): Include checkpoint data (default: "true")

### Single Miner Statistics 

`GET /statistics/<minerid>/`

Returns statistics for a specific miner.

### Eliminations

`GET /eliminations`

Returns information about which miners have been eliminated and why. Note: deregistered miners are not shown in this list.
More information can be found here: https://github.com/taoshidev/proprietary-trading-network/blob/main/docs/miner.md#miner

e.x:
```json
 "eliminations": [
    {
      "dd": 0.0,
      "elimination_initiated_time_ms": 1711184954891,
      "hotkey": "5Dk2u35LRYEi9SC5cWamtzRkdXJJDLES7gABuey6cJ6t1ajK",
      "reason": "LIQUIDATED"
    },
    {
      "elimination_initiated_time_ms": 1711204151504,
      "hotkey": "5G1iDH2gvdAyrpUD4QfZXATvGEtstRBiXWieRDeaDPRfPEcU",
      "reason": "PLAGIARISM"
    },
    {
```

### Validator Checkpoint 

`GET /validator-checkpoint`

Everything required for a validator to restore it's state when starting for the first time. This includes all miner positions as well as derived data such as perf ledgers, challenge period data, and eliminations.

Perf Ledger schema
```json
"perf_ledgers": {
    "5C5GANtAKokcPvJBGyLcFgY5fYuQaXC3MpVt75codZbLLZrZ": {
      "cps": [
        {
          "accum_ms": 21600000,
          "gain": 0.12586433994869853,
          "last_update_ms": 1714161050595,
          "loss": -0.12587360888356938,
          "n_updates": 17213,
          "open_ms": 21599768,
          "prev_portfolio_ret": 0.9999907311080851
        },
        {
          "accum_ms": 21600000,
          "gain": 0.017040557887505504,
          "last_update_ms": 1714182650595,
          "loss": -0.016984326534111933,
          "n_updates": 2219,
          "open_ms": 21599768,
          "prev_portfolio_ret": 1.0000469635212756
        },
        {
...
```
Perf ledgers are built based off realtime price data and are consumed in the scoring logic. More info in the PTN repo.


## Compression Support

The API server supports automatic gzip compression for REST responses, which can significantly reduce payload sizes and improve performance. Compression is particularly beneficial for large responses like miner positions and statistics.

### Compression Benefits

- **Reduced Bandwidth**: Responses can be compressed by 70-90%, especially for JSON data
- **Faster Transmission**: Smaller payloads lead to quicker response times
- **Lower Costs**: Reduced data transfer costs with cloud hosting providers

### Using Compression with curl

To request compressed responses with curl, use the `--compressed` flag:

```bash
# Get positions with compression
curl --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key"

# Get positions with a specific tier (compressed)
curl --compressed -X GET "http://<server_ip>:48888/miner-positions?tier=30" \
  -H "Authorization: Bearer your_api_key"

# Check the size of the response without displaying it
curl -s --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key" -o /dev/null -w 'Size: %{size_download} bytes\n'

# View detailed headers to confirm compression
curl -v --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key" > /dev/null
```

## WebSocket API

The WebSocket API enables real-time streaming of trading data and updates.

### Connection and Authentication

After connecting to the WebSocket server at `ws://<server_ip>:8765`, clients must authenticate:

```json
{
  "api_key": "your_api_key",
  "last_sequence": -1
}
```

The `last_sequence` parameter helps track message continuity and allows clients to resume from where they left off.

### Websocket Message Format

Messages from the websocket server are parsed into PTNWebsocketMessage ([ptn_api/websocket_client.py](ptn_api/websocket_client.py)) objects which mirror the position data from the REST endpoint.

This data is serialized into a PTN Position object, and the server will send a message with the sequence number of the last message sent. Sequence number can be used to detect gaps in the message stream. The client can use this to make a REST call to fill in the gaps.

Lag info is also included in the message. lag from the queue is lag from when the order was made available to the websocket server. Lag from the order is lag from when the order was placed and is larger due to variable time in price filling and order processing in the PTN repo.

Here a snippet of a terminal printing PTNWebsocketMessage objects:
```bash
Received message PTNWebSocketMessage(seq=237)
Position Summary:
{
  "miner_hotkey": "5CUUWxGzf4qU5DCgLcL65qAKsQF1ezUTvBzfD548zPEDzxmR",
  "position_uuid": "0c2aa350-2bea-4ccd-8cce-6da220d5bae6",
  "open_ms": 1745897174103,
  "trade_pair": [
    "USDJPY",
    "USD/JPY",
    7e-05,
    0.1,
    5
  ],
  "current_return": 1.0,
  "close_ms": null,
  "net_leverage": -0.3000000000000007,
  "return_at_close": 1.0,
  "average_entry_price": 142.37951594322527,
  "cumulative_entry_value": -42.71385478296768,
  "realized_pnl": 0.0,
  "position_type": "SHORT",
  "is_closed_position": false
}
New Order:
{
  "trade_pair_id": "USDJPY",
  "order_type": "SHORT",
  "leverage": -0.3000000000000007,
  "price": 142.386,
  "bid": 142.386,
  "ask": 142.399,
  "slippage": 4.553858367205138e-05,
  "processed_ms": 1745897174103,
  "price_sources": [
    {
      "source": "Polygon_rest",
      "timespan_ms": 1000,
      "open": 142.39249999999998,
      "close": 142.39249999999998,
      "vwap": null,
      "high": 142.39249999999998,
      "low": 142.39249999999998,
      "start_ms": 1745897173000,
      "websocket": false,
      "lag_ms": 104,
      "bid": 142.386,
      "ask": 142.399
    }
  ],
  "order_uuid": "0c2aa350-2bea-4ccd-8cce-6da220d5bae6",
  "src": 0
}
Approx Timelag (ms): from_queue=49, from_order=1111
```

### API Key Management

Each API key is limited to a maximum of 5 concurrent WebSocket connections. When this limit is reached, the oldest connection will be automatically disconnected to make room for new connections.

## Running the Server

### Prerequisites

- Python 3.8+
- Required packages: flask, waitress, flask_compress, websockets

### Launching the Server

```bash
python api_manager.py --serve
```

## Client Usage Examples

### REST API with cURL

```bash
# Get positions with compression
curl --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key"

# Get positions with a specific tier
curl --compressed -X GET "http://<server_ip>:48888/miner-positions?tier=30" \
  -H "Authorization: Bearer your_api_key"

# Check the size of the response without displaying it
curl -s --compressed -X GET "http://<server_ip>:48888/miner-positions" \
  -H "Authorization: Bearer your_api_key" -o /dev/null -w 'Size: %{size_download} bytes\n'
```

### REST API with Python

```python
import requests
import json

# Basic request (compression is enabled by default in requests)
url = 'http://<server_ip>:48888/validator-checkpoint'
headers = {'Authorization': 'Bearer your_api_key'}

response = requests.get(url, headers=headers)
data = response.json()

# Check if compression was used
if 'Content-Encoding' in response.headers:
    print(f"Response was compressed using: {response.headers['Content-Encoding']}")
    # The requests library automatically decompresses the response
    print(f"Original compressed size (approximate): {len(response.content)} bytes")
    print(f"Decompressed size: {len(response.text)} bytes")

# Save to file
with open('validator_checkpoint.json', 'w') as f:
    json.dump(data, f)

# Using a session for multiple requests (more efficient)
with requests.Session() as session:
    session.headers.update({'Authorization': 'Bearer your_api_key'})
    
    # Get miner positions
    pos_response = session.get('http://<server_ip>:48888/miner-positions')
    positions = pos_response.json()
    
    # Get statistics
    stats_response = session.get('http://<server_ip>:48888/statistics')
    statistics = stats_response.json()
```

### WebSocket Client with Python

Our WebSocket client ([ptn_api/websocket_client.py](ptn_api/websocket_client.py)) provides a simple interface for receiving real-time data:

```python
from ptn_api.websocket_client import PTNWebSocketClient
import sys

...

if __name__ == "__main__":
    # Get API key from command line argument or use default
    api_key = sys.argv[1] if len(sys.argv) > 1 else "test_key"
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8765


    # Define a simple message handler. print position and order details
    def handle_messages(messages):
        for msg in messages:
            print(f"\nReceived message {msg}")
    # Create client
    print(f"Connecting to ws://{host}:{port} with API key: {api_key}")
    client = PTNWebSocketClient(api_key=api_key, host=host, port=port)

    # Run client
    client.run(handle_messages)
```

### Async Message Handler

For more complex processing, you can use async handlers:

```python
from ptn_api.websocket_client import PTNWebSocketClient, PTNWebSocketMessage
import asyncio
from typing import List


async def process_message(msg: PTNWebSocketMessage):
    # Example: Check bid/ask spread
    if msg.new_order.bid and msg.new_order.ask:
        spread = msg.new_order.ask - msg.new_order.bid
        spread_bps = (spread / msg.new_order.price) * 10000  # Basis points

        # Only log significant spreads
        if spread_bps > 10:  # More than 10 bps
            return f"{msg.new_order.trade_pair}: Spread {spread_bps:.1f} bps"

    # Example: Track specific pairs
    if msg.new_order.trade_pair == "BTCUSD":
        return f"BTC: ${msg.new_order.price:.2f} ({msg.new_order.order_type})"

    return None


async def async_handler(messages: List[PTNWebSocketMessage]):
    # Process messages concurrently
    results = await asyncio.gather(*[process_message(msg) for msg in messages])

    # Print non-None results
    for result in filter(None, results):
        print(result)


# Run with async handler
client = PTNWebSocketClient(api_key="your_api_key_here")
client.subscribe()
client.run(async_handler)
```


## Best Practices for Trading Logic

1. **Use the Provided Client:**
   - Our WebSocketClient handles all the complexity of reliable message delivery
   - Focus on your trading strategy instead of communication infrastructure

2. **Process Messages Efficiently:**
    - Create separate handler functions for different order types
    - Use properties of the order object to make trading decisions
    - Track processed order UUIDs to avoid duplicate processing
    - Check the timestamp and time lag to ensure orders aren't stale before acting on them

3. **Optimize Network Usage:**
    - Always use compression for REST API requests to reduce bandwidth
    - For bulk data retrieval, use the REST API with compression
    - For real-time updates, use the WebSocket API
    - Monitor compression ratios to ensure they're working as expected

## Security Considerations

Store your API keys in a secure location with appropriate file permissions. The system will automatically reload the keys when the file changes.

For production, deploy the API server behind a reverse proxy that handles SSL/TLS termination:

```
Client <--HTTPS--> Nginx/Apache <--HTTP--> API Server
```


## Best Practices for Trading Logic

1. **Use the Provided Client:**
   - Our WebSocketClient handles all the complexity of reliable message delivery
   - Focus on your trading strategy instead of communication infrastructure

2. **Process Messages Efficiently:**
    - Create separate handler functions for different message types
    - Use a switch/case pattern based on message content
    - Track processed order uuids to avoid duplicate processing
    - Check the timestamp of orders to ensure they aren't stale before acting on them

3. **Optimize Network Usage:**
    - Always use compression for REST API requests to reduce bandwidth
    - For bulk data retrieval, use the REST API with compression
    - For real-time updates, use the WebSocket API

## Security Considerations

Store your API keys in a secure location with appropriate file permissions. The system will automatically reload the keys when the file changes.

For production, deploy the API server behind a reverse proxy that handles SSL/TLS termination:

```
Client <--HTTPS--> Nginx/Apache <--HTTP--> API Server
```

## Final Notes

The [Request Network](https://request.taoshi.io/) is a Taoshi product which serves subnet data while handling security, rate limiting, data customization, and provides a polished customer-facing and validator setup UI. Running this repo is a prerequisite to serving data on the Request Network.