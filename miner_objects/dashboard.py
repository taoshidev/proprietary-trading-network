import asyncio
import bittensor as bt
import uvicorn
import template.protocol
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:5173",
]

class Dashboard:
    def __init__(self, wallet, metagraph, config, is_testnet):
        self.wallet = wallet
        self.metagraph = metagraph
        self.config = config
        self.is_testnet = is_testnet

        self.miner_data = {}

        self.app = FastAPI()
        self._add_cors_middleware()
        self._setup_routes()

    def _add_cors_middleware(self):
        # Set up CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],  # allow all HTTP methods
            allow_headers=["*"],  # allow all headers
        )

    def _setup_routes(self):
        @self.app.get("/miner")
        async def get_miner_data():
            bt.logging.info("Dashboard stats request processing")
            asyncio.run(self.get_stats_positions_from_validator())
            if self.miner_data:
                return self.miner_data
            else:
                empty_data = {"statistics": {"data": []}, "positions": []}
                return empty_data

    def run(self, host="127.0.0.1", port=8000):
        uvicorn.run(self.app, host=host, port=port)

    async def get_stats_positions_from_validator(self):
        """
        get miner stats from validator
        """
        dendrite = bt.dendrite(wallet=self.wallet)
        if self.is_testnet:
            validator_axons = self.metagraph.axons
        else:
            validator_axons = [n.axon_info for n in self.metagraph.neurons if n.hotkey == "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v"]  # RT21

        try:
            miner_dash_synapse = template.protocol.GetDashData()
            validator_response = await dendrite.forward(axons=validator_axons, synapse=miner_dash_synapse, timeout=5)

            for response in validator_response:
                if response.successfully_processed:
                    self.miner_data = response.data
                    bt.logging.info("Dashboard stats request succeeded")
                    break
                else:
                    if response.error_message:
                        bt.logging.info(f"Dashboard stats request failed with error [{response.error_message}]")
        except Exception as e:
            bt.logging.info(
                f"Unable to receive dashboard info from RT21 with error [{e}]")