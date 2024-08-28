import asyncio
import json

import bittensor as bt
import uvicorn

import template.protocol
from fastapi import FastAPI, HTTPException

class Dashboard:
    def __init__(self, wallet, metagraph, config, is_testnet):
        self.wallet = wallet
        self.metagraph = metagraph
        self.config = config
        self.is_testnet = is_testnet

        self.miner_data = {}

        # asyncio.run(self.get_stats_positions_from_validator())

        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/miner/{miner_id}")
        async def get_miner_data(miner_id: str):
            asyncio.run(self.get_stats_positions_from_validator())
            return self.miner_data

            # TODO: make sure vali has miner?
            # if self.miner_data is not None:
            #     return self.miner_data
            # else:
            #     raise HTTPException(status_code=404, detail="Miner not found")

    def run(self, host="127.0.0.1", port=8000):
        uvicorn.run(self.app, host=host, port=port)

    async def get_stats_positions_from_validator(self):
        """
        get miner stats from validator
        """
        dendrite = bt.dendrite(wallet=self.wallet)
        if self.is_testnet:
            validator_axons = [n.axon_info for n in self.metagraph.neurons if n.hotkey == "5HY92b6TyroU2ZK1GHe5ccqu8QSFmzb835CDduQk2toDkRSZ"]
        else:
            validator_axons = [n.axon_info for n in self.metagraph.neurons if n.hotkey == "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v"]  # RT21 validator# self.position_inspector.get_possible_validators()  # RT21?

        try:
            miner_statistics_synapse = template.protocol.GetStatistics()
            validator_response = await dendrite.forward(axons=validator_axons, synapse=miner_statistics_synapse, timeout=10)

            for response in validator_response:
                if response.successfully_processed:
                    self.miner_data["statistics"] = response.stats
                    bt.logging.info("got miner stats")
                    break
                else:
                    if response.error_message:
                        bt.logging.info(f"Dashboard stats request failed with error [{response.error_message}]")


            miner_positions_synapse = template.protocol.GetPositions(for_dashboard=True)
            validator_response = await dendrite.forward(axons=validator_axons, synapse=miner_positions_synapse, timeout=10)

            for response in validator_response:
                if response.successfully_processed:
                    self.miner_data["positions"] = response.positions
                    bt.logging.info("got miner positions")
                    break
                else:
                    if response.error_message:
                        bt.logging.info(f"Dashboard positions request failed with error [{response.error_message}]")


            print("miner info: \n", json.dumps(self.miner_data, indent=4))
        except Exception as e:
            bt.logging.info(
                f"Unable to receive dashboard info from RT21 with error [{e}]")