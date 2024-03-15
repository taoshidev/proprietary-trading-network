# developer: Taoshidev
# Copyright © 2023 Taoshi, LLC

import json
from typing import Dict

from vali_objects.cmw.cmw_objects.cmw import CMW
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_miner import CMWMiner
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.exceptions.invalid_cmw_exception import InvalidCMWException


class CMWUtil:

    @staticmethod
    def load_cmw(vr) -> CMW:
        if "clients" in vr:
            cmw = CMW()
            for client in vr["clients"]:
                cmw_client = CMWClient().set_client_uuid(client["client_uuid"])
                for stream in client["streams"]:
                    cmw_stream = CMWStreamType().set_stream_id(stream["stream_id"]).set_topic_id(stream["topic_id"])
                    for miner in stream["miners"]:
                        cmw_stream.add_miner(CMWMiner(miner["miner_id"])
                                             .set_wins(miner["wins"])
                                             .set_win_value(miner["win_value"])
                                             .set_win_scores(miner["win_scores"])
                                             .set_unscaled_scores(miner["unscaled_scores"]))
                    cmw_client.add_stream(cmw_stream)
                cmw.add_client(cmw_client)
            return cmw
        else:
            raise InvalidCMWException("missing clients key in cmw")

    @staticmethod
    def dump_cmw(cmw: CMW) -> Dict:
        return json.loads(json.dumps(cmw, default=lambda o: o.__dict__))

    @staticmethod
    def initialize_cmw() -> Dict:
        return {
                "clients": []
            }