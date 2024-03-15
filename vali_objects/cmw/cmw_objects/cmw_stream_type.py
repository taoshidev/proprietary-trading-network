# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from vali_objects.cmw.cmw_objects.cmw_miner import CMWMiner


class CMWStreamType:
    def __init__(self):
        self.stream_id = None
        self.topic_id = None
        self.miners = []

    def set_stream_id(self, stream_id):
        self.stream_id = stream_id
        return self

    def set_topic_id(self, topic_id):
        self.topic_id = topic_id
        return self

    def add_miner(self, miner: CMWMiner):
        self.miners.append(miner)

    def get_miner(self, miner_id):
        for miner in self.miners:
            if miner.miner_id == miner_id:
                return miner
        return None
