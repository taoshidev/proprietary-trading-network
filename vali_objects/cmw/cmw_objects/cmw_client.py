# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType


class CMWClient:
    def __init__(self):
        self.client_uuid = None
        self.streams = []

    def set_client_uuid(self, client_uuid):
        self.client_uuid = client_uuid
        return self

    def add_stream(self, cmw_stream_type: CMWStreamType):
        self.streams.append(cmw_stream_type)

    def get_stream(self, stream_id: str):
        for stream in self.streams:
            if stream.stream_id == stream_id:
                return stream
        return None

