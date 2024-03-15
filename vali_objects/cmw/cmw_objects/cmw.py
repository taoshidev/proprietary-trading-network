# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from vali_objects.cmw.cmw_objects.cmw_client import CMWClient


class CMW:
    def __init__(self):
        self.clients = []

    def add_client(self, cmw_client: CMWClient):
        self.clients.append(cmw_client)

    def client_exists(self, client: CMWClient):
        return client in self.clients

    def get_client(self, client_uuid: str):
        for client in self.clients:
            if client.client_uuid == client_uuid:
                return client
        return None

