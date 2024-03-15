# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_objects.cmw_stream_type import CMWStreamType
from vali_objects.cmw.cmw_util import CMWUtil


class TestCMW(TestBase):

    def test_setup_cmw(self):
        client_uuid = 'fe43e80d-bb72-4773-8acc-50ee65b6413d'
        topic_id = 1
        stream_id = 1
        vm = CMWUtil.load_cmw(CMWUtil.initialize_cmw())

        client = vm.get_client(client_uuid)
        if client is None:
            cmw_client = CMWClient().set_client_uuid(client_uuid)
            cmw_client.add_stream(CMWStreamType().set_stream_id(stream_id).set_topic_id(topic_id))
            vm.add_client(cmw_client)
        else:
            client_stream_type = client.stream_exists(stream_id)
            if client_stream_type is None:
                client.add_stream(CMWStreamType().set_stream_id(stream_id).set_topic_id(topic_id))
        dumped_cmw = CMWUtil.dump_cmw(vm)

        dumped_cmw_test = {'clients': [{'client_uuid': 'fe43e80d-bb72-4773-8acc-50ee65b6413d',
              'streams': [{'miners': [], 'stream_id': 1, 'topic_id': 1}]}]}
        self.assertEqual(dumped_cmw, dumped_cmw_test)
