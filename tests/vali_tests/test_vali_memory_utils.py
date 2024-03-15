# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import json
import unittest

from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_memory_utils import ValiMemoryUtils


class TestValiMemoryUtils(TestBase):
    def test_set_and_get_vali_memory(self):
        # will use json as thats realistically what we'll be sending
        vm_data = {'test': 1, 'test2': 2}

        ValiMemoryUtils.set_vali_memory(json.dumps(vm_data))
        vm = ValiMemoryUtils.get_vali_memory()

        self.assertEqual(vm_data, json.loads(vm))


if __name__ == '__main__':
    unittest.main()
