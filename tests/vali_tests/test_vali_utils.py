# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import json
import os
import unittest

from tests.vali_tests.samples.testing_data import TestingData
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.cmw.cmw_objects.cmw import CMW
from vali_objects.cmw.cmw_objects.cmw_client import CMWClient
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.dataclasses.client_request import ClientRequest
from vali_objects.exceptions.corrupt_data_exception import ValiBkpCorruptDataException
from vali_objects.exceptions.vali_bkp_file_missing_exception import ValiFileMissingException
from vali_objects.exceptions.vali_memory_missing_exception import ValiMemoryMissingException
from vali_objects.exceptions.vali_records_misalignment_exception import ValiRecordsMisalignmentException
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_memory_utils import ValiMemoryUtils
from vali_objects.utils.vali_utils import ValiUtils


class TestValiUtils(TestBase):

    def test_get_vali_records(self):
        vm = ValiUtils.get_vali_records()
        self.assertIsInstance(vm, CMW)

    def test_get_vali_bkp_json(self):
        vbkp = ValiUtils.get_vali_bkp_json()
        self.assertIsInstance(vbkp, dict)

    def test_get_vali_memory_json(self):
        try:
            ValiUtils.get_vali_memory_json()
        except Exception as e:
            self.assertIsInstance(e, ValiMemoryMissingException)

    def test_check_memory_matches_bkp(self):
        vm = ValiUtils.get_vali_records()
        self.assertTrue(ValiUtils.check_memory_matches_bkp())

        vm.add_client(CMWClient())
        ValiMemoryUtils.set_vali_memory(json.dumps(CMWUtil.dump_cmw(vm)))

        try:
            ValiUtils.check_memory_matches_bkp()
        except Exception as e:
            self.assertIsInstance(e, ValiRecordsMisalignmentException)

    def test_set_memory_with_bkp(self):
        vm = ValiMemoryUtils.get_vali_memory()
        self.assertIsNone(vm)
        ValiUtils.set_memory_with_bkp()
        vm = ValiUtils.get_vali_memory_json()
        self.assertIsNotNone(vm)

    def test_create_and_get_vali_predictions(self):
        test_pred_filename = "test"

        try:
            os.remove(ValiBkpUtils.get_vali_predictions_dir() + test_pred_filename + ".pickle")
        except Exception as e:
            pass

        try:
            ValiUtils.get_miner_positions(ValiBkpUtils.get_vali_predictions_dir()
                                          + test_pred_filename + ".pickle")
        except Exception as e:
            self.assertIsInstance(e, ValiFileMissingException)

        test_cmw = CMWUtil.initialize_cmw()
        ValiBkpUtils.write_file(ValiBkpUtils.get_vali_predictions_dir(),
                                test_pred_filename + ".pickle",
                                test_cmw)
        try:
            ValiUtils.get_miner_positions(ValiBkpUtils.get_vali_predictions_dir()
                                          + test_pred_filename + ".pickle")
        except Exception as e:
            self.assertIsInstance(e, ValiBkpCorruptDataException)

        os.remove(ValiBkpUtils.get_vali_predictions_dir() + test_pred_filename + ".pickle")

        ValiUtils.save_predictions_request(test_pred_filename, TestingData.po)
        testing_preds_output = ValiUtils.get_miner_positions(ValiBkpUtils.get_vali_predictions_dir()
                                                             + test_pred_filename + ".pickle")
        self.assertTrue(testing_preds_output == TestingData.po)
        os.remove(ValiBkpUtils.get_vali_predictions_dir() + test_pred_filename + ".pickle")

    def test_generate_standard_request(self):
        std_request = ValiUtils.generate_standard_request(ClientRequest)
        self.assertTrue(std_request.stream_type == "BTCUSD-5m")

    # RUN ONLY IF YOU DONT HAVE VALI RECORDS SET AS IT WILL OVERRIDE
    # def test_set_vali_memory_and_bkp(self):
    #     init_cmw = CMWUtil.load_cmw(CMWUtil.initialize_cmw())
    #     init_cmw.add_client(CMWClient())
    #
    #     ValiUtils.set_vali_memory_and_bkp(CMWUtil.dump_cmw(init_cmw))
    #
    #     vm = CMWUtil.dump_cmw(ValiUtils.get_vali_memory_json())
    #     vbkp = ValiUtils.get_vali_bkp_json()
    #
    #     unload_init_cw = CMWUtil.dump_cmw(init_cmw)
    #
    #     self.assertEqual(vm, unload_init_cw)
    #     self.assertEqual(vbkp, unload_init_cw)


if __name__ == '__main__':
    unittest.main()
