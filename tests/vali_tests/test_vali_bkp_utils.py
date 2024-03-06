# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import json
import os
import unittest
import shutil

from tests.vali_tests.samples.testing_data import TestingData
from tests.vali_tests.base_objects.test_base import TestBase
from vali_config import ValiConfig
from vali_objects.cmw.cmw_util import CMWUtil
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


class TestValiBkpUtils(TestBase):
    def test_get_vali_bkp_dir(self):
        self.assertEqual(ValiBkpUtils.get_vali_bkp_dir(), ValiConfig.BASE_DIR + '/validation/backups/')

    def test_get_vali_responses_dir(self):
        self.assertEqual(ValiBkpUtils.get_vali_predictions_dir(), ValiConfig.BASE_DIR + '/validation/predictions/')

    def test_get_response_filename(self):
        self.assertEqual("test.pickle", ValiBkpUtils.get_response_filename("test"))

    def test_make_dir(self):
        test_dir = ValiConfig.BASE_DIR + '/validation/test_dir/'
        ValiBkpUtils.make_dir(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        shutil.rmtree(test_dir)

    def test_get_write_type(self):
        write_type = ValiBkpUtils.get_write_type(True)
        self.assertTrue(write_type, 'wb')
        write_type = ValiBkpUtils.get_write_type(False)
        self.assertTrue(write_type, 'w')

    def test_get_read_type(self):
        read_type = ValiBkpUtils.get_read_type(True)
        self.assertTrue(read_type, 'rb')
        read_type = ValiBkpUtils.get_read_type(False)
        self.assertTrue(read_type, 'r')

    def test_write_to_vali_dir(self):
        # test writing a valirecords file
        test_valirecords = "test_valirecords.json"
        test_cmw = CMWUtil.initialize_cmw()
        test_valirecords_location = ValiBkpUtils.get_vali_bkp_dir() + test_valirecords
        ValiBkpUtils.write_to_dir(test_valirecords_location, test_cmw)
        self.assertTrue(os.path.exists(test_valirecords_location))
        os.remove(test_valirecords_location)

        test_valipreds_location = ValiBkpUtils.get_vali_predictions_dir() + "test.pickle"
        ValiBkpUtils.write_to_dir(test_valipreds_location, TestingData.po, True)
        self.assertTrue(os.path.exists(test_valipreds_location))
        os.remove(test_valipreds_location)

    def test_write_and_read_vali_file(self):
        # test writing a valirecords file
        test_valirecords = "test_valirecords.json"
        test_cmw = CMWUtil.initialize_cmw()
        ValiBkpUtils.write_file(ValiBkpUtils.get_vali_bkp_dir(),
                                test_valirecords,
                                test_cmw)
        self.assertTrue(os.path.exists(ValiBkpUtils.get_vali_bkp_dir()+test_valirecords))

        test_valirecords_location = ValiBkpUtils.get_vali_bkp_dir() + test_valirecords
        get_test_cmw = json.loads(ValiBkpUtils.get_file(test_valirecords_location))
        self.assertEqual(test_cmw, get_test_cmw)

        os.remove(ValiBkpUtils.get_vali_bkp_dir()+test_valirecords)

        test_pred_filename = "test.pickle"
        ValiBkpUtils.write_file(ValiBkpUtils.get_vali_predictions_dir(),
                                test_pred_filename,
                                TestingData.po,
                                True)
        self.assertTrue(os.path.exists(ValiBkpUtils.get_vali_predictions_dir() + test_pred_filename))

        test_valipreds_location = ValiBkpUtils.get_vali_predictions_dir() + test_pred_filename
        get_test_valipreds = ValiBkpUtils.get_file(test_valipreds_location, True)
        self.assertTrue(get_test_valipreds == TestingData.po)

        os.remove(ValiBkpUtils.get_vali_predictions_dir() + test_pred_filename)

    def test_get_all_files_in_dir(self):
        # test writing a valirecords file
        test_valirecords = "test_valirecords.json"
        test_cmw = CMWUtil.initialize_cmw()
        test_valirecords_location = ValiBkpUtils.get_vali_bkp_dir() + test_valirecords
        ValiBkpUtils.write_to_dir(test_valirecords_location, test_cmw)

        vali_bkp_dir_files = ValiBkpUtils.get_all_files_in_dir(ValiBkpUtils.get_vali_bkp_dir())
        self.assertTrue(test_valirecords_location in vali_bkp_dir_files)

        os.remove(test_valirecords_location)


if __name__ == '__main__':
    unittest.main()
