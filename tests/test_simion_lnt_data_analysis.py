import sys
import os

import unittest
import numpy as np
import numpy.testing as nt

import pathlib

import simion
import simion.lnt_data_analysis


class Test(unittest.TestCase):
    def setUp(self):
        self.log_file_path = str(simion.__lib_path__) + "/../tests/test logs"

    def test_get_logfiles_data_small(self):
        data = simion.lnt_data_analysis.get_logfiles_data(self.log_file_path, "_start", ["Charge[e]", "Mass[amu]", "X Ion Init[mm]", "Y Ion Init[mm]"], 5, 2, only_success=False)

        assert len(data[0]) == 6
        
        nt.assert_almost_equal(data[0], np.array([ 4., 50., 57., 60., 65., 74.]))
        
        nt.assert_almost_equal(data[1], np.array([0., 0., 0., 0., 0., 0.]))
        nt.assert_almost_equal(data[2], np.array([1., 1., 1., 1., 1., 1.]))
        nt.assert_almost_equal(data[3], np.array([46.2389, 90.2021, 64.0189, 68.7892, 71.9572, 81.237 ]))
        nt.assert_almost_equal(data[4], np.array([58.3143, 56.2728, 68.6935, 56.8497, 58.3822, 57.4966]))

    def test_get_logfiles_data_empty(self):
        data = simion.lnt_data_analysis.get_logfiles_data(self.log_file_path, "_start", ["Charge[e]", "Mass[amu]", "X Ion Init[mm]", "Y Ion Init[mm]"], 3, 2, only_success=False)

        assert data == [None]*5

    def test_get_logfiles_data_full_empty(self):
        data = simion.lnt_data_analysis.get_logfiles_data(self.log_file_path, "_start", ["Charge[e]", "Mass[amu]", "X Ion Init[mm]", "Y Ion Init[mm]"], 4, 2, only_success=False)

        assert data == [None]*5