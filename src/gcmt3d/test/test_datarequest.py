#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for the data request class.

Run with pytest.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
:date:
    27.02.2019

"""
# from __future__ import print_function, division
# import inspect
from gcmt3d.data.download import DataRequest
from gcmt3d.data.download import InputError
import unittest





class TestDataRequest(unittest.TestCase):
    """
    Class to test the parameters of creating a time series data request.

    """

    def test_no_input(self):
        """
        Testing what happens when no data is given to the class.
        """
        # Checking whether error is thrown if no or wrong cmt is given
        with self.assertRaises(InputError):
            DataRequest()

