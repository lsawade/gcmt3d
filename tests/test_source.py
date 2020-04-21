#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pycmt3d tests suite.
Run with pytest.
:copyright:
    Wenjie Lei (lei@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import print_function, division
import inspect
import os
import obspy
from gcmt3d.source import CMTSource
import pytest
import tempfile
from obspy import read_events
from functions_for_testing import assertDictAlmostEqual

# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")


@pytest.fixture
def cmt():
    return CMTSource.from_CMTSOLUTION_file(CMTFILE)


def test_from_CMTSOLUTION_file(cmt):
    origin_time = obspy.UTCDateTime(2001, 9, 9, 23, 59, 17.78)
    cmt_time = origin_time + 2.0
    cmt_true = \
        CMTSource(origin_time=origin_time,
                  pde_latitude=34.0745, pde_longitude=-118.3792,
                  pde_depth_in_m=6400, mb=4.2, ms=4.2, region_tag="Hollywood",
                  eventname="9703873", cmt_time=cmt_time, half_duration=1.0,
                  latitude=34.1745, longitude=-118.4792, depth_in_m=5400.0,
                  m_rr=1.0e22, m_tt=-1.0e22)

    assert cmt == cmt_true


def test_write_CMTSOLUTION_File(tmpdir, cmt):
    fn = os.path.join(str(tmpdir), "CMTSOLUTION.temp")
    cmt.write_CMTSOLUTION_file(fn)
    new_cmt = CMTSource.from_CMTSOLUTION_file(fn)
    assert new_cmt == cmt


def test_load_quakeML():
    """Test the create directory method"""
    # Check one cmt file
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Cmtfile path
        cmtfile = os.path.join(DATA_DIR, "testCMT")

        # create new directory
        new_xml_path = os.path.join(tmp_dir, "tests.xml")
        xml = read_events(cmtfile)
        xml.write(new_xml_path, format="QUAKEML")

        assert(os.path.exists(new_xml_path)
               and os.path.isfile(new_xml_path))

        print("QuakeML\n", CMTSource.from_quakeml_file(new_xml_path))
        print("CMT\n", CMTSource.from_CMTSOLUTION_file(cmtfile))
        assertDictAlmostEqual(CMTSource.from_quakeml_file(new_xml_path),
                              CMTSource.from_CMTSOLUTION_file(cmtfile))
