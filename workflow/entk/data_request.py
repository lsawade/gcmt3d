#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script will download the observed data. To the necessary places.


:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""


from gcmt3d.data import DataRequest
from gcmt3d.utils import download
from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env
import os

# Get logger to log progress
from gcmt3d import logger

def data_request(cmt_filename):

    # Set directories of the parameter files
    param_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "params")
    request_param_path = os.path.join(param_path,
                                      "RequestParams/RequestParams.yml")

    # Read the parameter file
    rCparams = smart_read_yaml(request_param_path, mpi_mode=is_mpi_env())

    # Earthquake and Station parameters
    cmt_dir = os.path.dirname(cmt_filename)
    station_dir = os.path.join(cmt_dir, "station_data")

    # Get STATIONS file from CMT directory
    stationsfile = os.path.join(station_dir, "STATIONS")

    # Create Request Object
    Request = DataRequest.from_file(cmt_filename,
                                    stationlistfname=stationsfile,
                                    sfstationlist=True,
                                    duration=rCparams['duration'],
                                    channels=rCparams['channels'],
                                    locations=rCparams['locations'],
                                    starttime_offset=\
                                    rCparams['starttime_offset'],
                                    outputdir=cmt_dir)

    # Print Earthquake Download Info
    for line in Request.__str__.splitlines():
        logger.info(line)

    # Request download
    Request.download()
