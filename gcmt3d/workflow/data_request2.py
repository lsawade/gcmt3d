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
from gcmt3d.utils.download import download_waveform
from gcmt3d.utils.download import download_stationxml
from gcmt3d.utils.download import read_station_file
from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env
from gcmt3d.source import CMTSource
import os

def data_request(cmt_filename):

    # Set directories of the parameter files
    param_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "params")

    # Get stations file
    STATIONS = os.path.join(param_path,
                                      "RequestParams/STATIONS")
    
    request_param_path = os.path.join(param_path,
                                      "RequestParams/RequestParams.yml")

    # Read the parameter file
    rCparams = smart_read_yaml(request_param_path, mpi_mode=is_mpi_env())

    # Earthquake and Station parameters
    cmt_dir = os.path.dirname(cmt_filename)
    station_dir = os.path.join(cmt_dir, "station_data")

    # Get STATIONS file from CMT directory
    stationsfile = os.path.join(station_dir, "STATIONS")

    # Observed output dir
    obsd_dir = os.path.join(cmt_dir, "seismograms", "obs")

    # CMT parameter input
    cmt = CMTSource.from_CMTSOLUTION_file(cmt_filename)
    duration = rCparams['duration']
    starttime_offset = rCparams['starttime_offset']
    
    starttime = cmt.origin_time + starttime_offset
    endtime = starttime + duration 

    # Get station_list from station_file in database entry
    stations = read_station_file(stationsfile)
    station_ids = [station[0] + "_" + station[1] 
                   for station in stations]
    
    # Download Station Data
    _, _, filtered_station_ids = \
        download_stationxml(station_ids, starttime, endtime, 
                            outputdir=station_dir, client=None,
                            level="response")
    
    # Download waveform
    download_waveform(filtered_station_ids, starttime, endtime, 
                      outputdir=obsd_dir, client=None)
