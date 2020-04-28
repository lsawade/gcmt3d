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

# External imports
import os
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader
import logging

# Internal imports
from ..asdf.utils import smart_read_yaml, is_mpi_env
from ..source import CMTSource
from ..log_util import modify_logger
from ..utils.io import flex_read_stations
from ..utils.obspy_utils import write_stations_file

logger = logging.getLogger(__name__)
modify_logger(logger)


def data_request(cmt_filename, param_path):

    # Request config_file
    request_param_path = os.path.join(param_path,
                                      "RequestParams/RequestParams.yml")

    # Read the parameter file
    rCparams = smart_read_yaml(request_param_path, mpi_mode=is_mpi_env())

    # Earthquake and Station parameters
    cmt_dir = os.path.dirname(cmt_filename)
    station_dir = os.path.join(cmt_dir, "station_data")

    # Observed output dir
    obsd_dir = os.path.join(cmt_dir, "seismograms", "obs")

    # CMT parameter input
    cmt = CMTSource.from_CMTSOLUTION_file(cmt_filename)
    duration = rCparams['duration']
    starttime_offset = rCparams['starttime_offset']

    starttime = cmt.origin_time + starttime_offset
    endtime = starttime + duration

    # Get station_list from station_file in database entry
    network_string = rCparams["networks"]

    # Set domain containing all locations
    # Rectangular domain containing parts of southern Germany.
    domain = RectangularDomain(minlatitude=-90, maxlatitude=90,
                               minlongitude=-180, maxlongitude=180)

    # Set download restrictions
    restrictions = Restrictions(
        starttime=starttime,
        endtime=endtime,
        reject_channels_with_gaps=False,
        minimum_length=float(rCparams['minimum_length']),
        # Trace needs to be almost full length
        network=network_string,  # Only certain networks
        channel=",".join(rCparams['channels']),
        location=",".join(rCparams['locations']))

    # No specified providers will result in all known ones being queried.
    providers = ["IRIS"]
    mdl = MassDownloader(providers=providers)
    # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
    # folders with automatically chosen file n
    stationxml_storage = station_dir
    waveform_storage = obsd_dir
    logger.info("MSEEDs: %s" % waveform_storage)
    logger.info("XMLs: %s" % stationxml_storage)

    mdl.download(domain, restrictions, mseed_storage=waveform_storage,
                 stationxml_storage=stationxml_storage)

    # Post download read stations and create STATIONS file for SPECFEM.
    inv = flex_read_stations(os.path.join(station_dir, "*.xml"))
    write_stations_file(inv, filename=os.path.join(station_dir, "STATIONS"))
