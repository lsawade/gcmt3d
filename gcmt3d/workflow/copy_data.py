"""

This script contains functions to copy data from wenjie's simulation
workflow.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2020

"""
import os
from shutil import copyfile
from glob import glob

import logging
from ..log_util import modify_logger

# Create logger
logger = logging.getLogger(__name__)
modify_logger(logger)

attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
        "CMT_depth", "CMT_lat", "CMT_lon"]

# Maps the CMT file namin to pycmt3d input
PARMAP = {"CMT": "",
          "CMT_rr": "Mrr",
          "CMT_tt": "Mtt",
          "CMT_pp": "Mpp",
          "CMT_rt": "Mrt",
          "CMT_rp": "Mrp",
          "CMT_tp": "Mtp",
          "CMT_depth": "dep",
          "CMT_lon": "lon",
          "CMT_lat": "lat",
          "CMT_ctm": "ctm",
          "CMT_hdr": "hdr"}


def copy_data(synt_dir, obsd_dir, station_dir, cmt_file_in_db):
    """

    Args:
        inversion_dir: directory where Wenjie saved the source inversion data
        cmt_file_in_db: CMT file in the database

    Returns:
        None. Just copies data

    """

    # Get CMT ID
    cmt_id = os.path.basename(cmt_file_in_db)[:-4]
    cmt_dir = os.path.dirname(os.path.abspath(cmt_file_in_db))

    # Observed data
    raw_obsd = os.path.join(os.path.abspath(obsd_dir), cmt_id, "*.mseed")
    raw_xml = os.path.join(os.path.abspath(station_dir), cmt_id, "*.xml")
    print(raw_obsd)
    logger.verbose("Looking for data     here: %s" % raw_obsd)
    logger.verbose("Looking for stations here: %s" % raw_xml)
    if len(raw_obsd) < 2 or len(raw_xml) < 2:
        raise ValueError("Not enough observed data")

    # Copying the station files
    station_dir_in_db = os.path.join(cmt_dir, "station_data")
    logger.verbose("Copying station files ...")
    for station_file in glob(raw_xml):
        dest = os.path.join(station_dir_in_db, os.path.basename(station_file))
        logger.debug("Copying %s to %s" % (station_file, dest))
        copyfile(station_file, dest)

    # Copying the observed files
    obsd_waveform_dir = os.path.join(cmt_dir, "seismograms", "obs")
    logger.verbose("Copying observed waveform files ...")
    for waveform_file in glob(raw_obsd):
        dest = os.path.join(obsd_waveform_dir, os.path.basename(waveform_file))
        logger.debug("Copying %s to %s" % (waveform_file, dest))
        copyfile(waveform_file, dest)

    # Copying synthetic files
    synt_waveform_dir = os.path.join(cmt_dir, "seismograms", "syn")
    logger.verbose("Copying synthetic waveform files ...")
    for _at in attr:
        if PARMAP[_at] == "":
            suffix = ""
        else:
            suffix = "_" + PARMAP[_at]
        # Get source file
        src = os.path.join(synt_dir, cmt_id + suffix,
                           "OUTPUT_FILES", "synthetic.h5")

        # Define destination
        dest = os.path.join(synt_waveform_dir, _at + ".h5")

        logger.debug("Copying %s to %s" % (src, dest))
        copyfile(src, dest)
