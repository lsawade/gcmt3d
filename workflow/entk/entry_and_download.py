#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External imports
from __future__ import nested_scopes, generators, division, absolute_import, \
    with_statement, print_function
import os
import sys
import yaml
import logging
import argparse
import subprocess
from shlex import split


# Internal imports
from gcmt3d.workflow.create_database_entry import create_entry
from gcmt3d.workflow.data_request3 import data_request
from gcmt3d.utils.io import get_location_in_database
from gcmt3d.utils.io import get_cmt_id
from gcmt3d.log_util import modify_logger

logger = logging.getLogger('gcmt3d')
modify_logger(logger)


def read_yaml_file(filename):
    """read yaml file"""
    with open(filename) as fh:
        return yaml.load(fh, Loader=yaml.Loader)


def workflow(cmtfilenames, param_path):
    """This function submits the complete workflow

    :param cmt_filenames: str containing the path to the cmt solution that is
                          supposed to be inverted for

    Usage:
        ```bash
        python entry_and_download.py <path/to/cmtsolution(s)>
        ```

    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    db_params = read_yaml_file(databaseparam_path)

    for _cmtfile in cmtfilenames:

        logger.verbose("Creating Entry for Event: %s" % get_cmt_id(_cmtfile))

        # Create Entry
        create_entry(_cmtfile)

        # Earthquake specific database parameters: Dir and Cid
        cmt_file_db = get_location_in_database(_cmtfile,
                                               db_params["databasedir"])

        # # Download the data from the headnode before running the pipeline
        data_request(cmt_file_db, param_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmtfiles", help="Path to CMTSOLUTION file",
                        type=str, nargs="+")
    parser.add_argument("param_path", type=str,
                        help="Path to workflow paramater directory")
    args = parser.parse_args()

    # Run
    if type(args.cmtfiles) is str:
        cmtfiles = [args.cmtfiles]
    else:
        cmtfiles = args.cmtfiles

    workflow(cmtfiles, args.param_path)
