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
from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env
import os
import argparse

def main(cmt_filename):

    # Set directories of the parameter files
    param_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "params")
    request_param_path = os.path.join(param_path,
                                      "RequestParams/RequestParams.yml")

    # Read the parameter file
    req_params = smart_read_yaml(request_param_path, mpi_mode=is_mpi_env())

    # Earthquake and Station parameters
    cmt_dir = os.path.dirname(cmt_filename)

    # Create Request Object
    Request = DataRequest.from_file(cmt_filename,
                                    duration=req_params['duration'],
                                    channels=req_params['channels'],
                                    locations=req_params['locations'],
                                    starttime_offset=\
                                    req_params['starttime_offset'],
                                    outputdir=cmt_dir)

    # Print Earthquake Download Info
    print(Request)

    # Request download
    Request.download()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Set arguments
    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)

    # Get Arguments
    args = parser.parse_args()

    main(args.filename)
