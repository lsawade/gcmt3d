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
from gcmt3d.source import CMTSource
import os

def main():

    # Earthquake ID
    eqID = CMTSource.from_CMTSOLUTION_file(cmt_filename).eventname

    # specific output directory here left open if left open files will be
    # saved in the folder wher the `.cmt` file is located
    outputdir = os.path.join(databasedir, "eq_" + eqID)

    # Earthquake and Station parameters
    cmt_filename = os.path.join(outputdir, os.path.basename(outputdir)+".cmt")
    stationlist_filename = None

    # Create Request Object
    Request = DataRequest.from_file(cmt_filename,
                                    stationlistfname=stationlist_filename,
                                    duration=duration,
                                    channels=channels,
                                    locations=locations,
                                    starttime_offset=starttime_offset,
                                    outputdir=outputdir)

    # Print Earthquake Download Info
    print(Request)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Set arguments
    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)

    # Get Arguments
    args = parser.parse_args()

    main(args.filename)
