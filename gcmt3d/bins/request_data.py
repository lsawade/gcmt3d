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

from gcmt3d.workflow.data_request3 import data_request
import argparse


def main():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Set arguments
    parser.add_argument('-f', dest='filename', required=True,
                        help='Path to CMTSOLUTION file in database',
                        type=str)

    parser.add_argument('-p', dest='param_path', required=True,
                        help='Path to parameter directory',
                        type=str)

    # Get Arguments
    args = parser.parse_args()

    data_request(args.filename, args.param_path)


if __name__ == "__main__":
    main()
