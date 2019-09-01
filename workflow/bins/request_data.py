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

from ..entk import data_request
import argparse


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Set arguments
    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)

    # Get Arguments
    args = parser.parse_args()

    data_request(args.filename)