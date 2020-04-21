#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is a binary handler for the location of the CMT solution
inside the database.


:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

Last Update: January 2020
"""

from gcmt3d.utils.io import get_location_in_database
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('cmtfilename', type=str,
                        help='Path to CMTSOLUTION file')
    parser.add_argument('database', type=str,
                        help='Path to database directory')
    args = parser.parse_args()

    # Run
    print(get_location_in_database(args.cmtfilename, args.database))


if __name__ == "__main__":
    main()
