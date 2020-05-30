#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a script that will create a database entry given a cmt solution in the


:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from gcmt3d.workflow.copy_data import copy_data
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='filename',
                        help='Path to CMTSOLUTION file in the database',
                        required=True, type=str)
    parser.add_argument('-o', dest='obsd',
                        help='Path to directory containing the downloaded\n'
                             'observed files.',
                        required=True, type=str)
    parser.add_argument('-s', dest='synt',
                        help='Path to directory containing the simulated\n'
                             'synthetic files.',
                        required=True, type=str)
    parser.add_argument('-sta', dest='stations',
                        help='Path to directory containing the downloaded\n'
                             'station files.',
                        required=True, type=str)
    args = parser.parse_args()

    # Run The copy command
    copy_data(args.synt, args.obsd, args.stations, args.filename)



if __name__ == '__main__':
    main()
