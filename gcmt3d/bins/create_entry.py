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

from gcmt3d.workflow.create_database_entry import create_entry
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='filename',
                        help='Path to CMTSOLUTION file',
                        required=True, type=str)
    parser.add_argument('-d', dest='database',
                        help='Database directory',
                        required=True, type=str)
    parser.add_argument('-p', dest='param_path',
                        help='Path to param directory',
                        required=True, type=str)
    args = parser.parse_args()

    # Run
    create_entry(args.filename, args.database, args.param_path)


if __name__ == '__main__':
    main()
