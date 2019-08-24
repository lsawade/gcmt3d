#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script will clean up the simulation directories

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""

from ..entk.polish_specfem_files import clean_up
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    clean_up(args.filename)