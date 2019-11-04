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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "entk"))
from create_database_entry import create_entry
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    create_entry(args.filename)