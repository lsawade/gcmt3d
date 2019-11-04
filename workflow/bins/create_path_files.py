#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script writes specfem sources into the respective simulation directories.

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
import argparse
from prepare_path_files import make_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)
    args = parser.parse_args()

    # Run
    make_paths(args.filename)