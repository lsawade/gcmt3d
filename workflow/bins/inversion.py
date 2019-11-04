#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script finallay inverts the CMT Solution.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "entk"))
from tensor_inversion import invert
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='cmt_file',
                        required=True, help="Path to CMT file in database")
    parser.add_argument('-p', action='store', dest='param_path', required=True,
                        help="Path to Parameter Directory")
    args = parser.parse_args()

    invert(args.cmt_file, args.param_path)
