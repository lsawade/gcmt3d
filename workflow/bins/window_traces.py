#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script will create the perturbed Moment tensors in the perturbation
directories.


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
from set_up_trace_windows import windowing
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)
    args = parser.parse_args()

    # Run
    windowing(args.filename)
