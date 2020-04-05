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

import argparse
from gcmt3d.workflow.prepare_path_files import make_paths


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='filename',
                        help='Path to CMTSOLUTION file',
                        required=True, type=str)
    parser.add_argument('-p', dest='param_path',
                        help='Path to param directory',
                        required=True, type=str)
    args = parser.parse_args()

    # Run
    make_paths(args.filename, args.param_path)


if __name__ == "__main__":
    main()
