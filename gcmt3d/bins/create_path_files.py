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
from gcmt3d.utils.io import read_yaml_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='filename',
                        help='Path to CMTSOLUTION file in database',
                        required=True, type=str)
    parser.add_argument('-p', dest='param_path',
                        help='Path to param directory',
                        required=True, type=str)
    parser.add_argument('-nc', dest='conversion', action='store_false',
                        help='No conversion of observed data if used')
    parser.add_argument('-fr', dest='frechet',
                        help='YAML filename with frechet distances\n'
                        'in degrees and kilometres', default=None,
                        required=False, type=str)
    args = parser.parse_args()

    if args.frechet is not None:
        frechet = read_yaml_file(args.frechet)
        ddeg = frechet["ddeg"]
        dz = frechet["dz"]
    else:
        ddeg, dz = None, None

    # Run
    make_paths(args.filename, args.param_path, args.conversion,
               ddeg=ddeg, dz=dz)


if __name__ == "__main__":
    main()
