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

Last Update: January 2020
"""

from gcmt3d.stats.stats import Statistics
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('dirname', help='Database directory', type=str,
                        nargs="+")
    parser.add_argument('-d', action="store_true", dest='direct',
                        help='Direct directory')
    parser.add_argument('-f', dest='outfile', required=False, default='./',
                        help='Output file', type=str)
    args = parser.parse_args()

    if type(args.dirname) == list:
        dirs = args.dirname
    else:
        dirs = [args.dirname]

    # Load shit
    ST = Statistics._from_dir(dirs, direct=args.direct)

    # Save it
    ST.save(args.outfile)


if __name__ == "__main__":
    main()
