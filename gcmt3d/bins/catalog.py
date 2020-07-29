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

from gcmt3d.stats.stats import Catalog
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", dest='ocmtfiles', help='old cmtfiles', type=str,
                        nargs="+")
    parser.add_argument("-n", dest='ncmtfiles', help='new cmtfiles', type=str,
                        nargs="+")
    parser.add_argument("-s", dest='statfiles', help='station files', type=str,
                        nargs="+")
    parser.add_argument('-f', dest='outfile', required=False, default='./',
                        help='Output file', type=str)
    args = parser.parse_args()

    # Load shit
    ST = Catalog(args.ocmtfiles, args.ncmtfiles, args.statfiles)

    # Save it
    ST.save(args.outfile)


if __name__ == "__main__":
    main()
