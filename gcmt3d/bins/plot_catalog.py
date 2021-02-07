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

from gcmt3d.stats.stats import Catalog, CatalogStats
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='infile', required=False, default='./',
                        help='Input file', type=str)
    parser.add_argument('-o', dest='outdir', required=False, default='./',
                        help='OutputDirectory', type=str)
    args = parser.parse_args()

    # Load shit
    cat = Catalog.load(args.infile)

    # Save it
    cs = CatalogStats(cat)

    # Plot stuff
    cs.plot_changes(outdir=args.outdir)


if __name__ == "__main__":
    main()
