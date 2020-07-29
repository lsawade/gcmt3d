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

    parser.add_argument('file', help='Statistics pickle', type=str)
    parser.add_argument('-o', dest='outdir', required=False, default='./',
                        help='Output Dir', type=str)
    args = parser.parse_args()

    # Load shit
    ST = Statistics.load(args.file)

    # Plot it
    ST.plot_changes(savedir=args.outdir)


if __name__ == "__main__":
    main()
