#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This executable takes in an asdf file and plots its earthquake to station
distribution


:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

Last Update: January 2020
"""

import argparse
from obspy import read_events
from ..plot.plot_event import PlotEventSummary
from ..utils.io import load_asdf


def main():

    # Get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("cmt3d_json", type=str,
                        help="Stats JSON output by CMT3D")
    parser.add_argument("-g", dest="g3d_json", type=str, required=False,
                        help="Gridsearch stats JSON", default=None)
    parser.add_argument("-f", dest="outfile", type=str, required=False,
                        help="Outputfilename", default=None)
    # Parse Arguments
    args = parser.parse_args()
    cmt3d_json = args.cmt3d_json
    g3d_json = args.g3d_json
    outputfilename = args.outfile

    # Plot event summary
    P = PlotEventSummary(cmt3d=cmt3d_json, g3d=g3d_json)
    P.plot_summary(outputfilename=outputfilename)



if __name__ == "__main__":
    main()
