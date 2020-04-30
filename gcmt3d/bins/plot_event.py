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
from ..plot.plot_event import plot_event
from ..utils.io import load_asdf


def main():

    # Get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("observed", type=str,
                        help="ASDF file")
    parser.add_argument("-f", dest="outfile", type=str, required=False,
                        help="Outputfilename", default=None)
    parser.add_argument("-p", dest="projection", type=str, required=False,
                        help="Map", default="azi_equi")
    parser.add_argument("-c", dest="cmt", type=str, required=False,
                        help="CMTSOLUTION as event", default=None)
    # Parse Arguments
    args = parser.parse_args()
    asdffile = args.observed
    outputfilename = args.outfile
    projection = args.projection
    cmt = args.cmt

    # Load asdf file
    if cmt is not None:
        inv, _ = load_asdf(asdffile, no_event=True)
        event = read_events(cmt)[0]
    else:
        event, inv, _ = load_asdf(asdffile)

    # Plot event
    plot_event(event, inv, filename=outputfilename, projection=projection)


if __name__ == "__main__":
    main()
