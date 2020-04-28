#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This executable will load and asdf file and plots the traces as
seismic sections.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

Last Update: January 2020
"""

import argparse
from ..plot.plot_section import plot_section


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", dest="observed", type=str, required=True,
                        help="Observed ASDF file")
    parser.add_argument("-s", dest="synthetic", type=str or None,
                        required=False, default=None,
                        help="Synthetic ASDF file")
    parser.add_argument("-w", dest="windows", type=str or None,
                        required=False, default=None,
                        help="Window JSON")

    args = parser.parse_args()
    obsdfile = args.observed
    syntfile = args.synthetic
    winfile = args.windows
    plot_section(obsdfile, synt_file_name=syntfile, window_file_name=winfile)


if __name__ == "__main__":
    main()
