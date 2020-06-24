#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script finallay inverts the CMT Solution.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

from gcmt3d.workflow.tensor_rise import gradient

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='cmt_file',
                        required=True, help="Path to CMT file in database")
    args = parser.parse_args()

    gradient(args.cmt_file)


if __name__ == "__main__":
    main()
