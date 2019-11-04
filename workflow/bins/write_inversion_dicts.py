"""

This script contains binary to create inversion dictionaries.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2019

"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "entk"))
from shape_inversion_dictionaries import inversion_dictionaries
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='cmt_file',
                        required=True, help="Path to CMT file in database")
    parser.add_argument('-p', action='store', dest='param_path', required=True,
                        help="Path to Parameter Directory")
    args = parser.parse_args()

    inversion_dictionaries(args.cmt_file, args.param_path)