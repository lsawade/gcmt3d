"""

This script processes data according to the parameter files in the parameter
file directory.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2019

"""

import argparse
from ..entk.process_traces import process
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r'*.numerictypes')
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r'*.asdf_data_set')
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r'*.numerictypes')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    process(args.filename)
