#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from gcmt3d.asdf.process import ProcASDF
from gcmt3d.utils.io import smart_read_yaml
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r'.*numerictypes')
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r'.*asdf_data_set')
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r'.*numerictypes')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='params_file',
                        required=False, help="parameter file", default=None)
    parser.add_argument('-f', action='store', dest='path_file', required=True,
                        help="path file")
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help="verbose flag")
    args = parser.parse_args()

    # Little change to accommodate the full GCMT3D path file.
    if args.params_file is None:
        # Load process path file to get parameter file location
        try:
            params_file = smart_read_yaml(args.path_file)["process_param_file"]
        except KeyError:
            print("The given path file does not contain a parameter file "
                  "destination.")
            return
    else:
        params_file = args.params_file

    proc = ProcASDF(args.path_file, params_file, verbose=args.verbose)
    proc.smart_run()


if __name__ == '__main__':
    main()
