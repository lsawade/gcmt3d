#!/usr/bin/env python
import matplotlib as mpl
import argparse
from gcmt3d.asdf.window import WindowASDF
from gcmt3d.utils.io import smart_read_yaml
import warnings
mpl.use('Agg')  # NOQA

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module=r'.*numerictypes')
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r'.*asdf_data_set')
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r'.*numerictypes')
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r'.*pyplot')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='params_file',
                        required=False, help="parameter file", default=None)
    parser.add_argument('-f', action='store', dest='path_file', required=True,
                        help="path file")
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help="verbose")
    args = parser.parse_args()

    # Little change to accommodate the full GCMT3D path file.
    if args.params_file is None:
        # Load process path file to get parameter file location
        try:
            params_file = smart_read_yaml(args.path_file)["window_param_file"]
        except KeyError:
            print("The given path file does not contain a parameter file "
                  "destination.")
            return
    else:
        params_file = args.params_file

    proc = WindowASDF(args.path_file, params_file,
                      verbose=args.verbose)
    proc.smart_run()


if __name__ == '__main__':
    main()
