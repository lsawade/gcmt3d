#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')  # NOQA
import argparse
from gcmt3d.asdf.window import WindowASDF
import yaml

def read_yaml_file(filename):
    """read yaml file"""
    with open(filename) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


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
            params_file = read_yaml_file(args.path_file)["window_param_file"]
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
