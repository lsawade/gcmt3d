#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from gcmt3d.asdf.convert import ConvertASDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='path_file', required=True,
                        help="path file")
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help="verbose flag")
    parser.add_argument('-s', action='store_true', dest='status_bar',
                        help="status bar flag")
    args = parser.parse_args()

    if os.path.exists(args.path_file):
        converter = ConvertASDF(args.path_file, args.verbose, args.status_bar)
        converter.run()
    else:
        print("Path file does not exists.")


if __name__ == '__main__':
    main()
