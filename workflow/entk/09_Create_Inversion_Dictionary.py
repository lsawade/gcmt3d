#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script writes specfem sources into the respective simulation directories.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""


from gcmt3d.asdf.utils import smart_read_yaml
from gcmt3d.asdf.utils import write_yaml_file
from gcmt3d.asdf.utils import is_mpi_env
from gcmt3d.data.management.inversion_dicts import create_inversion_dict
import os
import glob
import argparse


def main(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Database Parameters
    DB_params = smart_read_yaml(databaseparam_path, mpi_mode=is_mpi_env())

    # Get processing path from cmt_filename in database
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Processed data directory
    processed_dir = os.path.join(cmt_dir, "seismograms",
                                "processed_seismograms")

    # Inversion dictionary directory
    inv_dict_dir = os.path.join(cmt_dir, "inversion", "inversion_dicts")

    # Get all files to be processed
    processed_files = glob.glob(os.path.join(processed_dir, "*observed*"))

    if DB_params["verbose"]:
        print("\n Creating inversion dictionaries ...\n")

    for _i, processed_file in enumerate(processed_files):

        # Get processing band
        bandstring = str(os.path.basename(processed_file)).split(".")[1]
        band = [float(x) for x in bandstring.split("_")]

        if DB_params["verbose"]:
            print("\nCreating inversion dictionary for period band:")
            print("Low: %d s || High: %d s \n" % tuple(band))
            print("...\n")

        params = create_inversion_dict(processed_dir, bandstring)

        # Print Inversion parameters:
        if DB_params["verbose"]:
            print("Files")
            print("_______________________________________________________\n")
            for key, value in params.items():
                print(key + ":", value)
            print("_______________________________________________________\n")

        # Outputfile:
        outfilename = os.path.join(inv_dict_dir,
                                   "inversion_file_dict."
                                   + bandstring + ".yml")

        # Write yaml file to inversion dictionary directory
        write_yaml_file(params, outfilename)

        if DB_params["verbose"]:
            print("\nDONE writing %s.\n" % outfilename)

    if DB_params["verbose"]:
        print("\nDONE writing all dictionaries!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)
    args = parser.parse_args()

    # Run
    main(args.filename)
