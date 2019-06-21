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
from gcmt3d.asdf.utils import is_mpi_env
from gcmt3d.source import CMTSource
from pycmt3d import DataContainer
from pycmt3d import DefaultWeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import Cmt3D

import os
import glob
import argparse


def main(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")

    # Load Database Parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = smart_read_yaml(databaseparam_path, mpi_mode=is_mpi_env())

    # Inversion Params
    inversionparam_path = os.path.join(param_path,
                                       "CMTInversion/InversionParams.yml")
    INV_params = smart_read_yaml(inversionparam_path, mpi_mode=is_mpi_env())

    # Get processing path from cmt_filename in database
    cmt_dir = os.path.dirname(os.path.abspath(cmt_filename))

    # Create cmt source:
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmt_filename)

    # Window directory
    window_dir = os.path.join(cmt_dir, "window_data")

    # Inversion dictionary directory
    inv_dict_dir = os.path.join(cmt_dir, "inversion", "inversion_dicts")

    # Inversion dictionaries
    inv_dicts = glob.glob(os.path.join(inv_dict_dir, "*"))

    # Inversion output directory
    inv_out_dir = os.path.join(cmt_dir, "inversion", "inversion_output")

    if DB_params["verbose"]:
        print("\n#######################################################")
        print("#                                                     #")
        print("#      Starting inversions ...                        #")
        print("#                                                     #")
        print("#######################################################\n")

    for _i, inv_dict in enumerate(inv_dicts):

        # Get processing band
        bandstring = str(os.path.basename(inv_dict)).split(".")[1]
        band = [float(x) for x in bandstring.split("_")]

        if band == [40.0, 100.0]:

            if DB_params["verbose"]:
                print("\n")
                print("  " + 54 * "*")
                print("  Starting inversion for period band:")
                print("  Low: %d s || High: %d s" % tuple(band))
                print("  " + 54 * "*" + "\n")

            # Load inversion file dictionary
            asdf_dict = smart_read_yaml(inv_dict, mpi_mode=is_mpi_env())
            window_file = os.path.join(window_dir,
                                       "windows." + bandstring
                                       + "#body_wave.json")

            # Print Inversion parameters:
            if DB_params["verbose"]:
                print("  Files used:")
                print("  _____________________________________________________\n")
                print("  Window file:\n", "  ", window_file)
                print("\n  ASDF files:")
                for key, value in asdf_dict.items():
                    print("    ", key + ":", value)
                print("  _____________________________________________________\n")
                print("  Inverting ... \n\n")


            # Creating Data container
            dcon = DataContainer(parlist=PARLIST[:DB_params["npar"]])
            dcon.add_measurements_from_asdf(window_file, asdf_dict)

            # Setting up weight config
            weight_config = DefaultWeightConfig(
                normalize_by_energy=False, normalize_by_category=False,
                comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
                love_dist_weight=1.0, pnl_dist_weight=1.0,
                rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

            # Setting up general inversion config
            config = Config(DB_params["npar"],
                            dlocation=float(INV_params["config"]["dlocation"]),
                            ddepth=float(INV_params["config"]["ddepth"]),
                            dmoment=float(INV_params["config"]["dmoment"]),
                            weight_data=True, station_correction=True,
                            zero_trace=True, double_couple=False,
                            bootstrap=True, bootstrap_repeat=100,
                            weight_config=weight_config)

            srcinv = Cmt3D(cmtsource, dcon, config)
            srcinv.source_inversion()

            # plot result
            srcinv.plot_summary(inv_out_dir, figure_format="pdf")

            if DB_params["verbose"]:
                print("  DONE inversion for period band: %d - %d s.\n"
                      % tuple(band))

    if DB_params["verbose"]:
        print("\n#######################################################")
        print("#                                                     #")
        print("#      All inversions DONE.                           #")
        print("#                                                     #")
        print("#######################################################\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file in database',
                        type=str)
    args = parser.parse_args()

    # Run
    main(args.filename)
