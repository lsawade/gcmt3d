"""

This script contains functions to invert the produced data

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2019

"""

import yaml

from gcmt3d.source import CMTSource
from pycmt3d import DataContainer
from pycmt3d import DefaultWeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import Cmt3D

import os
import glob


def read_yaml_file(filename):
    """read yaml file"""
    with open(filename) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)




def invert(cmt_file_db, param_path):
    """Runs the actual inversion.

    :param cmt_file_db:
    :param param_path:
    :return: Nothing, inversion results are written to file.
    """

    # Load Database Parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)

    # Inversion Params
    inversionparam_path = os.path.join(param_path,
                                       "CMTInversion/InversionParams.yml")
    INV_params = read_yaml_file(inversionparam_path)

    # Get processing path from cmt_filename in database
    cmt_dir = os.path.dirname(os.path.abspath(cmt_file_db))

    # Create cmt source:
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmt_file_db)

    # Inversion dictionary directory
    inv_dict_dir = os.path.join(cmt_dir, "inversion", "inversion_dicts")

    # Inversion dictionaries
    inv_dict_files = glob.glob(os.path.join(inv_dict_dir, "*"))

    # Inversion output directory
    inv_out_dir = os.path.join(cmt_dir, "inversion", "inversion_output")


    if DB_params["verbose"]:
        print("\n#######################################################")
        print("#                                                     #")
        print("#      Starting inversion ...                         #")
        print("#                                                     #")
        print("#######################################################\n")

    # Creating Data container
    dcon = DataContainer(parlist=PARLIST[:DB_params["npar"]])

    for _i, inv_dict_file in enumerate(inv_dict_files):

        # Get processing band
        bandstring = str(os.path.basename(inv_dict)).split(".")[1]
        band = [float(x) for x in bandstring.split("_")]

        if DB_params["verbose"]:
            print("\n")
            print("  " + 54 * "*")
            print("  Getting data for inversion from period band:")
            print("  Low: %d s || High: %d s" % tuple(band))
            print("  " + 54 * "*" + "\n")

        # Load inversion file dictionary
        inv_dict = read_yaml_file(inv_dict_file)
        asdf_dict = inv_dict["asdf_dict"]
        window_file = inv_dict["window_file"]

        # Adding measurements
        # Print Inversion parameters:
        if DB_params["verbose"]:
            print("  Adding measurements to data container:")
            print("  _____________________________________________________\n")

        # Add measurements from ASDF file and windowfile
        if DB_params["verbose"]:
            print("  Window file:\n", "  ", window_file)
            print("\n  ASDF files:")
            for key, value in asdf_dict.items():
                print("    ", key + ":", value)
        dcon.add_measurements_from_asdf(window_file, asdf_dict)

        if DB_params["verbose"]:
            print(
                "  _____________________________________________________\n")
            print("   ... \n\n")

    if DB_params["verbose"]:
        print("  Inverting for a new moment tensor .... ")
        print("  " + 54 * "*" + "\n\n")

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


if __name__ == "__main__":
