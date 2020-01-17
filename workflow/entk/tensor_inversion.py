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

# CMT3D
from gcmt3d.source import CMTSource
from pycmt3d import DataContainer
from pycmt3d import DefaultWeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import Cmt3D

# GRID3D
from pycmt3d import Grid3d
from pycmt3d import Grid3dConfig

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
        bandstring = str(os.path.basename(inv_dict_file).split(".")[1])
        if "surface" in bandstring  or "body" in bandstring:
            bandstring = bandstring.split("#")[0]
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
        print("  Setting up inversion classes .... ")
        print("  " + 54 * "*" + "\n\n")



    # Setting up weight config
    inv_weight_config = INV_params["weight_config"]

    weight_config = DefaultWeightConfig(
        normalize_by_energy=inv_weight_config["normalize_by_energy"],
        normalize_by_category=inv_weight_config["normalize_by_category"],
        comp_weight=inv_weight_config["comp_weight"],
        love_dist_weight=inv_weight_config["love_dist_weight"],
        pnl_dist_weight=inv_weight_config["pnl_dist_weight"],
        rayleigh_dist_weight=inv_weight_config["rayleigh_dist_weight"],
        azi_exp_idx=inv_weight_config["azi_exp_idx"])

    # Setting up general inversion config
    grid3d_config = INV_params["grid3d_config"]

    g3d_config = Grid3dConfig(
        origin_time_inv=bool(grid3d_config["origin_time_inv"]),
        time_start=float(grid3d_config["time_start"]),
        time_end=float(grid3d_config["time_end"]),
        dt_over_delta=float(grid3d_config["dt_over_delta"]),
        energy_inv=bool(grid3d_config["energy_inv"]),
        energy_start=float(grid3d_config["energy_start"]),
        energy_end=float(grid3d_config["energy_end"]),
        denergy=float(grid3d_config["denergy"]),
        energy_keys=grid3d_config['energy_keys'],
        energy_misfit_coef=grid3d_config["energy_misfit_coef"],
        weight_data=bool(grid3d_config["weight_data"]),
        taper_type=grid3d_config["taper_type"],
        weight_config=weight_config)

    print(grid3d_config['energy_keys'])

    # Setting up general inversion config
    inv_config = INV_params["config"]

    config = Config(
        DB_params["npar"],
        dlocation=float(inv_config["dlocation"]),
        ddepth=float(inv_config["ddepth"]),
        dmoment=float(inv_config["dmoment"]),
        weight_data=bool(inv_config["weight_data"]),
        station_correction=bool(inv_config["station_correction"]),
        zero_trace=bool(inv_config["zero_trace"]),
        double_couple=bool(inv_config["double_couple"]),
        bootstrap=bool(inv_config["bootstrap"]),
        bootstrap_repeat=int(inv_config["bootstrap_repeat"])
        dtx=bool(inv_config["dtx"]),
        dtx=bool(inv_config["dM0"]),
        weight_config=weight_config)

    if DB_params["verbose"]:
        print("  Grid3d is finding better moment and origin time .... ")
        print("  " + 54 * "*" + "\n\n")

    grid3d = Grid3d(cmtsource, dcon, g3d_config)
    grid3d.search()

    # Plot Statistics
    grid3d.plot_stats_histogram(outputdir=inv_out_dir, figure_format="pdf")

    # Plot Misfit
    grid3d.plot_misfit_summary(outputdir=inv_out_dir, figure_format="pdf")

    if DB_params["verbose"]:
        print("  PyCMT3D is finding an improved CMTSOLUTION .... ")
        print("  " + 54 * "*" + "\n\n")

    srcinv = Cmt3D(cmtsource, dcon, config)
    srcinv.source_inversion()

    # Plot result
    srcinv.plot_summary(inv_out_dir, figure_format="pdf")

    # Plot Statistics
    srcinv.plot_stats_histogram(outputdir=inv_out_dir,
                                figure_format="pdf")

    srcinv.plot_new_synt_seismograms(outputdir=os.path.join(inv_out_dir,
                                                            "waveform_plots"),
                                     figure_format="pdf")

    # Write new CMT file
    srcinv.write_new_cmtfile(outputdir=inv_out_dir)

    # Compute and write new synthetics.
    srcinv.compute_new_syn()
    srcinv.write_new_syn(outputdir=os.path.join(inv_out_dir, "new_synt"),
                         file_format="asdf")


