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
from pycmt3d.source import CMTSource
from pycmt3d import DataContainer
from pycmt3d import WeightConfig, Config
from pycmt3d.constant import PARLIST
from pycmt3d import Inversion

# Gradient3D
from pycmt3d.gradient3d import Gradient3dConfig

import os
import glob

# Get logger to log progress
from gcmt3d import logger


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

    # WRite start of inversion process
    logger.info(" ")
    logger.info("#######################################################")
    logger.info("#                                                     #")
    logger.info("#      Starting inversion ...                         #")
    logger.info("#                                                     #")
    logger.info("#######################################################")
    logger.info(" ")

    # Creating Data container
    dcon = DataContainer(parlist=PARLIST[:DB_params["npar"]])

    for _i, inv_dict_file in enumerate(inv_dict_files):

        # Get processing band
        bandstring1 = str(os.path.basename(inv_dict_file).split(".")[1])
        if "surface" in bandstring1 or "body" in bandstring1:
            bandstring = bandstring1.split("#")[0]
        else:
            bandstring = bandstring1

        band = [float(x) for x in bandstring.split("_")]

        logger.info(" ")
        logger.info("  " + 54 * "*")
        logger.info("  Getting data for inversion from period band:")
        logger.info("  Low: %d s || High: %d s" % tuple(band))
        logger.info("  " + 54 * "*")
        logger.info(" ")

        # Load inversion file dictionary
        inv_dict = read_yaml_file(inv_dict_file)
        asdf_dict = inv_dict["asdf_dict"]
        window_file = inv_dict["window_file"]

        # Adding measurements
        # Print Inversion parameters:
        logger.info("  Adding measurements to data container:")
        logger.info("  _____________________________________________________")
        logger.info(" ")

        # Add measurements from ASDF file and windowfile
        logger.info("  Window file:")
        logger.info("   " + window_file)
        logger.info(" ")
        logger.info("  ASDF files:")
        for key, value in asdf_dict.items():
            logger.info("     " + key + ": " + value)

        dcon.add_measurements_from_asdf(window_file, asdf_dict)

        logger.info("  _____________________________________________________")
        logger.info("  ... ")
        logger.info("  ")
        logger.info("  ... ")

    logger.info("  Setting up inversion classes .... ")
    logger.info("  " + 54 * "*")
    logger.info("  ... ")

    # Setting up weight config
    inv_weight_config = INV_params["weight_config"]

    weight_config = WeightConfig(
        normalize_by_energy=inv_weight_config["normalize_by_energy"],
        normalize_by_category=inv_weight_config["normalize_by_category"],
        azi_bins=inv_weight_config["azi_bins"],
        # comp_weight=inv_weight_config["comp_weight"],
        # love_dist_weight=inv_weight_config["love_dist_weight"],
        # pnl_dist_weight=inv_weight_config["pnl_dist_weight"],
        # rayleigh_dist_weight=inv_weight_config["rayleigh_dist_weight"],
        azi_exp_idx=inv_weight_config["azi_exp_idx"])

    # Setting up general inversion config
    inv_params = INV_params["config"]

    cmt3d_config = Config(
        DB_params["npar"],
        dlocation=float(inv_params["dlocation"]),
        ddepth=float(inv_params["ddepth"]),
        dmoment=float(inv_params["dmoment"]),
        weight_data=bool(inv_params["weight_data"]),
        station_correction=bool(inv_params["station_correction"]),
        zero_trace=bool(inv_params["zero_trace"]),
        double_couple=bool(inv_params["double_couple"]),
        bootstrap=bool(inv_params["bootstrap"]),
        bootstrap_repeat=int(inv_params["bootstrap_repeat"]),
        weight_config=weight_config,
        taper_type=inv_params["taper_type"],
        damping=float(inv_params["damping"]))

    grad3d_params = INV_params["grad3d_config"]

    weight_config_grad3d = WeightConfig(
        normalize_by_energy=True,  # Fixes issue
        normalize_by_category=True,  # Fixes issue
        # comp_weight=inv_weight_config["comp_weight"],
        # love_dist_weight=inv_weight_config["love_dist_weight"],
        # pnl_dist_weight=inv_weight_config["pnl_dist_weight"],
        # rayleigh_dist_weight=inv_weight_config["rayleigh_dist_weight"],
        azi_exp_idx=inv_weight_config["azi_exp_idx"])

    grad3d_config = Gradient3dConfig(
        method=grad3d_params["method"],
        weight_data=bool(grad3d_params["weight_data"]),
        weight_config=weight_config_grad3d,
        use_new=bool(grad3d_params["use_new"]),
        # flag to use the gradient method on inverted traces.
        taper_type=grad3d_params["taper_type"],
        c1=float(grad3d_params["c1"]),
        c2=float(grad3d_params["c2"]),
        idt=float(grad3d_params["idt"]),
        ia=float(grad3d_params["ia"]),
        nt=int(grad3d_params["nt"]),
        nls=int(grad3d_params["nls"]),
        crit=float(grad3d_params["crit"]),
        precond=bool(grad3d_params["precond"]),
        reg=bool(grad3d_params["reg"]),
        bootstrap=bool(grad3d_params["bootstrap"]),
        bootstrap_repeat=int(grad3d_params["bootstrap_repeat"]),
        bootstrap_subset_ratio=float(grad3d_params["bootstrap_subset_ratio"]),
        mpi_env=bool(grad3d_params["mpi_env"]),
        parallel=bool(grad3d_params["parallel"]))

    logger.info("  PyCMT3D is finding an improved CMTSOLUTION .... ")
    logger.info("  " + 54 * "*")
    logger.info(" ")
    logger.info(" ")

    # Create inversion class
    if bool(INV_params["gridsearch"]):
        inv = Inversion(cmtsource, dcon, cmt3d_config, grad3d_config)
    else:
        inv = Inversion(cmtsource, dcon, cmt3d_config, mt_config=None)

    # Run inversion
    inv.source_inversion()

    # Plot results
    if bool(INV_params["summary_plot"]):
        inv.plot_summary(inv_out_dir, figure_format="pdf")

    if bool(INV_params["summary_json"]):
        inv.write_summary_json(outputdir=inv_out_dir, mode="global")

    if bool(INV_params["write_new_cmt"]):
        inv.write_new_cmtfile(outputdir=inv_out_dir)

    if bool(INV_params["write_new_synt"]):
        inv.write_new_syn(outputdir=os.path.join(inv_out_dir, "new_synt"),
                          file_format="asdf")

    if bool(INV_params["plot_new_synthetics"]):
        inv.plot_new_synt_seismograms(outputdir=os.path.join(inv_out_dir,
                                                             "waveform_plots"),
                                      figure_format="pdf")
    if bool(INV_params["statistics_plot"]):
        # Plot Statistics for inversion
        inv.cmt3d.plot_stats_histogram(outputdir=inv_out_dir,
                                       figure_format="pdf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='cmt_file',
                        required=True, help="Path to CMT file in database")
    parser.add_argument('-p', action='store', dest='param_path', required=True,
                        help="Path to Parameter Directory")
    args = parser.parse_args()

    invert(args.cmt_file, args.param_path)
