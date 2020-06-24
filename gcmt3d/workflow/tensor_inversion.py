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
from pycmt3d import Cmt3D
from pycmt3d import DataContainer
from pycmt3d import WeightConfig, Config
from pycmt3d.constant import PARLIST

# Gradient3D

import os
import glob

# Get logger to log progress
from gcmt3d import logger


def read_yaml_file(filename):
    """read yaml file"""
    with open(filename) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def invert(cmt_file_db):
    """Runs the actual inversion.

    :param cmt_file_db:
    :param param_path:
    :return: Nothing, inversion results are written to file.
    """

    # Inversion Params
    cmt_params = os.path.join(os.path.dirname(cmt_file_db),
                              "workflow_files", "params")
    inversionparam_path = os.path.join(cmt_params, "inversion_params",
                                       "InversionParams.yml")
    INV_params = read_yaml_file(inversionparam_path)

    # Weight Params
    weightparam_path = os.path.join(cmt_params, "inversion_params",
                                    "WeightParams.yml")
    weight_params = read_yaml_file(weightparam_path)

    # Get processing path from cmt_filename in database
    cmt_dir = os.path.dirname(os.path.abspath(cmt_file_db))

    # Create cmt source:
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmt_file_db)

    # Inversion dictionary directory
    inv_dict_dir = os.path.join(cmt_dir, "workflow_files", "inversion_dicts")

    # Inversion dictionaries
    inv_dict_files = glob.glob(os.path.join(inv_dict_dir,
                                            "cmt3d.*.inv_dict.yml"))

    # Inversion output directory
    inv_out_dir = os.path.join(cmt_dir,
                               "inversion", "cmt3d")

    # WRite start of inversion process
    logger.info(" ")
    logger.info("#######################################################")
    logger.info("#                                                     #")
    logger.info("#      Starting CMT3D Inversion ...                   #")
    logger.info("#                                                     #")
    logger.info("#######################################################")
    logger.info(" ")

    # Creating Data container
    dcon = DataContainer(parlist=PARLIST[:INV_params["npar"]])

    for _i, inv_dict_file in enumerate(inv_dict_files):

        # Get processing band
        wave = inv_dict_file.split(".")[1]

        logger.info(" ")
        logger.info("  " + 54 * "*")
        logger.info("  Getting data for inversion for %s waves." % wave)
        logger.info("  " + 54 * "*")
        logger.info(" ")

        # Load inversion file dictionary
        inv_dict = read_yaml_file(inv_dict_file)
        asdf_dict = inv_dict["asdf_dict"]
        window_file = inv_dict["window_file"]
        velocity = inv_dict["velocity"]
        wave_weight = inv_dict["weight"]

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
        logger.info("  Weight: %r" % wave_weight)
        logger.info("  Velocity: %r" % velocity)

        dcon.add_measurements_from_asdf(window_file, asdf_dict,
                                        wave_weight=wave_weight,
                                        wave_type=wave,
                                        velocity=velocity)

        logger.info("  _____________________________________________________")
        logger.info("  ... ")
        logger.info("  ")
        logger.info("  ... ")

    logger.info("  Setting up inversion classes .... ")
    logger.info("  " + 54 * "*")
    logger.info("  ... ")

    # Setting up weight config
    inv_weight_config = weight_params["weight_config"]

    weight_config = WeightConfig(
        normalize_by_energy=inv_weight_config["normalize_by_energy"],
        normalize_by_category=inv_weight_config["normalize_by_category"],
        azi_bins=inv_weight_config["azi_bins"],
        azi_exp_idx=inv_weight_config["azi_exp_idx"])

    # Setting up general inversion config
    inv_params = INV_params["config"]

    cmt3d_config = Config(
        INV_params["npar"],
        dlocation=float(inv_params["dlocation"]),
        ddepth=float(inv_params["ddepth"]),
        dmoment=float(inv_params["dmoment"]),
        envelope_coef=float(inv_params["envelope_coef"]),
        weight_data=bool(inv_params["weight_data"]),
        station_correction=bool(inv_params["station_correction"]),
        zero_trace=bool(inv_params["zero_trace"]),
        double_couple=bool(inv_params["double_couple"]),
        bootstrap=bool(inv_params["bootstrap"]),
        bootstrap_repeat=int(inv_params["bootstrap_repeat"]),
        weight_config=weight_config,
        taper_type=inv_params["taper_type"],
        damping=float(inv_params["damping"]))

    logger.info("  PyCMT3D is finding an improved CMTSOLUTION .... ")
    logger.info("  " + 54 * "*")
    logger.info(" ")
    logger.info(" ")

    # Invert for parameters
    cmt3d = Cmt3D(cmtsource, dcon, cmt3d_config)
    cmt3d.source_inversion()
    cmt3d.compute_new_syn()
    # Create inversion class
    # if bool(INV_params["gridsearch"]):
    #     inv = Inversion(cmtsource, dcon, cmt3d_config, grad3d_config)
    # else:
    #     inv = Inversion(cmtsource, dcon, cmt3d_config, mt_config=None)

    # Run inversion
    if bool(INV_params["statistics_plot"]):
        cmt3d.plot_stats_histogram(outputdir=inv_out_dir)

    # Plot results
    if bool(INV_params["summary_plot"]):
        cmt3d.plot_summary(inv_out_dir, figure_format="pdf")
    #
    # if bool(INV_params["statistics_plot"]):
    #     # Plot Statistics for inversion
    #     inv.G.plot_stats_histogram(outputdir=inv_out_dir)
    #
    if bool(INV_params["summary_json"]):
        cmt3d.write_summary_json(outputdir=inv_out_dir, mode="global")

    if bool(INV_params["write_new_cmt"]):
        cmt3d.write_new_cmtfile(outputdir=inv_out_dir)

    if bool(INV_params["write_new_synt"]):
        cmt3d.write_new_syn(outputdir=os.path.join(inv_out_dir, "new_synt"),
                            file_format="asdf")

    if bool(INV_params["plot_new_synthetics"]):
        cmt3d.plot_new_synt_seismograms(
            outputdir=os.path.join(inv_out_dir, "waveform_plots"),
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
