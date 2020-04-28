"""

Plot event of to asdf files and their windows

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2020


"""

# External imports
# import os
# import json
import logging
# from numpy import max, array, argsort, arange
# from pyasdf import ASDFDataSet
# from obspy import Stream, Inventory
# from obspy.geodetics import gps2dist_azimuth, locations2degrees
# from matplotlib.pyplot import plot, xlabel, ylabel, xlim, ylim, title
# from matplotlib.pyplot import figure, axes, show, savefig, tight_layout
#
# from matplotlib.patches import Rectangle

# Internal imports
from .plot_util import set_mpl_params_section
from ..log_util import modify_logger

# Get logger
logger = logging.getLogger(__name__)
modify_logger(logger)

# Set Rcparams for the section plot
set_mpl_params_section()


def plot_event(ev, inv):
    """Takes in an inventory and plots paths one a map."""
    pass
