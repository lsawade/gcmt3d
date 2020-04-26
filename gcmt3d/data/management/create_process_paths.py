"""

This script contains functions to create path files for the processing of the
observed and synthetic data.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2019

"""

import os
import logging
import copy

from ...utils.io import read_yaml_file
from ...log_util import modify_logger

# Create logger
logger = logging.getLogger(__name__)
modify_logger(logger)


attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
        "CMT_depth", "CMT_lat", "CMT_lon"]


def get_processing_pathlist():
    """Returns a list of all the path files used for processing. This
    depends naturally of the cmt location in the database and
    the gloabl cmt classification for which waves are goin to be used."""

    return list


def get_windowing_pathlist():
    """Returns a list of all the path files used for processing. This
    depends naturally of the cmt location in the database and
    the gloabl cmt classification for which waves are goin to be used."""

    return list


def get_inversion_pathlist():
    """Returns a list of all the path files used for processing. This
        depends naturally of the cmt location in the database and
        the global cmt classification for which waves are goin to be used."""

    return list


def create_processing_dictionary(cmtparamdict, obsd_process_dict,
                                 synt_process_dict):
    """This function creates the processing file that contains the info on
    how every wave type is processed. The output dictionary can then be
    saved to a yaml file for example.


    Args:
        cmtparamdict: This is the dictionary that is output by the the
        obsd_process_dict: Base processing dict usually from the parameter
                           directory
        synt_process_dict: Base processing dict usually from the parameter
                           directory

    Returns:
        complete processing dictionary
    """

    # copy the base dictionaries
    obsd_dict = copy.deepcopy(obsd_process_dict)
    synt_dict = copy.deepcopy(synt_process_dict)

    # New full dictionary
    outdict = dict()

    for wave, paramdict in cmtparamdict.items():
        tmpobsd = copy.deepcopy(obsd_dict)
        tmpsynt = copy.deepcopy(synt_dict)
        tmpobsd["pre_filt"] = [1.0/x
                               for x in sorted(paramdict["filter"])[::-1]]
        tmpsynt["pre_filt"] = [1.0/x
                               for x in sorted(paramdict["filter"])[::-1]]

        outdict[wave] = {"obsd": tmpobsd, "synt": tmpsynt}

    return outdict


def get_window_parameter_dict(window_config_dir):
    """This function creates the windowin dictionary that contains the info on
    how every wave type is windowed. The output dictionary can then be
    saved to a yaml file for example.


    Args:
        window_config_dir: directory containing the window parameter files
                           which are created semi-dynamically. (only filter
                           corner frequencies will be updated.)

    Returns:
        dictionary of window files for each wave type

    """

    # List of possible wavetypes
    waves = ["body", "surface", "mantle"]

    # populate dictionary using the wavetypess
    outdict = dict()
    for wave in waves:
        wavedict = read_yaml_file(os.path.join(window_config_dir,
                                               "window." + wave + ".yml"))
        outdict[wave] = wavedict

    return outdict


def create_windowing_dictionary(cmtparamdict, windowconfigdict):
    """This function creates the windowin dictionary that contains the info on
    how every wave type is windowed. The output dictionary can then be
    saved to a yaml file for example.


    Args:
        cmtparamdict: This is the dictionary that is output by the the
        obsd_process_dict: Base processing dict usually from the parameter
                           directory
        synt_process_dict: Base processing dict usually from the parameter
                           directory

    Returns:
        complete windowing dictionary
    """

    outdict = dict()

    for wave, paramdict in cmtparamdict.items():
        outdict[wave] = windowconfigdict[wave]

        outdict[wave]["min_period"] = paramdict["filter"][2]
        outdict[wave]["max_period"] = paramdict["filter"][1]

    return outdict


class PathFileCreator(object):

    def __init__(self, cmt_in_db, windowbasedir, processbasedir):
        """ Using the location of the CMTSOLUTION in the data base this
        class populates the dataase entry with path files that are need for the
        processing. it can also return the location of the full path list for
        example for EnTK where knowlegde of certain files prior to execution
        of the workflow is necessary.

        Args:
            cmt_in_db: cmt solution in the database
            windowbasedir: window parameter directory
            processbasedir: process base directory
        """

    @property
    def windowpathlist(self):
        pass
    @property
    def processpathlist(self):
        pass
    @property
    def inversionpathlist(self):
        pass



