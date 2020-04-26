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

from .process_classifier import ProcessParams
from ...source import CMTSource
from ...utils.io import read_yaml_file
from ...utils.io import write_yaml_file
from ...log_util import modify_logger

# Create logger
logger = logging.getLogger(__name__)
modify_logger(logger)


attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
        "CMT_depth", "CMT_lat", "CMT_lon"]

# Maps the CMT file namin to pycmt3d input
PARMAP = {"CMT_rr": "Mrr",
          "CMT_tt": "Mtt",
          "CMT_pp": "Mpp",
          "CMT_rt": "Mrt",
          "CMT_rp": "Mrp",
          "CMT_tp": "Mtp",
          "CMT_depth": "dep",
          "CMT_lon": "lon",
          "CMT_lat": "lat",
          "CMT_ctm": "ctm",
          "CMT_hdr": "hdr"}

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


class PathCreator(object):

    def __init__(self, cmt_in_db, windowbasedir, processbasedir, npar=9,
                 figure_mode=True):
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

        # CMT
        self.cmtfile = cmt_in_db
        self.cmtdir = os.path.dirname(os.path.abspath(cmt_in_db))
        self.cmtsource = CMTSource.from_CMTSOLUTION_file(cmt_in_db)
        p = ProcessParams(self.cmtsource.moment_magnitude,
                          self.cmtsource.depth_in_m)
        self.cmtconfigdict = p.determine_all()
        self.npar = npar

        # File locations
        # Path directories
        self.pathdir = os.path.join(self.cmtdir, "workflow_files",
                                    "path_files")
        self.process_path_dir = os.path.join(self.pathdir, 'process_paths')
        self.window_path_dir = os.path.join(self.pathdir, 'window_paths')

        # Parameter directories
        self.paramdir = os.path.join(self.cmtdir, "workflow_files",
                                     "params")
        self.process_param_dir = os.path.join(self.paramdir, 'process_params')
        self.window_param_dir = os.path.join(self.paramdir, 'window_params')

        # Inversion directory
        self.invdir = os.path.join(self.cmtdir, "workflow_files",
                                     "inversion_dicts")

        # Window
        self.windowparam_dict = get_window_parameter_dict(windowbasedir)
        self.windowdict = create_windowing_dictionary(
            self.cmtconfigdict, self.windowparam_dict)

        # Processbases
        self.obsd_base_procdict = read_yaml_file(os.path.join(
            processbasedir, "process_observed.yml"))
        self.synt_base_procdict = read_yaml_file(os.path.join(
            processbasedir, "process_synthetic.yml"))
        self.processdict = create_processing_dictionary(
            self.cmtconfigdict, self.obsd_base_procdict,
            self.synt_base_procdict)

        # Dictionaries to be created
        self.windowparam_file_dict = None
        self.windowpath_file_dict = None
        self.processparam_file_dict = None
        self.processpath_file_dict = None
        self.cmt3d_invdicts = None
        self.g3d_invdicts = None

        # Miscellaneous
        self.figure_mode = figure_mode

        # Create base dictionaries
        self.create_window_parameter_struct()
        self.create_window_path_file_struct()
        self.create_process_parameter_struct()
        self.create_process_path_file_struct()
        self.create_create_inversion_structs()

    @property
    def windowpathlist(self):
        pass

    @property
    def processpathlist(self):
        """Returns the processpaath filenames in form of a list"""
        files = []
        for wave, datadict in self.processdict.items():
            for datatype, paramdict in datadict.items():
                files.append(os.path.join(self.process_param_dir,
                                          wave + "." + datatype + ".yml"))

    @property
    def inversionpathlist(self):
        pass

    def write_all(self):
        """writes out all created dictionaries."""

        writelist = [self.windowparam_file_dict,
                     self.windowpath_file_dict,
                     self.processparam_file_dict,
                     self.processpath_file_dict,
                     self.cmt3d_invdicts]

        for outdict in writelist:
            self.write_param_file_dict(outdict)


    def create_process_parameter_struct(self):
        """Creates window parameter dictionaries and corresponding file names
        for the parameter files to be written to disk:

        .. rubric:: Content

        I would write about the content f the processing files, but that would
        be a bit long for the doc string, instead this will be part of the
        general documentation.

        .. rubric:: Location in database

        The dots represent the CMT directory in the database. For certain
        magnitudes, a mantle wave path file is generated as well.

        .. code-block:: bash

            .../workflow_files/params/process_params/body.obsd.process.yml
            .../workflow_files/params/process_params/body.synt.process.yml
            .../workflow_files/params/process_params/surface.obsd.process.yml
            .../workflow_files/params/process_params/surface.synt.process.yml

        """
        self.processparam_file_dict = dict()
        for wave, datadict in self.processdict.items():
            for datatype, paramdict in datadict.items():
                filename = os.path.join(self.process_param_dir,
                                        wave + "." + datatype
                                        + ".process.yml")
                self.processparam_file_dict[wave + "_" + datatype] \
                    = {"filename": filename,
                       "params": paramdict}

    def create_window_parameter_struct(self):
        """Creates the window parameter structs to be written to the database.

        .. rubric:: Content

        I would write about the content f the processing files, but that would
        be a bit long for the doc string, instead this will be part of the
        general documentation.

        .. rubric:: Location in database

        The dots represent the CMT directory in the database. For certain
        magnitudes, a mantle wave path file is generated as well.

        .. code-block:: bash

            .../workflow_files/params/window_params/body.window.yml
            .../workflow_files/params/window_params/surface.window.yml

        """
        self.windowparam_file_dict = dict()

        for wave, paramdict in self.windowdict.items():
                filename = os.path.join(self.window_param_dir,
                                        wave + ".window.yml")
                self.windowparam_file_dict[wave] \
                    = {"filename": filename,
                       "params": paramdict}

    def create_process_path_file_struct(self):
        """This method creates the path files used to perform the processing
        on the traces.

        .. rubric:: Content of a path file

        For the body wave path file the content should look as follows, where
        the dots represent the location of the CMT directory in the database

        .. code-block:: yaml

            input_asdf: .../seismograms/obs/raw_observed.h5
            input_tag: obs
            output_asdf: .../seismograms/processed_seismograms/body.obsd.h5
            output_tag: Body
            process_param_file: .../workflow_files/params/\
            process_params/body.obsd.process.yml

        For a perturbed CMT solution the content looks as follows:

        .. code-block:: yaml

            input_asdf: .../seismograms/syn/CMT_rr.h5
            input_tag: syn
            output_asdf: .../seismograms/processed_seismograms/\
            body.synt.CMT_rr.h5
            output_tag: Body_CMT_rr
            process_param_file: .../workflow_files/params/\
            process_params/body.synt.process.yml

        .. rubric:: Location in database

        The dots represent the CMT directory in the database. For certain
        magnitudes, a mantle wave path file is generated as well.

        .. code-block:: bash

            .../workflow_files/path_files/process_paths/\
            body.obsd.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_rr.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_tt.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_pp.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_rt.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_rp.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_tp.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_depth.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_lat.process_path.yml
            .../workflow_files/path_files/process_paths/\
            body.synt.CMT_lon.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.obsd.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_rr.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_tt.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_pp.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_rt.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_rp.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_tp.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_depth.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_lat.process_path.yml
            .../workflow_files/path_files/process_paths/\
            surface.synt.CMT_lon.process_path.yml

        """

        # Create the observed path files.
        seismos = os.path.join(self.cmtdir, 'seismograms')
        obs = os.path.join(seismos, 'obs', 'raw_observed.h5')
        syndir = os.path.join(seismos, 'syn')
        procdir = os.path.join(self.cmtdir, 'seismograms',
                               'processed_seismograms')
        suffix = ".h5"

        self.processpath_file_dict = dict()

        for wave, datadict in self.processdict.items():
            for datatype in datadict.keys():

                if datatype == "obsd":
                    input_asdf = obs
                    input_tag = "obs"
                    output_asdf = os.path.join(procdir,
                                               wave + "." + datatype
                                               + suffix)
                    output_tag = wave.capitalize()

                    # Define process parameter file
                    process_param_file = os.path.join(self.process_param_dir,
                                                      wave + "." + datatype
                                                      + ".process.yml")

                    # Define path output dictionary
                    outdict = {
                        "input_asdf": input_asdf,
                        "input_tag": input_tag,
                        "output_asdf": output_asdf,
                        "output_tag": output_tag,
                        "process_param_file": process_param_file
                    }

                    # Find file name
                    filename = os.path.join(self.process_path_dir,
                                            wave + "." + datatype
                                            + ".process_path.yml")

                    self.processpath_file_dict[wave + "_" + datatype] \
                        = {"filename": filename,
                           "params": outdict}

                else:

                    # Loop over CMT perturbations
                    for at in attr:
                        input_asdf = os.path.join(syndir, at + suffix)
                        input_tag = "syn"
                        output_asdf = os.path.join(procdir,
                                                   wave + "." + datatype + "."
                                                   + at + suffix)
                        output_tag = wave.capitalize() + "_" + at

                        # Define process parameter file
                        process_param_file = os.path.join(
                            self.process_param_dir,
                            wave + "." + datatype
                            + ".process.yml")

                        # Define path output dictionary
                        outdict = {
                            "input_asdf": input_asdf,
                            "input_tag": input_tag,
                            "output_asdf": output_asdf,
                            "output_tag": output_tag,
                            "process_param_file": process_param_file
                        }

                        # Find file name
                        filename = os.path.join(self.process_path_dir,
                                                wave + "." + datatype
                                                + "." + at
                                                + ".process_path.yml")

                        self.processpath_file_dict[wave + "_" + datatype
                                                   + "_" + at] \
                            = {"filename": filename,
                               "params": outdict}


    def create_window_path_file_struct(self):
        """This method creates the path files used to perform the windowing
        on the traces

        .. rubric:: Content of a path file

        For the body wave path file the content should look as follows, where
        the dots represent the location of the CMT directory in the database

        .. code-block:: yaml

            figure_dir: .../window_data/body_wave_plts
            figure_mode: true
            obsd_asdf: .../seismograms/processed_seismograms/body.obsd.h5
            obsd_tag: Body
            output_file: .../window_data/body.windows.json
            synt_asdf: .../seismograms/processed_seismograms/processed_synthetic_CMT.040_100.h5
            synt_tag: Body_CMT
            window_param_file: .../workflow_files/params/window_params/\
            body.window.yml

        .. rubric:: Location in database

        The dots represent the CMT directory in the database. For certain
        magnitudes, a mantle wave path file is generated as well.

        .. code-block:: bash

            .../workflow_files/path_files/window_paths/body.window_path.yml
            .../workflow_files/path_files/window_paths/surface.window_path.yml

        """

        windowdir = os.path.join(self.cmtdir, 'window_data')
        procdir = os.path.join(self.cmtdir, 'seismograms',
                               'processed_seismograms')
        suffix = ".h5"

        self.windowpath_file_dict = dict()

        for wave, datadict in self.windowdict.items():

            obsd_asdf = os.path.join(procdir, wave + ".obsd" + suffix)
            obsd_tag = wave.capitalize()
            synt_asdf = os.path.join(procdir, wave + ".synt" + suffix)
            synt_tag = wave.capitalize() + "_CMT"
            output_file = os.path.join(windowdir, wave + ".windows.json")
            figure_dir = os.path.join(windowdir, wave + "_wave_plots")
            window_param_file = os.path.join(self.window_param_dir,
                                             wave + ".window.yml")

            d = {"obsd_asdf": obsd_asdf,
                 "obsd_tag": obsd_tag,
                 "synt_asdf": synt_asdf,
                 "synt_tag": synt_tag,
                 "output_file": output_file,
                 "figure_mode": self.figure_mode,
                 "figure_dir": figure_dir,
                 "window_param_file": window_param_file}

            filename = os.path.join(self.window_path_dir,
                                    wave + ".window_path.yml")

            self.windowpath_file_dict[wave] \
                = {"filename": filename,
                   "params": d}

    def create_create_inversion_structs(self):
        """Creates inversion dictionaries and their respectice filenames.

        .. rubric:: Content

        The dots represent the CMT directory inside the database.

        .. code-block:: yaml

            asdf_dict:
              Mpp: .../seismograms/processed_seismograms/body.synt.CMT_pp.h5
              Mrp: .../seismograms/processed_seismograms/body.synt.CMT_rp.h5
              Mrr: .../seismograms/processed_seismograms/body.synt.CMT_rr.h5
              Mrt: .../seismograms/processed_seismograms/body.synt.CMT_rt.h5
              Mtp: .../seismograms/processed_seismograms/body.synt.CMT_tp.h5
              Mtt: .../seismograms/processed_seismograms/body.synt.CMT_tt.h5
              dep: .../seismograms/processed_seismograms/body.synt.CMT_depth.h5
              lat: .../seismograms/processed_seismograms/body.synt.CMT_lat.h5
              lon: .../seismograms/processed_seismograms/body.synt.CMT_lon.h5
              obsd: .../seismograms/processed_seismograms/body.obsd.h5
              synt: .../seismograms/processed_seismograms/body.synt.CMT.h5
            window_file: .../window_data/body.windows.json

        .. rubric:: Location in the database


        """

        seismodir = os.path.join(self.cmtdir, "seismograms",
                                 "processed_seismograms")
        windir = os.path.join(self.cmtdir, "window_data")

        self.cmt3d_invdicts = dict()
        for wave in self.windowdict.keys():

            asdf_dict = dict()
            asdf_dict['obsd'] = os.path.join(seismodir, wave
                                             + ".obsd.h5")
            asdf_dict['synt'] = os.path.join(seismodir, wave
                                             + ".synt.CMT.h5")

            for key in attr[1:]:
                cmt_pert = PARMAP[key]
                asdf_dict[cmt_pert] = os.path.join(seismodir, wave
                                              + ".synt." + key + ".h5")

            window_file = os.path.join(windir, wave + ".windows.json")

            paramdict = {"asdf_dict": asdf_dict,
                         "window_file": window_file}

            filename = os.path.join(self.invdir, "cmt3d." + wave
                                    + ".inv_dict.yml")

            self.cmt3d_invdicts[wave + "cmt3d"] = \
                {"filename": filename,
                 "params": paramdict}

    def create_create_gridsearch_structs(self):
        pass

    @staticmethod
    def write_param_file_dict(paramfiledict):
        for _type, typedict in paramfiledict.items():
            logger.verbose("Creating File: %s" % typedict["filename"])
            logger.debug("Parameters: %s" % typedict["params"])
            write_yaml_file(typedict["params"], typedict["filename"])