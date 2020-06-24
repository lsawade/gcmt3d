"""

This script contains functions to create the skeleton structure for the GCMT3D
database of Earthquakes. If wanted.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.orcopyleft/gpl.html)

Last Update: June 2019

"""

import glob
import os
import shutil
import warnings
from distutils.dir_util import copy_tree
from obspy import read_events
from time import time
from ...source import CMTSource

import logging
from ...log_util import modify_logger

# Create logger
logger = logging.getLogger(__name__)
modify_logger(logger)

STATIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STATIONS")


class DataBaseSkeleton(object):
    """Class to handle data skeleton creation. If specfem directory is given,
    the class copies the necessary data from the specfem folder."""

    def __init__(self, basedir=None, cmt_fn=None, specfem_dir=None,
                 stations_file=None, npar=9,
                 overwrite=False):
        """
        Args:
            basedir: str with path to database directory, e.g
                            ".../path/to/database/"
            cmt_fn: path to cmt solution, e.g ".../path/to/CMTSOLUTION", or,
                            e.g ".../path/to/cmtsolutionfile with list."
            npar: number of parameters to be inverted for. 6,7 or 9. Default
                  9
            specfem_dir: str with path to specfem directory, e.g
                            ".../path/to/specfem3d_globe"
            overwrite: number which sets what should be over written. If not
                       a number or False, nothing will be overwritten.
                        0: Everything is overwritten, including the database
                        1: The specific earthquake is overwritten:
                        2: Seismogram, Simulation and Window Directories are
                           overwritten for a certain Earthquake.
                        3: Simulation subdirectories overwritten
                        4: Overwrite yaml path files only
            verbose: boolean that sets whether the output should be verbose
                     or not

        """

        self.basedir = basedir
        self.specfem_dir = specfem_dir
        self.npar = npar

        # Modifiers
        self.ow = overwrite

        # Check if things exists
        self.cmt_fn = cmt_fn

        # Check if cmt file name exists and if there are more than one
        self.cmtfile_list = glob.glob(self.cmt_fn)

        # Throw error if list empty
        if self.cmtfile_list is []:
            raise ValueError("No CMTSOLUTION file exists of that name.")

        # Create empty earthquake directory list
        self.Cdirs = []
        self.Cids = []

        # List of possible attributes
        # Parameters
        self.attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
                     "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

        if stations_file is None:
            self.stations_file = STATIONS
        else:
            self.stations_file = stations_file

    def create_all(self):
        """ Writes complete database structure."""

        logger.info("Creating earthquake entry in database %s ... "
                    % self.basedir)
        t1 = time()

        # Create earthquake directory
        logger.info("Creating simulation directories...")
        self.create_Cdirs()

        # Create station metadata directory
        logger.info("Creating station directory...")
        self.create_station_dir()

        # Create Log directory
        logger.info("Creating log directory...")
        self.create_log_dir()

        # Create window data directory
        logger.info("Creating window...")
        self.create_window_dir()

        # Create Seismogram directory
        logger.info("Creating seismogram directories")
        self.create_seismogram_dir()

        # Creating the workflow file dir
        logger.info("Creating workflow directories")
        self.create_workflow_dir()

        # Create Inversion directory
        logger.info("Creating inversion directories")
        self.create_inversion_dir()

        if self.specfem_dir is not None:
            # Create
            self.create_CMT_SIM_dir()

        logger.info("Done. Time Elapsed %.1f s" % (time() - t1))

    def create_base(self):
        """Creates Base directory if it doesn't exist."""

        # Overwrite 0 overwrite total database
        if self.ow in [0] and type(self.ow) is not bool:
            self._create_dir(self.basedir, True)
        else:
            self._create_dir(self.basedir, False)

    def create_Cdirs(self):
        """ If more than one earthquake exist with regex, call all of them"""
        for cmtfile in self.cmtfile_list:

            # Create directory
            self.create_1_Cdir(cmtfile)

    def create_1_Cdir(self, cmtfile):
        """Creates 1 Earthquake directory"""

        # create CMT
        cmt = CMTSource.from_CMTSOLUTION_file(cmtfile)

        # Create CMTSource to extract the file name
        Cid = cmt.eventname

        self.Cids.append(Cid)

        # Earthquake directory
        Cdir = os.path.join(self.basedir, Cid)

        # Create directory
        if self.ow in [0, 1] and type(self.ow) is not bool:
            self._create_dir(Cdir, True)
        else:
            self._create_dir(Cdir, False)

        # Append directory path to the list.
        self.Cdirs.append(Cdir)

        # Create new CMT path
        cmt_path = os.path.join(Cdir, Cid + ".cmt")

        # Create new CMT path
        xml_path = os.path.join(Cdir, Cid + ".xml")

        # Copy the Earthquake file into the directory with C<ID>.cmt
        if self.ow in [0, 1] and type(self.ow) is not bool:
            self._copy_file(cmtfile, cmt_path, True)
        else:
            self._copy_file(cmtfile, cmt_path, False)

        # Copy the Earthquake file into the directory with C<ID>.xml
        if self.ow in [0, 1] and type(self.ow) is not bool:
            self._write_quakeml(cmtfile, xml_path, True)
        else:
            self._write_quakeml(cmtfile, xml_path, False)

    def create_station_dir(self):
        """Creates station_data directory for station metadata."""

        for _i, _Cdir in enumerate(self.Cdirs):

            # Create station_data dirs
            station_dir = os.path.join(_Cdir, "station_data")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(station_dir, True)
            else:
                self._create_dir(station_dir, False)

            # Set new destination path
            dst_path0 = os.path.join(station_dir, "STATIONS")

            if self.ow in [0, 1, 2, 3] \
                    and type(self.ow) is not bool:
                self._copy_file(self.stations_file, dst_path0, True)
            else:
                self._copy_file(self.stations_file, dst_path0, False)

    def create_log_dir(self):
        """Creates station_data directory for station metadata."""

        for _i, _Cdir in enumerate(self.Cdirs):

            # Create station_data dirs
            log_dir = os.path.join(_Cdir, "logs")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(log_dir, True)
            else:
                self._create_dir(log_dir, False)

    def create_workflow_dir(self):
        """Creates station_data directory for station metadata."""

        for _i, _Cdir in enumerate(self.Cdirs):

            # Create station_data dirs
            wf_dir = os.path.join(_Cdir, "workflow_files")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(wf_dir, True)
            else:
                self._create_dir(wf_dir, False)

            pathfiles = os.path.join(wf_dir, "path_files")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(pathfiles, True)
            else:
                self._create_dir(pathfiles, False)

            convpaths = os.path.join(pathfiles, "conversion_paths")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(convpaths, True)
            else:
                self._create_dir(convpaths, False)

            windowpaths = os.path.join(pathfiles, "window_paths")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(windowpaths, True)
            else:
                self._create_dir(windowpaths, False)

            processpaths = os.path.join(pathfiles, "process_paths")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(processpaths, True)
            else:
                self._create_dir(processpaths, False)

            paramfiles = os.path.join(wf_dir, "params")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(paramfiles, True)
            else:
                self._create_dir(pathfiles, False)

            windowparams = os.path.join(paramfiles, "window_params")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(windowparams, True)
            else:
                self._create_dir(windowparams, False)

            processparams = os.path.join(paramfiles, "process_params")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(processparams, True)
            else:
                self._create_dir(processparams, False)

            inversionparams = os.path.join(paramfiles, "inversion_params")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(inversionparams, True)
            else:
                self._create_dir(inversionparams, False)

            invfiles = os.path.join(wf_dir, "inversion_dicts")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(invfiles, True)
            else:
                self._create_dir(invfiles, False)

    def create_inversion_dir(self):
        """Creates station_data directory for station metadata."""

        for _i, _Cdir in enumerate(self.Cdirs):

            # Create station_data dirs
            inv_out_dir = os.path.join(_Cdir, "inversion")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(inv_out_dir, True)
            else:
                self._create_dir(inv_out_dir, False)

            inv_types = ["cmt3d", "g3d"]

            for _inv_type in inv_types:

                type_dir = os.path.join(inv_out_dir, _inv_type)

                if self.ow in [0, 1, 2, 3, 4] and type(self.ow) is not bool:
                    # Create new directory
                    self._create_dir(type_dir, True)
                else:
                    self._create_dir(type_dir, False)

                inv_new_syn_dir = os.path.join(type_dir, "new_synt")

                if self.ow in [0, 1, 2, 3, 4] and type(self.ow) is not bool:
                    # Create new directory
                    self._create_dir(inv_new_syn_dir, True)
                else:
                    self._create_dir(inv_new_syn_dir, False)

                inv_new_syn_plots_dir = os.path.join(type_dir,
                                                     "waveform_plots")

                if self.ow in [0, 1, 2, 3, 4] and type(self.ow) is not bool:
                    # Create new directory
                    self._create_dir(inv_new_syn_plots_dir, True)
                else:
                    self._create_dir(inv_new_syn_plots_dir, False)

    def create_window_dir(self):
        """Creates window_data directory for pyflex window data metadata."""

        for _i, _Cdir in enumerate(self.Cdirs):

            # Create window_data dirs
            window_dir = os.path.join(_Cdir, "window_data")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(window_dir, True)
            else:
                self._create_dir(window_dir, False)

            # Create Window path directory
            window_path_dir = os.path.join(window_dir, "window_paths")

            if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(window_path_dir, True)
            else:
                self._create_dir(window_path_dir, False)

    def create_CMT_SIM_dir(self):
        """
        Creates CMT simulation directory and copies necessary files from given
        specfem directory. Important here is the fact that specfem has to be
        compiled already and there is no way of testing that prior to running
        GCMT.
        """

        for _i, _eq in enumerate(self.Cdirs):

            # First create main directory
            sim_path = os.path.join(_eq, "CMT_SIMs")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                self._create_dir(sim_path, True)
            else:
                self._create_dir(sim_path, False)

            # Second create subdirectories of CMT specfem directories
            for _j, _attr in enumerate(self.attr[:self.npar+1]):

                # Create subdirectory for simulation packages.
                cmt_der_path = os.path.join(sim_path, _attr)

                if self.ow in [0, 1, 2, 3] and type(self.ow) is not bool:
                    self._create_dir(cmt_der_path, True)
                else:
                    self._create_dir(cmt_der_path, False)

                # Copy specfem directory into cmt_der_path
                subdirs = ["DATA", "OUTPUT_FILES"]

                for _k, _subdir in enumerate(subdirs):
                    # Path to specfem subdirectory
                    src_path = os.path.join(self.specfem_dir, _subdir)

                    # Path to destination directory
                    dst_path = os.path.join(cmt_der_path, _subdir)

                    # Only copy some files from the DATA directory
                    if _subdir == "DATA":

                        # Create DATA directory
                        if self.ow in [0, 1, 2, 3] and type(
                                self.ow) is not bool:
                            self._create_dir(dst_path, True)
                        else:
                            self._create_dir(dst_path, False)

                        # Set files to be copied into directory
                        files = ["Par_file", "CMTSOLUTION"]

                        for file in files:

                            # Set new source path
                            src_path0 = os.path.join(src_path, file)

                            # Set new destination path
                            dst_path0 = os.path.join(dst_path, file)

                            if self.ow in [0, 1, 2, 3] and type(
                                    self.ow) is not bool:
                                self._copy_file(src_path0, dst_path0, True)
                            else:
                                self._copy_file(src_path0, dst_path0, False)

                        # Set new destination path
                        dst_path0 = os.path.join(dst_path, "STATIONS")

                        if self.ow in [0, 1, 2, 3] \
                                and type(self.ow) is not bool:
                            self._copy_file(self.stations_file,
                                            dst_path0, True)
                        else:
                            self._copy_file(self.stations_file,
                                            dst_path0, False)

                    else:
                        if self.ow in [0, 1, 2, 3] and type(
                                self.ow) is not bool:
                            self._copy_dir(src_path, dst_path, True)
                        else:
                            self._copy_dir(src_path, dst_path, False)

                # Create symbolic link to destination folders
                if not os.path.islink((os.path.join(cmt_der_path, "bin"))):
                    os.symlink(os.path.join(self.specfem_dir, "bin"),
                               os.path.join(cmt_der_path, "bin"),
                               target_is_directory=True)

                # Create symbolic link to destination folders
                if not os.path.islink((os.path.join(cmt_der_path,
                                                    "DATABASES_MPI"))):
                    os.symlink(os.path.join(self.specfem_dir,
                                            "DATABASES_MPI"),
                               os.path.join(cmt_der_path,
                                            "DATABASES_MPI"),
                               target_is_directory=True)

    def create_seismogram_dir(self):
        """Creates response subdirectory"""

        for _i, _Cdir in enumerate(self.Cdirs):
            # Create response path
            seismogram_dir = os.path.join(_Cdir, "seismograms")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(seismogram_dir, True)
            else:
                self._create_dir(seismogram_dir, False)

            # Create the subdirectory for synthetic data
            syn_dir_path = os.path.join(seismogram_dir, 'syn')

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(syn_dir_path, True)
            else:
                self._create_dir(syn_dir_path, False)

            # Create the subdirectory for observed data
            obs_dir_path = os.path.join(seismogram_dir, 'obs')

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(obs_dir_path, True)
            else:
                self._create_dir(obs_dir_path, False)

            # Create the subdirectory for processing pathfiles
            process_dir_path = os.path.join(seismogram_dir, 'process_paths')

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(process_dir_path, True)
            else:
                self._create_dir(process_dir_path, False)

            # Create the subdirectory for processed seismograms
            processed_dir_path = os.path.join(seismogram_dir,
                                              'processed_seismograms')

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(processed_dir_path, True)
            else:
                self._create_dir(processed_dir_path, False)

    def _copy_dir(self, source, destination, ow, **kwargs):
        """ Copies a directory source to destination. It checks also for
        potential duplicates in the same place."""

        if os.path.isdir(destination) and ow:
            logger.verbose("Directory %s exists already. It will "
                           "be overwritten." % destination)
            shutil.rmtree(destination)
            copy_tree(source, destination, **kwargs)

        elif os.path.isdir(destination) and ow is False:
            logger.verbose("Directory %s exists already. It will "
                           "NOT be overwritten." % destination)
        else:
            logger.verbose("Copying directory %s file to %s"
                           % (source, destination))
            copy_tree(source, destination)

    def _copy_file(self, source, destination, ow):
        """ Copies file from source to destination. """

        if os.path.isfile(destination) and ow:
            logger.verbose("File %s exists already. It will "
                           "be overwritten." % destination)
            self._replace_file(source, destination)

        elif os.path.isfile(destination) and ow is False:
            logger.verbose("File %s exists already. It will "
                           "NOT be overwritten." % destination)
        else:
            logger.verbose("Copying file %s file to %s."
                           % (source, destination))
            shutil.copyfile(source, destination)

    def _write_quakeml(self, source, destination, ow):
        """ Copies CMTSOLUTION from source to QuakeML destination. It checks
        also for potential duplicates in the same place, warns whether they are
        different but have the name."""

        # CMT Source file
        catalog = read_events(source)

        if os.path.isfile(destination) and ow:
            logger.verbose("Earthquake file %s exists already. It will "
                           "be overwritten." % destination)
            os.remove(destination)
            catalog.write(destination, format="QUAKEML")

        elif os.path.isfile(destination) and ow is False:
            logger.verbose("Earthquake file %s exists already. It will "
                           "NOT be overwritten." % destination)

            # Warn if destination eq is not equal to new eq
            if not CMTSource.from_CMTSOLUTION_file(source) \
                    == CMTSource.from_quakeml_file(destination):
                warnings.warn("CMT solution in the database is not "
                              "the same as the file with the same ID.")
        else:
            logger.verbose("Writing earthquake %s file to %s." % (source,
                                                                  destination))
            catalog.write(destination, format="QUAKEML")

    def _create_dir(self, directory, ow):
        """Create subdirectory"""
        if os.path.exists(directory) and os.path.isdir(directory) \
                and ow is False:
            logger.verbose("%s exists already. Not overwritten." % directory)

        elif os.path.exists(directory) and os.path.isdir(directory) \
                and ow is True:
            logger.verbose("%s exists already, but overwritten." % directory)
            self._replace_dir(directory)
        else:
            logger.verbose("Creating directory %s." % directory)
            os.makedirs(directory)

    @staticmethod
    def _replace_dir(destination, source=None):
        """Mini function that replaces a directory"""
        if os.path.exists(destination) and os.path.isdir(destination):
            shutil.rmtree(destination)
        if source is None:
            os.makedirs(destination)

    @staticmethod
    def _replace_file(source, destination):
        """Mini function that replaces a file"""
        if os.path.exists(destination) and os.path.isfile(destination):
            os.remove(destination)
        shutil.copyfile(source, destination)

    def __str__(self):
        """This is what is printed when the initialized class is called in the
        print function. the function returns a string that contains the paths
        to the different directories specified."""

        string = "Database directory:\n"
        string += self.basedir + "\n\n"
        string += "CMTSOLUTION path:\n"
        string += self.cmt_fn + "\n\n"
        string += "Specfem directory:\n"
        string += self.specfem_dir + "\n\n"
        return string
