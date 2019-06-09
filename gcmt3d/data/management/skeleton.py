"""

This script contains functions to create the skeleton structure for the GCMT3D
database of Earthquakes. If wanted.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2019

"""

from ...source import CMTSource
import glob
import os
import shutil
import warnings
from distutils.dir_util import copy_tree
from obspy import read_events


class DataBaseSkeleton(object):
    """Class to handle data skeleton creation. If specfem directory is given,
    the class copies the necessary data from the specfem folder."""

    def __init__(self, basedir=None, cmt_fn=None, specfem_dir=None,
                 verbose=False, overwrite=False, npar=9):
        """
        Args:
            basedir: str with path to database directory, e.g
                            ".../path/to/database/"
            cmt_fn: path to cmt solution, e.g ".../path/to/CMTSOLUTION", or,
                            e.g ".../path/to/cmtsolutionfile with list."
            specfem_dir: str with path to specfem directory, e.g
                            ".../path/to/specfem3d_globe"

        """

        self.basedir = basedir
        self.specfem_dir = specfem_dir
        self.npar = npar

        # Modifiers
        self.v = verbose
        self.ow = overwrite

        # Check if things exists
        self.cmt_fn = cmt_fn

        # Check if cmt file name exists and if there are more than one
        self.cmtfile_list = glob.glob(self.cmt_fn)

        # Throw error if list empty
        if self.cmtfile_list is []:
            raise ValueError("No CMTSOLUTION file exists of that name.")

        # Create empty earthquake directory list
        self.eq_dirs = []
        self.eq_ids = []

    def create_all(self):
        """ Writes complete database structure."""

        # Create earthquake directory
        self.create_eq_dirs()

        # Create station metadata directory
        self.create_station_dir()

        # Create window data directory
        self.create_window_dir()

        # Create Seismogram directory
        self.create_seismogram_dir()

        if self.specfem_dir is not None:
            # Create
            self.create_CMT_SIM_dir()

    def create_base(self):
        """Creates Base directory if it doesn't exist."""

        self._create_dir(self.basedir)

    def create_eq_dirs(self):
        """ If more than one earthquake exist with regex, call all of them"""

        for cmtfile in self.cmtfile_list:

            # Create directory
            self.create_1_eq_dir(cmtfile)

    def create_1_eq_dir(self, cmtfile):
        """Creates 1 Earthquake directory"""

        # create CMT
        cmt = CMTSource.from_CMTSOLUTION_file(cmtfile)

        # Create CMTSource to extract the file name
        eq_id = cmt.eventname
        self.eq_ids.append(eq_id)

        # Earthquake directory
        eq_dir = os.path.join(self.basedir, "eq_" + eq_id)

        # Create directory
        self._create_dir(eq_dir)

        # Append directory path to the list.
        self.eq_dirs.append(eq_dir)

        # Create new CMT path
        cmt_path = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")

        # Copy the Earthquake file into the directory with eq_<ID>.cmt
        self._copy_cmt(cmtfile, cmt_path)

    def create_station_dir(self):
        """Creates station_data directory for station metadata."""

        for _i, _eq_dir in enumerate(self.eq_dirs):

            # Create station_data dirs
            station_dir = os.path.join(_eq_dir, "station_data")

            # Create new directory
            self._create_dir(station_dir)

    def create_window_dir(self):
        """Creates window_data directory for pyflex window data metadata."""

        for _i, _eq_dir in enumerate(self.eq_dirs):

            # Create window_data dirs
            station_dir = os.path.join(_eq_dir, "window_data")

            # Create new directory
            self._create_dir(station_dir)

    def create_CMT_SIM_dir(self):
        """
        Creates CMT simulation directory and copies necessary files from given
        specfem directory. Important here is the fact that specfem has to be
        compiled already and there is no way of testing that prior to running
        GCMT.
        """
        # Parameters
        attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
                "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

        for _i, _eq in enumerate(self.eq_dirs):

            # First create main directory
            sim_path = os.path.join(_eq, "CMT_SIMs")
            self._create_dir(sim_path)

            # Second create subdirectories of CMT specfem directories
            for _j, _attr in enumerate(attr[:self.npar+1]):

                # Create subdirectory for simulation packages.
                cmt_der_path = os.path.join(sim_path, _attr)
                self._create_dir(cmt_der_path)

                # Copy specfem directory into cmt_der_path
                subdirs = ["DATA", "DATABASES_MPI", "OUTPUT_FILES"]

                for _k, _subdir in enumerate(subdirs):
                    # Path to specfem subdirectory
                    src_path = os.path.join(self.specfem_dir, _subdir)

                    # Path to destination directory
                    dst_path = os.path.join(cmt_der_path, _subdir)
                    self._copy_dir(src_path, dst_path)

                # Create symbolic link to destination folders
                if not os.path.islink((os.path.join(cmt_der_path, "bin"))):
                    os.symlink(os.path.join(self.specfem_dir, "bin"),
                               os.path.join(cmt_der_path, "bin"),
                               target_is_directory=True)

    def create_seismogram_dir(self):
        """Creates response subdirectory"""

        for _i, _eq_dir in enumerate(self.eq_dirs):
            # Create response path
            seismogram_dir = os.path.join(_eq_dir, "seismograms")

            # Create new directory
            self._create_dir(seismogram_dir)

    def _copy_dir(self, source, destination):
        """ Copies a directory source to destination. It checks also for
        potential duplicates in the same place."""

        if os.path.isdir(destination) and self.ow:
            if self.v:
                print("Directory %s exists already. It will "
                      "be overwritten." % destination)
            self._replace_dir(source, destination)

        elif os.path.isdir(destination) and self.ow is False:
            if self.v:
                print("Directory %s exists already. It will "
                      "NOT be overwritten." % destination)

        else:
            if self.v:
                print("Copying directory %s file to %s"
                      % (source, destination))
            copy_tree(source, destination)

    def _copy_cmt(self, source, destination):
        """ Copies CMT solution from source to destination. It checks also
        for potential duplicates in the same place, warns whether they are
        different but have the name."""

        if os.path.isfile(destination) and self.ow:
            if self.v:
                print("Earthquake file %s exists already. It will "
                      "be overwritten." % destination)
            self._replace_file(source, destination)

        elif os.path.isfile(destination) and self.ow is False:
            if self.v:
                print("Earthquake file %s exists already. It will "
                      "NOT be overwritten." % destination)

            # Warn if destination eq is not equal to new eq
            if not CMTSource.from_CMTSOLUTION_file(source) \
                    == CMTSource.from_CMTSOLUTION_file(destination):
                warnings.warn("CMT solution in the database is not "
                              "the same as the file with the same ID.")

        else:
            if self.v:
                print("Copying earthquake %s file to %s." % (source,
                                                             destination))
            shutil.copyfile(source, destination)

    def _write_quakeml(self, source, destination):
        """ Copies CMT solution from source to QuakeML destination. It checks
        also for potential duplicates in the same place, warns whether they are
        different but have the name."""

        # CMT Source file
        catalog = read_events(source)

        if os.path.isfile(destination) and self.ow:
            if self.v:
                print("Earthquake file %s exists already. It will "
                      "be overwritten." % destination)
            os.remove(destination)
            catalog.write(destination, format="QUAKEML")

        elif os.path.isfile(destination) and self.ow is False:
            if self.v:
                print("Earthquake file %s exists already. It will "
                      "NOT be overwritten." % destination)

            # Warn if destination eq is not equal to new eq
            if not CMTSource.from_CMTSOLUTION_file(source) \
                    == CMTSource.from_quakeml_file(destination):
                warnings.warn("CMT solution in the database is not "
                              "the same as the file with the same ID.")
        else:
            if self.v:
                print("Copying earthquake %s file to %s." % (source,
                                                             destination))
            catalog.write(destination, format="QUAKEML")

    def _create_dir(self, directory):
        """Create subdirectory"""
        if os.path.exists(directory) and os.path.isdir(directory) \
                and self.ow is False:
            if self.v:
                print("%s exists already. Not overwritten." % directory)

        elif os.path.exists(directory) and os.path.isdir(directory) \
                and self.ow is True:
            if self.v:
                print(
                    "%s exists already, but overwritten." % directory)
                self._replace_dir(directory)
        else:
            if self.v:
                print("Creating directory %s." % directory)
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
