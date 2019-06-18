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
from ...asdf.utils import write_yaml_file
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
                 npar=9, verbose=False, overwrite=False):
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

        # List of possible attributes
        # Parameters
        self.attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
                     "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

    def create_all(self):
        """ Writes complete database structure."""

        if self.v:
            print("Creating earthquake entry in database %s ... " %
                  self.basedir)

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

        if self.v:
            print("Done.")

    def create_base(self):
        """Creates Base directory if it doesn't exist."""

        # Overwrite 0 overwrite total database
        if self.ow in [0] and type(self.ow) is not bool:
            self._create_dir(self.basedir, True)
        else:
            self._create_dir(self.basedir, False)

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
        if self.ow in [0, 1] and type(self.ow) is not bool:
            self._create_dir(eq_dir, True)
        else:
            self._create_dir(eq_dir, False)

        # Append directory path to the list.
        self.eq_dirs.append(eq_dir)

        # Create new CMT path
        cmt_path = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")

        # Create new CMT path
        xml_path = os.path.join(eq_dir, "eq_" + eq_id + ".xml")

        # Copy the Earthquake file into the directory with eq_<ID>.cmt
        if self.ow in [0, 1] and type(self.ow) is not bool:
            self._copy_file(cmtfile, cmt_path, True)
        else:
            self._copy_file(cmtfile, cmt_path, False)

        # Copy the Earthquake file into the directory with eq_<ID>.xml
        if self.ow in [0, 1] and type(self.ow) is not bool:
            self._write_quakeml(cmtfile, xml_path, True)
        else:
            self._write_quakeml(cmtfile, xml_path, False)

    def create_station_dir(self):
        """Creates station_data directory for station metadata."""

        for _i, _eq_dir in enumerate(self.eq_dirs):

            # Create station_data dirs
            station_dir = os.path.join(_eq_dir, "station_data")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(station_dir, True)
            else:
                self._create_dir(station_dir, False)

    def create_inversion_output_dir(self):
        """Creates station_data directory for station metadata."""

        for _i, _eq_dir in enumerate(self.eq_dirs):

            # Create station_data dirs
            inv_dir = os.path.join(_eq_dir, "inversion_output")

            if self.ow in [0, 1, 2] and type(self.ow) is not bool:
                # Create new directory
                self._create_dir(inv_dir, True)
            else:
                self._create_dir(inv_dir, False)

    def create_window_dir(self):
        """Creates window_data directory for pyflex window data metadata."""

        for _i, _eq_dir in enumerate(self.eq_dirs):

            # Create window_data dirs
            window_dir = os.path.join(_eq_dir, "window_data")

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

        for _i, _eq in enumerate(self.eq_dirs):

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
                subdirs = ["DATA", "DATABASES_MPI", "OUTPUT_FILES"]

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
                        files = ["STATIONS", "Par_file", "CMTSOLUTION"]

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
                    else:
                        if self.ow in [0, 1, 2, 3] and type(
                                self.ow) is not bool:
                            self._copy_dir(src_path, dst_path, True)
                        else:
                            self._copy_dir(src_path, dst_path, False)

                    # Create the Path file for later usage of the ASDF
                    # conversion
                    if _subdir == "OUTPUT_FILES":
                        if self.v:
                            print("Writing YAML path file from waveform dir "
                                  "%s" % dst_path)
                        self._create_syn_path_yaml(dst_path)

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

            # write Observed path file
            if self.v:
                print("Writing the YAML path file to %s" %
                      os.path.join(obs_dir_path, "observed.yml"))
            self._create_obs_path_yaml(self.eq_ids[_i], _eq_dir)

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
            if self.v:
                print("Directory %s exists already. It will "
                      "be overwritten." % destination)
            shutil.rmtree(destination)
            copy_tree(source, destination, **kwargs)

        elif os.path.isdir(destination) and ow is False:
            if self.v:
                print("Directory %s exists already. It will "
                      "NOT be overwritten." % destination)
        else:
            if self.v:
                print("Copying directory %s file to %s"
                      % (source, destination))
            copy_tree(source, destination)

    def _copy_file(self, source, destination, ow):
        """ Copies file from source to destination. """

        if os.path.isfile(destination) and ow:
            if self.v:
                print("File %s exists already. It will "
                      "be overwritten." % destination)
            self._replace_file(source, destination)

        elif os.path.isfile(destination) and ow is False:
            if self.v:
                print("File %s exists already. It will "
                      "NOT be overwritten." % destination)

        else:
            if self.v:
                print("Copying file %s file to %s." % (source, destination))
            shutil.copyfile(source, destination)

    def _write_quakeml(self, source, destination, ow):
        """ Copies CMTSOLUTION from source to QuakeML destination. It checks
        also for potential duplicates in the same place, warns whether they are
        different but have the name."""

        # CMT Source file
        catalog = read_events(source)

        if os.path.isfile(destination) and ow:
            if self.v:
                print("Earthquake file %s exists already. It will "
                      "be overwritten." % destination)
            os.remove(destination)
            catalog.write(destination, format="QUAKEML")

        elif os.path.isfile(destination) and ow is False:
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
                print("Writing earthquake %s file to %s." % (source,
                                                             destination))
            catalog.write(destination, format="QUAKEML")

    def _create_syn_path_yaml(self, waveform_dir):
        """ This function writes a yaml conversion path file for 1 Simulation
        file. This file is later on need for the creation of ASDF files and the
        processing involved ASDF files.

        The function assumes that
        * the QuakeML file is located in the OUTPUT_FILES directory with the
          name `Quake.xml`
        * The output directory name is the
          `../database/eq_<id>/seismograms/syn/<attr>.h5

        Args:
              waveform_dir: path to OUTPUT_FILES

        """

        # File Type
        filetype = "sac"

        # Tag
        tag = "raw_synthetic"

        # QuakeML file path
        quakeml_file = os.path.join(waveform_dir, "Quake.xml")

        # Outputfile
        cmt_sim_dir = os.path.dirname(waveform_dir)
        eq_dir = os.path.dirname(os.path.dirname(cmt_sim_dir))
        cmt_name = os.path.basename(cmt_sim_dir)
        output_file = os.path.join(eq_dir, "seismograms", "syn",
                                   cmt_name + ".h5")

        # Pathfile directory
        yaml_file_path = os.path.join(cmt_sim_dir, cmt_name + ".yml")

        # Create dictionary
        if self.v:
            print("Writing path file %s." % yaml_file_path)

        d = {"waveform_dir": waveform_dir,
             "filetype": filetype,
             "quakeml_file": quakeml_file,
             "tag": tag,
             "output_file": output_file}

        # Writing the directory to file
        write_yaml_file(d, yaml_file_path)

    def _create_obs_path_yaml(self, eq_id, eq_dir):
        """ This function writes a yaml path file for 1 Simulation file. This
        file is later on need for the creation of ASDF files and the
        processing involved ASDF files.

        The function assumes that
        * the QuakeML file is located in the main EQ directory with the
          name `eq_<id>.xml`
        * The output file name is the
          `../database/eq_<id>/seismograms/obs/raw_observed.h5

        Args:
              waveform_dir: path to OUTPUT_FILES

        """

        # Tag
        tag = "raw_observed"

        # Waveform file
        waveform_files = os.path.join(eq_dir, "seismograms", "obs",
                                      eq_id + ".mseed")

        # QuakeML file path
        quakeml_file = os.path.join(eq_dir, "eq_" + eq_id + ".xml")

        # Station file
        staxml_file = os.path.join(eq_dir, "station_data", "station.xml")

        # Outputfile
        output_file = os.path.join(eq_dir, "seismograms",
                                   "obs", "raw_observed.h5")

        # Pathfile directory
        yaml_file_path = os.path.join(eq_dir, "seismograms",
                                      "obs", "observed.yml")

        # Create dictionary
        if self.v:
            print("Writing path file %s." % yaml_file_path)

        d = {"waveform_files": waveform_files,
             "quakeml_file": quakeml_file,
             "tag": tag,
             "staxml_files": staxml_file,
             "output_file": output_file}

        # Writing the directory to file
        write_yaml_file(d, yaml_file_path)

    def _create_dir(self, directory, ow):
        """Create subdirectory"""
        if os.path.exists(directory) and os.path.isdir(directory) \
                and ow is False:
            if self.v:
                print("%s exists already. Not overwritten." % directory)

        elif os.path.exists(directory) and os.path.isdir(directory) \
                and ow is True:
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
