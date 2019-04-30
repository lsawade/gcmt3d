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


from gcmt3d.source import CMTSource
import glob, os, shutil
import warnings



class DataBaseSkeleton(object):
    """Class to handle data skeleton creation. If specfem directory is given,
    the class copies the necessary data from the specfem folder."""

    def __init__(self, basedir=None, cmt_fn=None, specfem_dir=None,
                 verbose=False, overwrite=False):
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

        # Check if things exists
        self.cmt_fn = cmt_fn
        self.specfem_dir = specfem_dir

        # Modifiers
        self.v = verbose
        self.ow = overwrite

    def create_all(self):
        """ Writes complete database structure."""

        # Create earthquake directory
        self.create_eq_dir()

        # Create
        self.create_CMT_SIM_dir()

    def create_base(self):
        """Creates Base directory if it doesn't exist."""

        self._create_dir(self.basedir)


    def create_eq_dir(self):
        """Creates response subdirectory"""

        # Check if cmt file name exists and if there are more than one
        cmtfile_list = glob.glob(self.cmt_fn)
        print(cmtfile_list)

        # Throw error if list empty
        if cmtfile_list is []:
            raise ValueError("No CMTSOLUTION file exists of that name.")

        # Create empty earthquake directory list
        self.eq_dirs = []
        self.eq_ids = []

        # Go through list of CMT solutions
        for cmtfile in cmtfile_list:

            # create CMT
            cmt = CMTSource.from_CMTSOLUTION_file(cmtfile)

            # Create CMTSource to extract the file name
            eq_id = cmt.eventname
            self.eq_ids.append(eq_id)

            # Earthquake directory
            eq_dir = os.path.join(self.basedir, "eq_" + eq_id)

            # Create directory
            self._create_dir(eq_dir)

            self.eq_dirs.append(eq_dir)

            # Create new CMT path
            cmt_path = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")

            # Copy the Earthquake file into the directory with eq_<ID>.cmt
            self._copy_cmt(cmtfile, cmt_path)

    def create_CMT_SIM_dir(self):
        """
        Creates CMT simulation directory and copies necessary files from given
        specfem directory. Important here is the fact that specfem has to be
        compiled already and there is no way of testing that prior to running
        GCMT.
        """

        # First create main directory


        # Throw error if specfem path
        # os.symlink(os.path.join(self.specfem_dir, "bin"))

    def create_response_dir(self):
        """Creates response subdirectory"""

        for _i, _eq_dir in enumerate(self.eq_dirs):

            # Create response path
            response_dir = os.path.join(_eq_dir, "responses")

            # Create new directory
            self._create_dir(response_dir)

    def create_seismogram_dir(self):
        """Creates response subdirectory"""

        for _i, _eq_dir in enumerate(self.eq_dirs):
            # Create response path
            seismogram_dir = os.path.join(_eq_dir, "seismograms")

            # Create new directory
            self._create_dir(seismogram_dir)

    def _copy_cmt(self, source, destination):
        """ Copies CMT solution from source to destination. It checks also
        for potential duplicates in the same place, warns whether they are
        different but have the name."""

        if os.path.exists(destination) and self.ow:
            if self.v:
                print("Earthquake file %s exists already. It will "
                      "be overwritten." % destination)
                self._replace_file(source, destination)

            elif os.path.exists(eq_dir) and self.ow == False:
                print("Earthquake file %s exists already. It will "
                      "NOT be overwritten." % destination)

            # Warn if destination eq is not equal to new eq
            if CMTSource.from_CMTSOLUTION_file(source) \
            is not CMTSource.from_CMTSOLUTION_file(destination):
                warnings.warn("CMT solution in the database is not "
                              "the same as the file with the same ID.")

        else:
            if self.v:
                print("Copying earthquake %s file to database." % destination)
        shutil.copy2(source, destination)

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
    def _replace_dir(directory):
        """Mini function that replaces a directory"""
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    @staticmethod
    def _replace_file(source, destination):
        """Mini function that replaces a directory"""
        if os.path.exists(destination) and os.path.isfile(destination):
            os.remove(destination)
        shutil.copy2(source, destination)

    def __str__(self):
        """This is what is printed when the initialized class is called in the
        print function. the function returns a string that contains the paths
        to the different directories specified."""

        string = "Database directory:\n"
        string += self.basedir + "\n\n"
        string +="CMTSOLUTION path:\n"
        string += self.cmt_fn + "\n\n"
        string +="Specfem directory:\n"
        string += self.specfem_dir + "\n\n"
        return string