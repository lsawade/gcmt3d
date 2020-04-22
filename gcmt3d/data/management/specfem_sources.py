'''

This script/set of functions is used to create a set of sources for parallel
specfem simulations

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: June 2019

'''

import os
import shutil
import warnings
import logging
from copy import deepcopy
from obspy import read_events
from ...source import CMTSource
from ...log_util import modify_logger

logger = logging.getLogger(__name__)
modify_logger(logger)


def replace_file(source, destination):
    """Mini function that replaces a file"""
    if os.path.exists(destination) and os.path.isfile(destination):
        os.remove(destination)
    shutil.copyfile(source, destination)


class SpecfemSources(object):
    '''
    This class handles the writing of specfem sources in form of CMT solutions
    and the stations
    '''

    def __init__(self, cmt, cmt_dir, npar, dm=10.0*24, dx=2., ddeg=0.02,
                 hdur0=False, outdir=None, verbose=False):
        '''
        :param cmt: The original CMT source loaded using CMTSource
        :type cmt: CMTSource
        :param cmt: Directory in which the cmt solution resides within the
                    database
        :type cmt: str
        :param npar: Number of parameters to be inverted for: 6 - only moment
                     tensor, 7 - moment tensor and depth, 9 moment tensor,
                     depth and geolocation.
        :type npar: int
        :param dm: magnitude scalar -- we only need a scalar since the
                   derivative is independent of magnitude
        :type dm: float
        :param dx: depth change constant in m for Frechet derivative
        :type dx: float
        :param ddeg: location change constant for Frechet derivative
        :type ddeg: float
        :param dtshift: change in cmt time shift for Frechet derivative
        :type dtshift: float
        :param hdur0: Sets the half duration to 0 if True.
                      Default False.
        :type hdur0: float
        :param outdir: output directory for sources Should be CMT_SIMs
                       directory as created by the Skeleton class.
        :type outdir: str

        '''

        if type(cmt) != CMTSource:
            raise ValueError('Given CMT parameter not a CMTSource.')
        self.cmt = cmt

        if npar not in [6, 7, 9]:
            raise ValueError('The parameters to be inverted for must be an '
                             + 'integer between 6 and 9.')
        self.npar = npar

        if type(dm) != float:
            raise ValueError('Change in magnitude needs to be a float.')
        self.dm = dm

        if type(dx) != float:
            raise ValueError("Change in depth should be a float")
        self.dx = dx

        if type(ddeg) != float:
            raise ValueError("Change in degrees should be a float")
        self.ddeg = ddeg

        if type(verbose) != bool:
            raise ValueError("Verbose flag must be boolean")
        self.v = verbose

        if outdir is None:
            raise ValueError("The output directory needs wo be set.")
        elif not os.path.exists(outdir):
            os.makedirs(outdir)
            warnings.warn("The chosen output directory does not exist.\n"
                          + "A new one will be created.")

        self.station_dir = os.path.join(cmt_dir, 'station_data')
        self.outdir = outdir

        self.hdur0 = hdur0

    def write_sources(self):
        """Function to write the CMT Solutions to CMT files
        """

        # Write initial CMT solution
        cmtsim_outdir = os.path.join(self.outdir, "CMT",
                                     "DATA")

        # Create new CMT solution
        new_cmt = deepcopy(self.cmt)

        # Set half duration to 0
        if self.hdur0:
            new_cmt.half_duration = 0

        # write file
        new_cmt.write_CMTSOLUTION_file(os.path.join(cmtsim_outdir,
                                                    "CMTSOLUTION"))

        # Print info if verbose flag is True
        logger.verbose("%s has been written." % os.path.join(cmtsim_outdir,
                                                             "CMTSOLUTION"))

        # Write the QuakeML to OUTPUT_FILES folder
        src = os.path.join(cmtsim_outdir, "CMTSOLUTION")
        outfiles = os.path.join(self.outdir, "CMT", "OUTPUT_FILES")
        dst = os.path.join(outfiles, "Quake.xml")

        # Write Solution
        self._write_quakeml(src, dst)

        # Print info if verbose flag is True
        logger.verbose("%s has been written." % dst)

        # Replace STATIONS FILE
        station_source = os.path.join(self.station_dir, "STATIONS")
        replace_file(station_source, os.path.join(cmtsim_outdir,
                                                  "STATIONS"))

        # Attribute list
        attr = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

        for index in range(6):

            cmtsim_outdir = os.path.join(self.outdir, "CMT_"
                                         + attr[index][-2:],
                                         "DATA")

            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # make everything zero
            for attribute in attr:
                setattr(new_cmt, attribute, 0)

            # Set half duration to 0
            if self.hdur0:
                new_cmt.half_duration = 0

            # except one variable
            setattr(new_cmt, attr[index], self.dm)

            # write file
            new_cmt.write_CMTSOLUTION_file(os.path.join(cmtsim_outdir,
                                                        "CMTSOLUTION"))

            # Print info if verbose flag is True
            logger.verbose("%s has been written."
                           % os.path.join(cmtsim_outdir, "CMTSOLUTION"))

            # Write the QuakeML to OUTPUT_FILES folder
            src = os.path.join(cmtsim_outdir, "CMTSOLUTION")
            outfiles = os.path.join(self.outdir, "CMT_" + attr[index][-2:],
                                    "OUTPUT_FILES")
            dst = os.path.join(outfiles, "Quake.xml")

            self._write_quakeml(src, dst)

            # Print info if verbose flag is True
            logger.verbose("%s has been written." % dst)

            # Replace STATIONS FILE
            station_source = os.path.join(self.station_dir, "STATIONS")
            replace_file(station_source, os.path.join(cmtsim_outdir,
                                                      "STATIONS"))

        if self.npar > 6:

            # Attribute name
            depth_str = "depth_in_m"

            # Set output dir
            cmtsim_outdir = os.path.join(self.outdir, "CMT_depth", "DATA")

            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # change a depth
            setattr(new_cmt, depth_str, new_cmt.depth_in_m + self.dx)

            # Set half duration to 0
            if self.hdur0:
                new_cmt.half_duration = 0

            # write new solution
            new_cmt.write_CMTSOLUTION_file(os.path.join(cmtsim_outdir,
                                                        "CMTSOLUTION"))

            # Print info if verbose flag is True
            logger.verbose("%s has been written."
                           % os.path.join(cmtsim_outdir, "CMTSOLUTION"))

            # Write the QuakeML to OUTPUT_FILES folder
            src = os.path.join(cmtsim_outdir, "CMTSOLUTION")
            outfiles = os.path.join(self.outdir, "CMT_depth",
                                    "OUTPUT_FILES")
            dst = os.path.join(outfiles, "Quake.xml")

            self._write_quakeml(src, dst)

            # Print info if verbose flag is True
            logger.verbose("%s has been written." % dst)

            # Replace STATIONS FILE
            station_source = os.path.join(self.station_dir, "STATIONS")
            replace_file(station_source, os.path.join(cmtsim_outdir,
                                                      "STATIONS"))

        if self.npar == 9:

            # Attribute name
            lat_str = "latitude"
            lon_str = "longitude"

            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # change a depth
            setattr(new_cmt, lat_str, new_cmt.latitude + self.ddeg)

            # Set half duration to 0
            if self.hdur0:
                new_cmt.half_duration = 0

            # Set outdir
            cmtsim_outdir = os.path.join(self.outdir, "CMT_lat", "DATA")

            # write new solution
            new_cmt.write_CMTSOLUTION_file(os.path.join(cmtsim_outdir,
                                                        "CMTSOLUTION"))

            # Print info if verbose flag is True
            logger.verbose("%s has been written."
                           % os.path.join(cmtsim_outdir, "CMTSOLUTION"))

            # Write the QuakeML to OUTPUT_FILES folder
            src = os.path.join(cmtsim_outdir, "CMTSOLUTION")
            outfiles = os.path.join(self.outdir, "CMT_lat",
                                    "OUTPUT_FILES")
            dst = os.path.join(outfiles, "Quake.xml")

            self._write_quakeml(src, dst)

            # Replace STATIONS FILE
            station_source = os.path.join(self.station_dir, "STATIONS")
            replace_file(station_source, os.path.join(cmtsim_outdir,
                                                      "STATIONS"))

            # Print info if verbose flag is True
            logger.verbose("%s has been written." % dst)

            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # change a depth
            setattr(new_cmt, lon_str, new_cmt.longitude + self.ddeg)

            # Set half duration to 0
            new_cmt.half_duration = 0

            # Set outdir
            cmtsim_outdir = os.path.join(self.outdir, "CMT_lon", "DATA")

            # write new solution
            new_cmt.write_CMTSOLUTION_file(os.path.join(cmtsim_outdir,
                                                        "CMTSOLUTION"))

            # Print info if verbose flag is True
            logger.verbose("%s has been written."
                           % os.path.join(cmtsim_outdir, "CMTSOLUTION"))

            # Write the QuakeML to OUTPUT_FILES folder
            src = os.path.join(cmtsim_outdir, "CMTSOLUTION")
            outfiles = os.path.join(self.outdir, "CMT_lon",
                                    "OUTPUT_FILES")
            dst = os.path.join(outfiles, "Quake.xml")

            self._write_quakeml(src, dst)

            # Print info if verbose flag is True
            logger.verbose("%s has been written." % dst)

            # Replace STATIONS FILE
            station_source = os.path.join(self.station_dir, "STATIONS")
            replace_file(station_source, os.path.join(cmtsim_outdir,
                                                      "STATIONS"))

    def _write_quakeml(self, source, destination):
        """ Copies CMT solution from source to QuakeML destination."""

        # CMT Source file
        catalog = read_events(source)
        catalog.write(destination, format="QUAKEML")

    def __str__(self):
        string = "-------- CMT Source Writer --------\n"
        string += "Number of parameters to invert for: %d\n" % self.npar
        string += "dM: %f in Nm\n" % self.dm

        if self.npar > 6:
            string += "dx: %f in m\n" % self.dx
        if self.npar >= 9:
            string += "ddeg: %f in degrees\n" % self.ddeg

        return string
