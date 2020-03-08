"""

This file contains functions to compute statistics

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: November 2019

"""

from gcmt3d.source import CMTSource
from gcmt3d.plot.stats import PlotStats
from glob import glob
import os
import numpy as np


def read_specfem_station_list(filename):
    """Gets stationlist from file

    Args:
        filename (string): specfem station filename

    Returns:
        list of arguments for each station

    Station list:
        [network station latitude longitude elevation]
    """

    stationlist = []
    with open(filename, 'r') as specfemfile:

        for line in specfemfile:
            # Read stations into list of stations
            line = line.split()
            # Append the [network station latitude longitude elevation]
            newline = [line[1], line[0], float(line[2]), float(line[3]),
                       float(line[4]), float(line[5])]
            # to the station list
            stationlist.append(newline)

    return stationlist


class Statistics(object):
    """Governs the statistics of multiple inversions"""

    def __init__(self, old_cmts, old_ids, new_cmts, new_ids, stations,
                 npar=9, verbose=True):
        """ Initialize Statistics class

        Args:
        -----
        old_cmts (numpy.ndarray): matrix with old CMT data
        old_ids (list): List of event ids corresponding to matrix rows
        new_cmts (numpy.ndarray): matrix with new CMT data
        new_ids (list): List of event ids corresponding to matrix rows
        stations: list of stations
        npar: number of parameters to perform the analysis on
        verbose: Set verbosity

        Returns:
        --------
        Statistics class containing all necessary data and tools to perform
        an analysis on the data.

        The matrices below should have following columns:
        -------------------------------------------------
        M0, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, depth, lat, lon, CMT, t_shift, hdur

        """
        self.ocmt = old_cmts
        self.oids = old_ids
        self.ncmt = new_cmts
        self.nids = new_ids
        self.npar = npar

        # Sanity check
        if any([False for a, b in zip(self.oids, self.nids) if a == b]):
            raise ValueError("Can only compare equal earthquakes.")

        self.ids = self.nids
        self.stations = stations

        # Compute difference/evolution
        self.dCMT = self.ncmt - self.ocmt

        # Compute Correlation Coefficients
        self.xcorr_mat = np.corrcoef(self.dCMT.T)

        # Compute Mean
        self.mean_mat = np.mean(self.dCMT, axis=0)

        self.mean_dabs = np.mean(np.abs(self.dCMT), axis=0)
        print(self.mean_dabs)

        # Compute Standard deviation
        self.std_mat = np.std(self.dCMT, axis=0)

        # Create labels
        self.labels = ["$M_0$",
                       "$M_{rr}$", "$M_{tt}$",
                       "$M_{pp}$", "$\\delta M_{rt}$",
                       "$M_{rp}$", "$M_{tp}$",
                       "$z$", "Lat", "Lon",
                       "$t_{CMT}$", "$t_{shift}$", "$hdur$"]

        self.dlabels = ["$\\delta M_0$",
                        "$\\delta M_{rr}$", "$\\delta M_{tt}$",
                        "$\\delta M_{pp}$", "$\\delta M_{rt}$",
                        "$\\delta M_{rp}$", "$\\delta M_{tp}$",
                        "$\\delta z$", "$\\delta$Lat", "$\\delta$Lon",
                        "$\\delta t_{CMT}$", "$\\delta t_{shift}$",
                        "$\\delta hdur$"]

        self.verbose = verbose

    @classmethod
    def _from_dir(cls, directory, npar=9, verbose=True):
        """Load old and new inverted CMTSOLUTIONS into lists of CMTSources.

        Args:
        -----
            database_dir (string): database directory containing all the CMT
                                    solutions

        Returns:
        --------
            tuple of two lists containing the original cmtsolution and its
            corresponding inversion

        """

        # Get list of inversion files.
        Clist = glob(os.path.join(database_dir, "C*"))

        if verbose:
            print("Looking for earthquakes here: %s" % (database_dir))

        # Old CMTs
        old_cmts = []
        new_cmts = []
        station_list = set()

        # Loop of files
        for invdata in Clist:

            if verbose:
                print(invdata)

            # Get Cid
            bname = os.path.basename(invdata)
            # print(invdata)
            id = bname[1:]

            # Original CMT file name
            cmt_file = os.path.join(invdata, bname + ".cmt")
            # print(cmt_file)

            # Inverted CMT filename
            glob_path = os.path.join(invdata,
                                     'inversion',
                                     'inversion_output', id + "*.inv")
            inv_cmt = glob(glob_path)
            # print(inv_cmt)

            try:
                # Reading the CMT solutions
                old_cmt = CMTSource.from_CMTSOLUTION_file(cmt_file)
                new_cmt = CMTSource.from_CMTSOLUTION_file(inv_cmt[0])

                if verbose:
                    print("Got both for %s" % bname)

                # Append CMT files
                old_cmts.append(old_cmt)
                new_cmts.append(new_cmt)

            except Exception:
                if verbose:
                    print("Got nothing for %s" % bname)

            # Get stations file
            try:
                station_list = list(station_list)

                for row in read_specfem_station_list(
                        os.path.join(invdata, 'station_data', 'STATIONS')):

                    station_list.append(tuple(row))

                station_list = set(station_list)

            except Exception as e:
                print(e)

        old_cmt_mat, old_ids = Statistics.create_cmt_matrix(old_cmts)
        new_cmt_mat, new_ids = Statistics.create_cmt_matrix(new_cmts)

        print(len(station_list))

        return cls(old_cmt_mat, old_ids, new_cmt_mat, new_ids,
                   list(station_list),
                   npar=npar, verbose=verbose)

    def plot_changes(self, savedir=None):
        """This function plots and saves the plots with the statistics.

        Args:
            savedir (str, optional): Sets directory where to save the
                                     plots. If None the plots will be
                                     displayed. Defaults to None.
        """

        PS = PlotStats(ocmt=self.ocmt, ncmt=self.ncmt, dCMT=self.dCMT,
                       xcorr_mat=self.xcorr_mat, mean_mat=self.mean_mat,
                       std_mat=self.std_mat, labels=self.labels,
                       dlabels=self.dlabels, stations=self.stations,
                       npar=self.npar, verbose=self.verbose,
                       savedir=savedir)

        PS.plot_changes()
        PS.plot_xcorr_matrix()
        PS.plot_xcorr_heat()

    @staticmethod
    def create_cmt_matrix(cmt_source_list):
        """Takes in list of CMT sources and transforms it into a
        :class:`numpy.ndarray`

        Args:
        -----
            CMT source list

        Returns:
        --------
            :class:`numpy.ndarray`

        """

        # Number of Earthquakes
        N = len(cmt_source_list)

        # Create empty array for the values
        event_id = []

        # Initialize empty matrix
        cmt_mat = np.zeros((N, 13))

        for _i, cmt in enumerate(cmt_source_list):

            # Populate id list with ids
            event_id.append(cmt.eventname)

            # Populate CMT matrix
            cmt_mat[_i, :] = np.array([cmt.M0, cmt.m_rr, cmt.m_tt, cmt.m_pp,
                                       cmt.m_rt,
                                       cmt.m_rp, cmt.m_tp, cmt.depth_in_m,
                                       cmt.latitude, cmt.longitude,
                                       cmt.cmt_time,
                                       cmt.half_duration,
                                       cmt.time_shift,
                                       ])

        return cmt_mat, event_id


if __name__ == "__main__":

    # Load shit
    database_dir = "/Users/lucassawade/straverse/database"

    ST = Statistics._from_dir(database_dir)
    ST.plot_changes(savedir="/Users/lucassawade")
