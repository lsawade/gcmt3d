"""

This file contains functions to compute statistics

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: November 2019

"""

from pycmt3d.source import CMTSource
from gcmt3d.plot.stats import PlotStats
from glob import glob
import os
import numpy as np
from ..utils.io import load_json


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


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)


def get_stats_json(filename):
    """Reads the stats json and outputs dictionary with
    all the necessary data
    """
    d = load_json(os.path.abspath(filename))
    data_container = {"sta_lon": np.array(d["sta_lon"]),
                      "sta_lat": np.array(d["sta_lat"]),
                      "nwindows": d["nwindows"],
                      "nwin_on_trace": d["nwin_on_trace"]}
    cmtsource = CMTSource.from_dictionary(d["oldcmt"])
    new_cmtsource = CMTSource.from_dictionary(d["newcmt"])
    config = Struct(**d["config"])
    nregions = d["nregions"]
    bootstrap_mean = np.array(d["bootstrap_mean"])
    bootstrap_std = np.array(d["bootstrap_std"])
    var_reduction = d["var_reduction"]
    mode = d["mode"]
    G = d["G"]

    return (data_container, cmtsource, new_cmtsource,
            config, nregions, bootstrap_mean,
            bootstrap_std, var_reduction, mode,
            G)


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
        self.dCMT[:, :7] /= self.ocmt[:, 0, None]

        for _i, (dcmt, id) in enumerate(zip(self.dCMT, self.ids)):
            if dcmt[0] > 0.5 or dcmt[0] < -0.5 or \
                    dcmt[7] > 20000 or dcmt[7] < -20000:

                print("ID:", id,
                      "M0_0:", self.ocmt[_i, 0],
                      "M0_1:", self.ncmt[_i, 0],
                      "dCMT:", dcmt[0],
                      "dz:", dcmt[7])

        # Compute Correlation Coefficients
        self.xcorr_mat = np.corrcoef(self.dCMT.T)

        # Compute Mean
        self.mean_mat = np.mean(self.dCMT, axis=0)

        self.mean_dabs = np.mean(np.abs(self.dCMT), axis=0)
        # print(self.mean_dabs)

        # Compute Standard deviation
        self.std_mat = np.std(self.dCMT, axis=0)

        # Create labels
        self.labels = [r"$M_0$",
                       r"$M_{rr}$", r"$M_{tt}$",
                       r"$M_{pp}$", r"$\delta M_{rt}$",
                       r"$M_{rp}$", r"$M_{tp}$",
                       r"$z$", r"Lat", r"Lon",
                       r"$t_{CMT}$", r"$t_{shift}$", r"$hdur$"]

        self.dlabels = [r"$\delta M_0/M_0$",
                        r"$\delta M_{rr}/M_0$", r"$\delta M_{tt}/M_0$",
                        r"$\delta M_{pp}/M_0$", r"$\delta M_{rt}/M_0$",
                        r"$\delta M_{rp}/M_0$", r"$\delta M_{tp}/M_0$",
                        r"$\delta z$", r"$\delta$Lat", r"$\delta$Lon",
                        r"$\delta t_{CMT}$", r"$\delta t_{shift}$",
                        r"$\delta hdur$"]

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

        if verbose:
            print("Looking for earthquakes here: %s" % (database_dir))
        # Get list of inversion files.
        Clist = glob(os.path.join(database_dir, "C*",
                                  "inversion", "inversion_output",
                                  "*.json"))

        if verbose:
            print("glob done")

        # Old CMTs
        old_cmts = []
        new_cmts = []
        station_list = set()

        # Loop of files
        for invdata in Clist:
            # # print(invdata)
            # id = bname[1:]

            try:
                print(invdata)
                (data_container,
                 cmtsource,
                 new_cmtsource,
                 config,
                 nregions,
                 bootstrap_mean,
                 bootstrap_std,
                 var_reduction,
                 mode,
                 G) = get_stats_json(invdata)

                # Append CMT files
                old_cmts.append(cmtsource)
                new_cmts.append(new_cmtsource)

                # stations
                for lat, lon in zip(data_container["sta_lat"],
                                    data_container["sta_lon"]):
                    station_list.add((lat, lon))

            except Exception as e:
                print(e)

        old_cmt_mat, old_ids = Statistics.create_cmt_matrix(old_cmts)
        new_cmt_mat, new_ids = Statistics.create_cmt_matrix(new_cmts)

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
        # PS.plot_xcorr_matrix()
        PS.plot_xcorr_heat()

    @staticmethod
    def get_PTI_from_cmts(cmt_mat):
        """ Computes the Principal axes and components
        and gets the Isotropic component

        Args:
            cmt_mat (numpy.ndarray): Matrix of of cmt solution.
                                     Nx6
        Returns: Nx3x3 matrix and
        """
        pass

        """  for cmt in cmt_mat:
        m = np.array([cmt[0]]) """

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

            if cmt.eventname == "C201005051629A":
                print(cmt.eventname, cmt.M0)
                print(cmt)

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
