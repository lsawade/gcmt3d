"""

This file contains functions to compute statistics

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: November 2019

"""

import logging
from glob import glob
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
from ..source import CMTSource
from ..plot.plot_stats import PlotStats, PlotCatalogStatistics
from ..utils.io import load_json
from ..log_util import modify_logger
from ..plot.plot_event import extract_stations_from_traces, unique_locations

logger = logging.getLogger(__name__)
modify_logger(logger)


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


def get_stats_json(cmt3dfile, g3dfile, simple=False):
    """Reads the stats json and outputs dictionary with
    all the necessary data
    """
    c = load_json(os.path.abspath(cmt3dfile))
    g = load_json(os.path.abspath(g3dfile))

    sta_lat, sta_lon = extract_stations_from_traces(c["wave_dict"])
    sta_lat, sta_lon = unique_locations(sta_lat, sta_lon)
    station_list = [(lat, lon) for lat, lon in zip(sta_lat, sta_lon)]
    cmtsource = CMTSource.from_dictionary(c["oldcmt"])
    new_cmtsource = CMTSource.from_dictionary(g["newcmt"])

    if simple:
        return (station_list, cmtsource, new_cmtsource)

    config = Struct(**c["config"])
    nregions = c["nregions"]
    bootstrap_mean = np.array(c["bootstrap_mean"])
    bootstrap_std = np.array(c["bootstrap_std"])
    var_reduction = c["var_reduction"] * g["var_reduction"]
    mode = c["mode"]
    G = g
    stats = c["stats"]

    return (station_list, cmtsource, new_cmtsource,
            config, nregions, bootstrap_mean,
            bootstrap_std, var_reduction, mode,
            G, stats)


class Catalog(object):

    def __init__(self, ocmtfiles, ncmtfiles, stationfiles):

        """Simple statistics between two 

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        self.ocmtfiles = ocmtfiles
        self.ncmtfiles = ncmtfiles
        self.stationfiles = stationfiles

        # To be populated
        self.ocmtlist = []
        self.ncmtlist = []
        self.station_list = []
        self.get_cmtsourcelists()
        self.get_stationlist()

        # For stats
        self.ocmt, self.event = self.create_cmt_matrix(self.ocmtlist)
        self.ncmt, _ = self.create_cmt_matrix(self.ncmtlist)

    def get_cmtsourcelists(self):

        ocmts, ncmts = [], []
        for _ocmt in self.ocmtfiles:
            ocmts.append(CMTSource.from_CMTSOLUTION_file(_ocmt))
        for _ncmt in self.ncmtfiles:
            ncmts.append(CMTSource.from_CMTSOLUTION_file(_ncmt))

        for _ocmt in ocmts:
            logger.debug(f"Event: {_ocmt.eventname} ...")
            for _ncmt in ncmts:
                # logger.debug(f"oid: {_ocmt.eventname}  nid: {_ncmt.eventname}")
                if _ocmt.eventname == _ncmt.eventname:

                    self.ocmtlist.append(_ocmt)
                    self.ncmtlist.append(_ncmt)
                    logger.debug(f"...  found partner event: {_ncmt.eventname}")
                    break

    def get_stationlist(self):
        """ Gets unique stations used in all inversions.
        """
        sta_lat, sta_lon = [], []
        for _sfile in self.stationfiles:
            for station in read_specfem_station_list(_sfile):
                sta_lat.append(station[2])
                sta_lon.append(station[3])          
        sta_lat, sta_lon = unique_locations(sta_lat, sta_lon)
        self.station_list = [(lat, lon) for lat, lon in zip(sta_lat, sta_lon)]

    def save(self, filename):
        """saves stats file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, filename):
        """Loads Stats file"""
        with open(filename, 'rb') as file:
            self = pickle.load(file)

        return self

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


class CatalogStats(object):

    def __init__(self, cat: Catalog):
        """
        """

        # Compute difference before after
        self.cat = cat
        self.dcmt = self.cat.ncmt - self.cat.ocmt
        self.dcmt[:, :7] /= self.cat.ocmt[:, 0, None]
        self.dcmt[:, 7] /= 1000
        self.stations = np.array(self.cat.station_list)
        self.lat = self.stations[:, 0]
        self.lon = self.stations[:, 1]

        good_stats = []
        counter = 0
        for _i, (dcmt, ncmt, id) in enumerate(zip(self.dcmt,
                                                  self.cat.ncmt,
                                                  self.cat.event)):
            if (np.abs(dcmt[0]) > 0.5) or (np.abs(dcmt[7]) > 30) or (ncmt[7] < 0):
                logger.info("ID: %s" % id)
                logger.info("  M0_0: %e" % self.cat.ocmt[_i, 0])
                logger.info("  M0_1: %e" % self.cat.ncmt[_i, 0])
                logger.info("  dCMT: %f" % dcmt[0])
                logger.info("  dz: %f" % dcmt[7])
                logger.info("  nz: %f" % ncmt[7])
                logger.info("Removed C%s from matrix." % id)
                # good_stats.append(_i)
                counter += 1
            else:
                good_stats.append(_i)
        logger.info(" ")
        logger.info(f"    Removed {counter} CMTs from statistics"
                    "due to being outliers.")
        logger.info(" ")
        # Fix outliers
        good_stats = np.array(good_stats)
        self.ocmt = self.cat.ocmt[good_stats, :]
        self.event = np.array(self.cat.event)[good_stats]
        self.ncmt = self.cat.ncmt[good_stats, :]
        self.dcmt = self.dcmt[good_stats, :]

        # Compute Correlation Coefficients
        self.xcorr_mat = np.corrcoef(self.dcmt.T)

        # Compute Mean
        self.mean_mat = np.mean(self.dcmt, axis=0)
        self.mean_dabs = np.mean(np.abs(self.dcmt), axis=0)

        # Compute Standard deviation
        # Will throw warning for unchanged things.
        self.std_mat = np.std(self.dcmt, axis=0)

        # Labels for the stat plots
        self.labels = [r"$M_0$",
                       r"$M_{rr}$", r"$M_{tt}$",
                       r"$M_{pp}$", r"$M_{rt}$",
                       r"$M_{rp}$", r"$M_{tp}$",
                       r"$z$", r"Lat", r"Lon",
                       r"$t_{CMT}$", r"$hdur$"]
        self.tags = [
            "M_0", "M_rr", "M_tt", "M_pp", "M_rt", "M_rp", "M_tp", 
            "depth", "lat", "lon", "tcmt", "hdur"
            ]
        self.bounds = [
            [-10, 10], [-5, 5], [-10, 10], [-10, 10], [-10, 10], [-10, 10],
            [-10, 10], [-7.5, 7.5], [-1e-2, 1e-2], [-1e-2, 1e-2], [-2.5, 2.5], 
            [-30, 30]
        ]
        self.factor = [100, 100, 100, 100, 100, 100, 100, 1, 1, 1, 1, 1]
        self.units = ["%", "%", "%", "%", "%", "%", "%",
                      "km", "deg", "deg", "s", "s"]
        self.dlabels = [r"$\delta M0/M0$",
                        r"$\delta M{rr}/M0$", r"$\delta M{tt}/M0$",
                        r"$\delta M{pp}/M0$", r"$\delta M{rt}/M0$",
                        r"$\delta M{rp}/M0$", r"$\delta M{tp}/M0$",
                        r"$\delta z$", r"$\delta$Lat", r"$\delta$Lon",
                        r"$\delta t{CMT}$", r"$\delta hdur$"]

    def plot_changes(self, outdir="./", cmttime=False, hdur=False,
                     subsets=True):

        # Complete statistics
        ps = PlotCatalogStatistics(self.event, self.ocmt, self.ncmt, self.dcmt,
                                   self.xcorr_mat, self.mean_mat, self.std_mat,
                                   self.mean_dabs, self.stations, self.bounds,
                                   self.labels, self.dlabels, self.tags, 
                                   self.factor, self.units,
                                   outdir=outdir, prefix=None,
                                   cmttime=cmttime, hdur=hdur)
        ps.plot_main_stats()
        ps.plot_spatial_change()
        ps.plot_dM_dz()

        # Subsets
        ind0_70 = np.where((0 <= self.ocmt[:, 7])
                           & (self.ocmt[:, 7] <= 70000))[0]
        ind70_300 = np.where((70000 < self.ocmt[:, 7])
                             & (self.ocmt[:, 7] <= 300000))[0]
        ind300_ = np.where(300000 < self.ocmt[:, 7])[0]
        subsets = (("lt70", ind0_70),
                   ("70_300", ind70_300),
                   ("gt300", ind300_))
        print(len(ind0_70), len(ind70_300), len(ind300_))
        for _prefix, _ind in subsets:
            ps = PlotCatalogStatistics(
                self.event[_ind], self.ocmt[_ind, :],
                self.ncmt[_ind, :], self.dcmt[_ind, :],
                self.xcorr_mat, np.mean(self.dcmt[_ind, :], axis=0),
                np.std(self.dcmt[_ind, :], axis=0),
                np.mean(np.abs(self.dcmt[_ind, :]), axis=0), 
                self.stations, self.bounds,
                self.labels, self.dlabels, self.tags, 
                self.factor, self.units,
                outdir=outdir, prefix=_prefix,
                cmttime=cmttime, hdur=hdur)
            ps.plot_main_stats()
            ps.plot_spatial_change()
            ps.plot_dM_dz()

class Statistics(object):
    """Governs the statistics of multiple inversions"""

    def __init__(self, old_cmts, old_ids, new_cmts, new_ids, stations,
                 npar=9, stat_dict: dict or None = None):
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
        self.oids = np.array(old_ids)
        self.ncmt = new_cmts
        self.nids = np.array(new_ids)
        self.npar = npar

        # Sanity check
        if any([False for a, b in zip(self.oids, self.nids) if a == b]):
            raise ValueError("Can only compare equal earthquakes.")

        self.ids = self.nids
        self.stations = stations

        # Compute difference/evolution
        self.dCMT = self.ncmt - self.ocmt
        self.dCMT[:, :7] /= self.ocmt[:, 0, None]

        good_stats = []
        counter = 0
        for _i, (dcmt, ncmt, id) in enumerate(zip(self.dCMT, self.ncmt, self.ids)):
            if (np.abs(dcmt[0]) > 0.5) or (np.abs(dcmt[7]) > 30000) or (ncmt[7] < 0):
                logger.info("ID: %s" % id)
                logger.info("  M0_0: %e" % self.ocmt[_i, 0])
                logger.info("  M0_1: %e" % self.ncmt[_i, 0])
                logger.info("  dCMT: %f" % dcmt[0])
                logger.info("  dz: %f" % dcmt[7])
                logger.info("  nz: %f" % ncmt[7])
                logger.info("Removed C%s from matrix." % id)
                # good_stats.append(_i)
                counter += 1
            else:
                good_stats.append(_i)
        logger.info(" ")
        logger.info(f"    Removed {counter} CMTs from statistics"
                    "due to being outliers.")
        logger.info(" ")
        # Fix outliers
        good_stats = np.array(good_stats)
        self.ocmt = self.ocmt[good_stats, :]
        self.oids = self.oids[good_stats]
        self.ncmt = self.ncmt[good_stats, :]
        self.nids = self.nids[good_stats]
        self.dCMT = self.dCMT[good_stats, :]

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
                       r"$M_{pp}$", r"$M_{rt}$",
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

        self.stat_dict = stat_dict

    def save(self, filename):
        """saves stats file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, filename):
        """Loads Stats file"""
        with open(filename, 'rb') as file:
            self = pickle.load(file)
        return self

    @classmethod
    def _from_dir(self, directory, direct=False, npar=9):
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

        logger.info("Looking for earthquakes here: %s" % directory)
        # Get list of inversion files.
        clist = []
        glist = []
        for dir in directory:
            if direct:
                clist.extend(glob(os.path.join(dir, 'g3d', "*.json")))
                glist.extend(glob(os.path.join(dir, 'cmt3d', "*.json")))
            else:
                clist.extend(glob(os.path.join(dir, "C*", "inversion",
                                               "cmt3d", "*.json")))
                glist.extend(glob(os.path.join(dir, "C*", "inversion",
                                               "g3d", "*.json")))

        logger.info("Found all files.")
        logger.info(" ")

        # Old CMTs
        old_cmts = []
        new_cmts = []
        stat_dicts = []
        station_list = set()

        logger.info("Reading individual files ...")
        logger.info(" ")

        # Loop of files
        for cmt3d, g3d in zip(clist, glist):

            logger.debug("CMT3D: %s -- G3D: %s" % (os.path.basename(cmt3d),
                                                   os.path.basename(g3d)))
            try:
                logger.info("  File: %s" % cmt3d)

                (sta_list,
                 cmtsource,
                 new_cmtsource,
                 config,
                 nregions,
                 bootstrap_mean,
                 bootstrap_std,
                 var_reduction,
                 mode,
                 G,
                 stats) = get_stats_json(cmt3d, g3d)
                logger.debug("nd: %f -- od: %f -- dd: %f"
                             % (new_cmtsource.depth_in_m,
                                cmtsource.depth_in_m,
                                new_cmtsource.depth_in_m
                                - cmtsource.depth_in_m
                                ))
                # Append CMT files
                old_cmts.append(cmtsource)
                new_cmts.append(new_cmtsource)
                stat_dicts.append(stats)
                station_list.update(set(sta_list))

            except Exception as e:
                logging.warning(e)

        logger.info(" ")
        logger.info("Done.")
        logger.info(" ")

        old_cmt_mat, old_ids = Statistics.create_cmt_matrix(old_cmts)
        new_cmt_mat, new_ids = Statistics.create_cmt_matrix(new_cmts)

        complete_dict = self.compute_change(stat_dicts)

        return self(old_cmt_mat, old_ids, new_cmt_mat, new_ids,
                    list(station_list), npar=npar, stat_dict=complete_dict)

    def plot_changes(self, savedir=None):
        """This function plots and saves the plots with the statistics.

        Args:
            savedir (str, optional): Sets directory where to save the
                                     plots. If None the plots will be
                                     displayed. Defaults to None.
        """

        PS = PlotStats(ocmt=self.ocmt, ncmt=self.ncmt, dCMT=self.dCMT,
                       xcorr_mat=self.xcorr_mat, mean_mat=self.mean_mat,
                       std_mat=self.std_mat, stat_dict=self.stat_dict,
                       labels=self.labels, dlabels=self.dlabels,
                       stations=self.stations, npar=self.npar,
                       savedir=savedir)

        PS.plot_main_stats()
        PS.plot_dM_dz_oz()
        PS.plot_dM_dz_nz()
        PS.plot_dM_nz_dz()
        PS.plot_dM_oz_dz()
        PS.plot_dz_oz_dM()
        PS.plot_dz_nz_dM()
        PS.plot_dt_oz_dz()
        PS.plot_dt_nz_dz()
        PS.plot_z_z_dM()
        # PS.plot_changes()
        # PS.plot_xcorr_matrix()
        # PS.plot_xcorr_heat()
        # PS.plot_measurement_changes()
        # PS.plot_mean_measurement_change_stats()
        # PS.save_table()

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
    def compute_change(stat_dicts: list):
        """ Takes in the measurement statistics for and compiles them in
        vectors.

        :param stat_dicts:
        :return: tuple of dicts with before and after data

        """

        complete_dict = dict()

        for _sdict in stat_dicts:
            for tag, measurement_dict in _sdict.items():
                if tag not in complete_dict.keys():
                    complete_dict[tag] = dict()

                for measurement, ba_dict in measurement_dict.items():
                    if measurement not in complete_dict[tag].keys():
                        complete_dict[tag][measurement] = \
                            {"before": {"std": [], "mean": []},
                             "after": {"std": [], "mean": []}}

                    # Add STD
                    for time in ['before', 'after']:
                        complete_dict[tag][measurement][time]["std"].append(
                            np.std(ba_dict[time]))
                        complete_dict[tag][measurement][time]["mean"].append(
                            np.mean(ba_dict[time]))

        return complete_dict

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
