"""

This file contains functions to compute statistics

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: November 2019

"""

from ..source import CMTSource
from ..plot.stats import PlotStats
from glob import glob
import os
import numpy as np


def compute_differences(ocmts_mat, ncmts_mat):
    """
    :param ocmts: List of global :class:`gmct3d.source.CMTSource`'s
    :param ncmts: List of inverted :class:`gmct3d.source.CMTSource`'s
    :return:
    """

    # Difference in time
    dt = ncmts_mat[:, 0] - ocmts_mat[:, 0]

    # Difference in location
    dlat = ncmts_mat[:, 3] - ocmts_mat[:, 3]
    dlon = ncmts_mat[:, 4] - ocmts_mat[:, 4]
    dd = ncmts_mat[:, 5] - ocmts_mat[:, 5]

    # Difference in Moment tensor compononents
    dm = ncmts_mat[:, 6:12] - ocmts_mat[:, 6:12]

    # Compute Scalar Moment difference
    dm0 = ncmts_mat[:, 12] - ocmts_mat[:, 12]

    return dt, dlat, dlon, dd, dm, dm0


def get_differences(database_dir):
    """Use above functions to output differences"""

    ocmts, ncmts = load_cmts(database_dir)

    # return: dt, dlat, dlon, dd, dM, dM0
    return compute_differences(ocmts, ncmts)

def compute_correlation_matrix(dlat, dlon, dd, dm, dm0):
    """This takse in the computed differences and computes the correlation
    matrix

    :param dlat:
    :param dlon:
    :param dd:
    :param dm:
    :param dm0:
    :return:
    """

    # Create one large Matrix
    M = np.stack(dd, dlat, dlon, dm0, dm.T).T

    return M, np.corrcoef(M)


class Statistics(object):
    """Governs the statistics of multiple inversions"""

    def __init__(self, old_cmts, old_ids, new_cmts, new_ids, npar=9,
                 verbose=True):
        """ Initialize Statistics class

        Args:
        -----
        old_cmts (numpy.ndarray): matrix with old CMT data
        old_ids (list): List of event ids corresponding to matrix rows
        new_cmts (numpy.ndarray): matrix with new CMT data
        new_ids (list): List of event ids corresponding to matrix rows
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

        # Sanity check
        if any([False for a, b in zip(self.oids, self.nids) if a == b]):
            raise ValueError("Can only compare equal earthquakes.")

        self.ids = self.nids

        # Compute difference/evolution
        self.dCMT = self.ncmt - self.ocmt

        # Compute Correlation Coefficients
        self.xcorr_mat = np.corrcoef(self.dCMT.T)

        # Compute Mean
        self.mean_mat = np.mean(self.dCMT, axis=0)

        # Compute Standard deviation
        self.std_mat = np.std(self.dCMT, axis=0)


        # Create labels
        self.labels = ["$M_0",
                       "$M_{rr}$", "$M_{tt}$",
                       "$M_{pp}$", "$\\delta M_{rt}$",
                       "$M_{rp}", "$M_{tp}$",
                       "$z$", "Lat", "Lon",
                       "t_{CMT}", "$t_{shift}$", "$hdur$"]

        self.dlabels = ["$\\delta M_0",
                        "$\\delta M_{rr}$", "$\\delta M_{tt}$",
                        "$\\delta M_{pp}$", "$\\delta M_{rt}$",
                        "$\\delta M_{rp}", "$\\delta M_{tp}$",
                        "$\\delta z$", "$\\delta$Lat", "$\\delta$Lon",
                        "\\delta t_{CMT}", "$\\delta t_{shift}$",
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

            except Exception as e:
                if verbose:
                    print("Got nothing for %s" % bname)
                # print(e)

        old_cmt_mat, old_ids = create_cmt_matrix(old_cmts)
        new_cmt_mat, new_ids = create_cmt_matrix(new_cmts)

        return cls(old_cmt_mat, old_ids, new_cmt_mat, new_ids, npar=npar,
                    verbose=verbose)

    def plot_changes(self, savedir=None):
        """
        Plots changes or saves figure to given directory.
        
        """

        PS = PlotStats(ocmt=self.ocmt, ncmt=self.ncmt, dCMT=self.sCMT, 
                       xcorr_mat=self.xcorr_mat, mean_mat=self.mean_mat, 
                       std_mat=self.std_mat, labels=self.labels,
                       dlabels=self.dlabels, npar= self.npar,
                       verbose=self.verbose, savedir=savedir)
        PS.plot_changes()

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

        #

        for _i, cmt in enumerate(cmt_source_list):

            # Populate id list with ids
            event_id.append(cmt.eventname)

            # Populate CMT matrix
            cmt_mat[_i, :] = np.array([cmt.M0, cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt,
                                       cmt.m_rp, cmt.m_tp, cmt.depth_in_m,
                                       cmt.latitude, cmt.longitude,
                                       cmt.cmt_time,
                                       cmt.half_duration,
                                       cmt.time_shift,
                                       ])

        return cmt_mat, event_id


if __name__ == "__main__":

    # Load shit
    database_dir = "/Users/lucassawade/tigress/database"
    
    ST = Statistics._from_dir(database_dir)
    ST.plot_changes()
