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
from glob import glob
import os
import numpy as np

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

    # time_shift, hdur, lat, lon, depth, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, M0

    for _i, cmt in enumerate(cmt_source_list):

        # Populate id list with ids
        event_id.append(cmt.eventname)


        # Populate CMT matrix
        cmt_mat[_i, :] = np.array([cmt.cmt_time, cmt.time_shift,
                                   cmt.half_duration,
                                   cmt.latitude, cmt.longitude, cmt.depth_in_m,
                                   cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt,
                                   cmt.m_rp, cmt.m_tp, cmt.M0])


    return cmt_mat, event_id



def load_cmts(database_dir):
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
    eq_list = glob(os.path.join(database_dir, "eq_*"))
    print("Looking for earthquakes here: %s" % (database_dir))

    # Old CMTs
    old_cmts = []
    new_cmts = []

    # Loop of files
    for invdata in eq_list:

        # Get eq_id
        bname = os.path.basename(invdata)
        id = bname[3:]

        # Original CMT file name
        cmt_file = os.path.join(invdata, bname + ".cmt")
        print(cmt_file)

        # Inverted CMT filename
        inv_cmt = glob(os.path.join(invdata,
                                    'inversion',
                                    'inversion_output', id + ".*.inv"))
        print(inv_cmt)

        try:
            # Reading the CMT solutions

            print('ocmt')
            old_cmt = CMTSource.from_CMTSOLUTION_file(cmt_file)

            print('ncmt')
            new_cmt = CMTSource.from_CMTSOLUTION_file(inv_cmt[0])

            # Append CMT files
            old_cmts.append(old_cmt)
            new_cmts.append(new_cmt)

        except Exception as e:
            print(e)

    return old_cmts, new_cmts


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

def get_difference_stats(dt, dlat, dlon, dd, dM, dM0):
    """
    :param dt: Change in centroid time
    :param dlat: Change in latitude
    :param dlon: Change in latitude
    :param dd: Change in depth
    :param dM: Change in moment tensor elements
    :param dM0: Change in scalar magnitude
    :return:
    """

    # Mean,std Change in Time
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)

    # Mean,std Change in latitude
    mean_dlat = np.mean(dlat)
    std_dlat = np.std(dlat)

    # Mean,std Change in longitude
    mean_dlon = np.mean(dlon)
    std_dlon = np.std(dlon)

    # Mean,std Change in depth
    mean_dd = np.mean(dd)
    std_dd = np.std(dd)

    # Mean,std change in scalar moment
    mean_dM0 = np.mean(dM0)
    std_dM0 = np.std(dM0)

    # Mean,std change in moment tensor elements
    mean_dm = np.mean(dM, axis=0)
    std_dm = np.std(dM, axis=0)

    return [[mean_dt, std_dt], [mean_dlat, std_dlat], [mean_dlon, std_dlon],
            [mean_dd, std_dd], [mean_dM0, std_dM0], [mean_dm, std_dm]]


if __name__ == "__main__":

    database_dir = "/Users/lucassawade/tigress/database"
    ocmts, ncmts = load_cmts(database_dir)

    print(len(ocmts))
    print(len(ncmts))

    # Create matrices:    cmt_time = origin time + time shift
    # cmt_time, time_shift, hdur, lat, lon, depth, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
    # o = old, n = new
    ocmts_mat, ocmt_ids = create_cmt_matrix(ocmts)
    ncmts_mat, ncmt_ids = create_cmt_matrix(ncmts)

    print(ocmts_mat[:, 0])
    print(ncmts_mat[:, 0])

    dt, dlat, dlon, dd, dM, dM0 = compute_differences(ocmts_mat, ncmts_mat)

    print(dlat)

