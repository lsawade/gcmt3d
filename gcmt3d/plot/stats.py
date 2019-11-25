"""

This file contains functions to plot statistics

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: November 2019


"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import PlateCarree
import cartopy
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy.imaging.beachball import beach
from gcmt3d.stats.stats import compute_differences
from gcmt3d.stats.stats import load_cmts
from gcmt3d.stats.stats import create_cmt_matrix
from gcmt3d.stats.stats import get_difference_stats
import matplotlib
matplotlib.rcParams['text.usetex'] = True



def plot_map(ax):

    ax.set_global()
    ax.frameon = True
    # ax.outline_patch.set_visible(False)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=PlateCarree(), draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = True

    # Change fontsize
    fontsize = 12
    font_dict = {"fontsize": fontsize,
                 "weight": "bold"}
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=font_dict)

    # Get WMS map if wanted
    # ax.stock_img()
    # ax.add_feature(cfeature.LAND, zorder=10)
    # ax.add_feature(cfeature.OCEAN, zorder=10)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black',
                   facecolor=(0.85, 0.85, 0.85))
    # ax.coastlines(resolution='50m', color='black', linewidth=1, facecolor=(
    #     0.8, 0.8, 0.8))



def plot_cmts(ax, latitude, longitude, depth, mt, nmmt, alpha):

    for (lon, lat, d, m, sm) \
            in zip(longitude.tolist(), latitude.tolist(), depth.tolist(),
                   mt.tolist(), nmmt.tolist()):
        try:
            b = beach(m, linewidth=0.25, facecolor='k', bgcolor='w',
                      edgecolor='k', alpha=alpha, xy=(lon, lat), width=40,
                      size=100, nofill=False, zorder=100,
                      axes=ax)

            ax.add_collection(b)
        except Exception as e:
            print(e)


def plot_histogram(ax, ddata, n_bins, facecolor=(0.8, 0.3, 0.3)):
    """Plots histogram of input data."""

    # the histogram of the data
    n, bins, patches = ax.hist(ddata, n_bins, facecolor=facecolor, alpha=1)


def setup_figure():
    """Plots figure with statistics."""

    # Create figure handle
    fig = plt.figure(figsize=(11, 8.5))

    # Create subplot layout
    GS = GridSpec(5, 6)

    # Create axis for map
    map_ax = fig.add_subplot(GS[:2, 1:4], projection=PlateCarree(0.0))

    # table axes
    table_ax = fig.add_subplot(GS[0:2, 0])

    # MT
    mt_ax = []
    for _i in range(3):
        for _j in range(2):
            mt_ax.append(fig.add_subplot(GS[0+_i, 4+_j]))

    # loc_ax
    lat_ax = fig.add_subplot(GS[2, 0])
    lon_ax = fig.add_subplot(GS[2, 1])
    dep_ax = fig.add_subplot(GS[2, 2])
    m0_ax = fig.add_subplot(GS[2, 3])

    # Change of parameter as function of depth
    m0d = fig.add_subplot(GS[3:, 0:2])  # moment as function of depth
    dad = fig.add_subplot(GS[3:, 2:4])  # depth as function of depth
    ddM0 = fig.add_subplot(GS[3:, 4:])  # change in depth vs change in M0


    return table_ax, map_ax, mt_ax, lat_ax, lon_ax, dep_ax, m0_ax, m0d, dad, \
           ddM0


if __name__ =="__main__":

    # Load shit
    database_dir = "/Users/lucassawade/tigress/database"
    ocmts, ncmts = load_cmts(database_dir)

    # Create matrices:    cmt_time = origin time + time shift
    # cmt_time, time_shift, hdur, lat, lon, depth, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
    # o = old, n = new
    ocmts_mat, ocmt_ids = create_cmt_matrix(ocmts)
    ncmts_mat, ncmt_ids = create_cmt_matrix(ncmts)

    # Compute difference
    dt, dlat, dlon, dd, dM, dM0 = compute_differences(ocmts_mat, ncmts_mat)

    # Get statistics of the differences
    stats_list = get_difference_stats(dt, dlat, dlon, dd, dM, dM0)

    columns = ('$\overline{d}$', '$\sigma$')
    rows = ['$\delta t$', '$\delta$Lat', '$\delta$Lon', '$\delta z$',
            '$\delta M_0$', "$\delta M_{rr}$", "$\delta M_{tt}$",
            "$\delta M_{pp}$", "$\delta M_{rt}$", "$\delta M_{rp}$",
            "$\delta M_{tp}$"]

    cell_text = []

    for _i in range(len(stats_list)):
        if _i == 3:
            cell_text.append(["%3.3f" % (stats_list[_i][0]/1000),
                              "%3.3f" % (stats_list[_i][1]/1000)])
        elif _i == 4:
            cell_text.append(["%3.1e" % (stats_list[_i][0]),
                              "%3.1e" % (stats_list[_i][1])])
        elif _i == 5:
            for _j in range(6):
                cell_text.append(["%3.1e" % (stats_list[_i][0][_j]),
                                  "%3.1e" % (stats_list[_i][1][_j])])
        else:
            cell_text.append(["%3.3f" % (stats_list[_i][0]),
                              "%3.3f" % (stats_list[_i][1])])

    # Everythings an axes except mt_ax because list
    table_ax, map_ax, mt_ax, lat_ax, lon_ax, \
    dep_ax, m0_ax, m0d_ax, dad_ax, ddM0_ax = \
        setup_figure()

    # Plot map
    plot_map(map_ax)
    map_ax.set_title("Inversion Statistics for %d earthquakes."
                     % (dt.shape[0]))

    # Plot table
    table_ax.axis('tight')
    table_ax.axis('off')
    table_ax.table(cellText=cell_text,
                   rowLabels=rows,
                   colLabels=columns,
                   loc='center',
                   edges='horizontal', fontsize=13)

    # Plot CMTs
    plot_cmts(map_ax,
              ncmts_mat[:, 3],  # lat
              ncmts_mat[:, 4],  # lon
              ncmts_mat[:, 5],  # dep
              ncmts_mat[:, 6:12],  # mt
              ncmts_mat[:, 12],  # M0
              alpha=1)

    ## Histograms
    # Number of bins
    n_bins = 9

    # Location
    plot_histogram(lat_ax, dlat, n_bins)
    lat_ax.set_xlabel("$\delta$Lat [$^{\circ}$]")
    plot_histogram(lon_ax, dlon, n_bins)
    lon_ax.set_xlabel("$\delta$Lon [$^{\circ}$]")
    plot_histogram(dep_ax, dd/1000, n_bins)
    dep_ax.set_xlabel("$\delta z$ [km]")

    # Moment Tensor
    mt_str = ["M_{rr}", "M_{tt}", "M_{pp}", "M_{rt}", "M_{rp}", "M_{tp}"]
    for _i in range(6):
        plot_histogram(mt_ax[_i], dM[:, _i], n_bins, facecolor=(0.8, 0.8,
                                                                0.8))
        mt_ax[_i].set_xlabel("$\delta %s$" % (mt_str[_i]))

    plot_histogram(m0_ax, dM0, n_bins)
    m0_ax.set_xlabel("$\delta M_0$")

    ## Stuff as function of depth
    # Scalar magnitude change as function of depth
    m0d_ax.plot(dM0, ocmts_mat[:, 5]/1000, "ko")
    m0d_ax.invert_yaxis()
    m0d_ax.set_xlabel("$\delta M_0$")
    m0d_ax.set_ylabel("$z$ [km]")
    # depth change as function of depth
    dad_ax.plot(dd/1000, ocmts_mat[:, 5]/1000, "ko")
    dad_ax.set_xlabel("$\delta z$")
    dad_ax.set_ylabel("$z$ [km]")
    dad_ax.invert_yaxis()
    # depth change vs scalar magnitude change
    ddM0_ax.plot(dM0, dd / 1000, "ko")
    ddM0_ax.set_ylabel("$\delta z$")
    ddM0_ax.set_xlabel("$\delta M_0$")
    ddM0_ax.invert_yaxis()

    plt.tight_layout()
    plt.show(block=True)










