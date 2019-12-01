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
# import cartopy.feature as cfeature
# import matplotlib.ticker as mticker
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy.imaging.beachball import beach
from gcmt3d.stats.stats import compute_differences
from gcmt3d.stats.stats import load_cmts
from gcmt3d.stats.stats import create_cmt_matrix
from gcmt3d.stats.stats import get_difference_stats
from gcmt3d.stats.stats import compute_correlation_matrix
import matplotlib
matplotlib.rcParams['text.usetex'] = True





def plot_cmts(ax, latitude, longitude, depth, mt, nmmt, alpha):

    for (lon, lat, d, m, sm) \
            in zip(longitude.tolist(), latitude.tolist(), depth.tolist(),
                   mt.tolist(), nmmt.tolist()):
        try:
            b = beach(m, linewidth=0.25, facecolor='k', bgcolor='w',
                      edgecolor='k', alpha=alpha, xy=(lon, lat), width=20,
                      size=100, nofill=False, zorder=100,
                      axes=ax)

            ax.add_collection(b)
        except Exception as e:
            print(e)


def plot_histogram(ddata, n_bins, facecolor=(0.8, 0.3, 0.3)):
    """Plots histogram of input data."""

    # the histogram of the data
    ax = plt.gca()
    n, bins, patches = ax.hist(ddata, n_bins, facecolor=facecolor, alpha=1)


def plot_cross_correlation_matrix(M, xcorrcoeff):
    """

    :param M: Data matrix
    :param xcorrcoeff: cross-correlation coefficient
    :return:
    """

    plt.figure()


class PlotStats(object):
    """Plots statistics given the necessary variables."""

    def __init__(self, ocmt=None, ncmt=None, dCMT=None, xcorr_mat=None,
                 mean_mat=None, std_mat=None, labels=None, dlabels=None,
                 npar= 9, verbose=True, savedir=None):
        """
        Parameters
        ----------
        ocmt: old cmt matrix
        ncmt: new cmt matrix
        dCMT: diff cmt matrix
        xcorr_mat: cross correlation matrix
        mean_mat: mean vector
        std_mat: std vector
        labels: labels
        dlabels: delta label
        nbins: bins in the histograms
        savedir: directory to save the figure output
        verbose: verbosity
        """

        self.ocmt = ocmt
        self.depth = ocmt[:, 7]
        self.latitude = ocmt[:, 8]
        self.longitude = ocmt[:, 9]
        self.N = ocmt.shape[0]
        self.ncmt = ncmt
        self.dCMT = dCMT
        self.xcorr_mat = xcorr_mat
        self.mean_mat = mean_mat
        self.std_mat = std_mat
        self.labels = labels
        self.dlabels = dlabels
        self.nbins
        self.verbose = verbose
        self.savedir = savedir

    def plot_changes(self):
        """Plots figure with statistics."""

        # Create figure handle
        fig = plt.figure(figsize=(11, 8.5))

        # Create subplot layout
        GS = GridSpec(5, 6)

        # Xcorr

        # Create axis for map
        fig.add_subplot(GS[:2, 0:3], projection=PlateCarree(0.0))
        self.plot_map()

        # table axes
        fig.add_subplot(GS[0:2, 3])
        self.plot_table()

        # MT
        counter = 1
        for _i in range(3):
            for _j in range(2):
                fig.add_subplot(GS[0 + _i, 4 + _j])
                self.plot_histogram(dCMT[:, counter], n_bins, facecolor=(0.8, 0.8,
                                                                0.8))
                plt.xlabel("%s" % (self.dlabels[counter]))
                counter += 1

        # loc_ax
        fig.add_subplot(GS[2, 0])
        plot_histogram(dCMT[:,8], n_bins)
        plt.xlabel("$\\delta$Lat [$^{\\circ}$]")
        fig.add_subplot(GS[2, 1])
        plot_histogram(dCMT[:,9], n_bins)
        plt.xlabel("$\\delta$Lon [$^{\\circ}$]")
        fig.add_subplot(GS[2, 2])
        plot_histogram(dCMT[:,7], n_bins)
        plt.xlabel("$\\delta z$ [$^{\\circ}$]")
        fig.add_subplot(GS[2, 3])
        plot_histogram(dCMT[:,0], n_bins)
        plt.xlabel("$\\delta$M_0 [$^{\\circ}$]")

        # Change of parameter as function of depth
        fig.add_subplot(GS[3:, 0:2])  # moment vs depth
        plt.plot(dCMT[:, 0], ocmts_mat[:, 7]/1000, "ko")
        plt.gca().invert_yaxis()
        plt.xlabel("$\\delta M_0$")
        plt.ylabel("$z$ [km]")
        
        # ddepth vs depth
        fig.add_subplot(GS[3:, 2:4]) 
        dlt.plot(dCMT[:, 7] / 1000, ocmts_mat[:, 7]/1000, "ko")
        plt.xlabel("$\\delta z$")
        plt.ylabel("$z$ [km]")
        plt.gca().invert_yaxis()
        
        # ddepth vs dM0
        fig.add_subplot(GS[3:, 4:]) 
        plt.plot(dCMT[:,0], dCMT[:, 7]/ 1000, "ko")
        plt.ylabel("$\\delta z$")
        plt.xlabel("$\\delta M_0$")
        plt.gca().invert_yaxis()

        # Finally plot shot
        plt.tight_layout()
        if savefig is not None:
            plt.savefig(os.path.join(self.savedir, "statfigure.pdf"))
        else:
            plt.show(block=True)

    def plot_map(self):

        ax = plt.gca()
        ax.set_global()
        ax.frameon = True
        ax.outline_patch.set_linewidth(0.75)

        # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
        # function around 180deg
        gl = ax.gridlines(crs=PlateCarree(), draw_labels=False,
                          linewidth=1, color='lightgray', alpha=0.5,
                          linestyle='-')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True

        # Add Coastline
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black',
                       facecolor=(0.85, 0.85, 0.85))

        ax.set_title("Inversion Statistics for %d earthquakes."
                     % (self.N))

    def plot_cmts(self):
        ax = plt.gca()
        for (lon, lat, m) in zip(self.longitude, self.latitude,
                                 self.ncmt[1:7]):
            try:
                b = beach(m, linewidth=0.25, facecolor='k', bgcolor='w',
                          edgecolor='k', alpha=alpha, xy=(lon, lat), width=20,
                          size=100, nofill=False, zorder=100,
                          axes=ax)

                ax.add_collection(b)
            except Exception as e:
                if self.verbose:
                    print(e)

    def plot_histogram(ddata, n_bins, facecolor=(0.8, 0.3, 0.3)):
        """Plots histogram of input data."""

        # the histogram of the data
        ax = plt.gca()
        n, bins, patches = ax.hist(ddata, n_bins, facecolor=facecolor, alpha=1)


    def plot_table(self):
        """Plots minimal summary"""

        columns = ('$\\overline{d}$', '$\\sigma$')
        rows = ['$\\delta t$', '$\\delta$Lat', '$\\delta$Lon', '$\\delta z$',
                '$\\delta M_0$', "$\\delta M_{rr}$", "$\\delta M_{tt}$",
                "$\\delta M_{pp}$", "$\\delta M_{rt}$", "$\\delta M_{rp}$",
                "$\\delta M_{tp}$"]

        cell_text = []

        for _i in range(len(stats_list)):
            if _i == 3:
                cell_text.append(["%3.3f" % (stats_list[_i][0] / 1000),
                                  "%3.3f" % (stats_list[_i][1] / 1000)])
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

        # Plot table
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
                       loc='center', edges='horizontal', fontsize=13)


if __name__ == "__main__":

    pass