"""

This file contains functions to plot statistics

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: November 2019

"""


import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from cartopy.crs import PlateCarree
import cartopy
# import cartopy.feature as cfeature
# import matplotlib.ticker as mticker
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from obspy.imaging.beachball import beach
import matplotlib
from matplotlib import cm
from matplotlib import colors
from scipy.odr import RealData, ODR, Model
from ..log_util import modify_logger

logger = logging.getLogger(__name__)
modify_logger(logger)

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# Define a function (quadratic in our case) to fit the data with.
def linear_func(p, x):
    m, c = p
    return m*x + c


def fit_xy(x, y):
    # Create a model for fitting.
    linear_model = Model(linear_func)

    # Create a RealData object using our initiated data from above.
    data = RealData(x, y)

    # Set up ODR with the model and data.
    odr = ODR(data, linear_model, beta0=[0., 1.])

    # Run the regression.
    out = odr.run()

    return out.beta


def format_exponent(ax, axis='y'):

    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment = 'left'
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    # THIS IS A BUG
    # Well, at least it's sub-optimal because you might not
    # want to use tight_layout(). If anyone has a better way of
    # ensuring the offset text is updated appropriately
    # please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = '$\\times\\mathregular{10^{%d}}$' % expo

        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)

        # Add in a text box at the top of the y axis
        ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment)
    return ax


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


class PlotStats(object):
    """Plots statistics given the necessary variables."""

    def __init__(self, ocmt=None, ncmt=None, dCMT=None, xcorr_mat=None,
                 mean_mat=None, std_mat=None, stat_dict=None, labels=None,
                 dlabels=None, stations=None, nbins=15, npar=9, savedir=None):
        """
        Parameters:
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

        The matrices below should have following columns:
            M0, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp,
            depth, lat, lon, CMT, hdur, t_shift

        Station list rows have following content
            [network station latitude longitude elevation]
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
        self.stations = stations
        self.stat_dict = stat_dict

        # Save destination
        self.savedir = savedir

        # Fix depth
        self.ocmt[:, 7] = self.ocmt[:, 7]/1000
        self.ncmt[:, 7] = self.ncmt[:, 7]/1000
        self.dCMT[:, 7] = self.dCMT[:, 7]/1000
        self.mean_mat[7] = self.mean_mat[7]/1000
        self.std_mat[7] = self.std_mat[7]/1000

        self.nbins = nbins
        self.dmbins = np.linspace(-0.5, 0.5 + 0.5 / self.nbins, self.nbins)
        self.ddegbins = np.linspace(-0.1, 0.1 + 0.1 / self.nbins, self.nbins)
        self.dzbins = np.linspace(-25, 25 + 25 / self.nbins, self.nbins)
        self.dtbins = np.linspace(-10, 10 + 10 / self.nbins, self.nbins)

        # Measurement label and tag dictionary
        self.vtype_dict = {r'Time-shift: ${\Delta t}_{CC}$': "tshift",
                           r'$CC_{max} = _{max}|\frac{(d \star s)(\tau)}'
                           r'{\sqrt{(d_i^2) * (s_i^2)}}|$': "cc",
                           r'$P{L_1} = 10\log\left(\frac{d_i}'
                           r'{s_i}\right)$ [dB]':
                               "power_l1",
                           r'$P{L_2} = 10\log\left(\frac{d_i^2}'
                           r'{s_i^2}\right)$ [dB]':
                               "power_l2",
                           r'$P_{CC} = 10\log\frac{(d_i s_i)}'
                           r'{s_i^2}$ [dB]': "cc_amp",
                           r'$\frac{1}{2}\left|  d_i - s_i \right|^2$': "chi"}

    def plot_changes(self):
        """Plots figure with statistics."""

        # Create figure handle
        fig = plt.figure(figsize=(11, 10))

        # Create subplot layout
        GS = GridSpec(6, 6)

        # Xcorr

        # Create axis for map
        fig.add_subplot(GS[:2, :3], projection=PlateCarree(0.0))
        self.plot_map()
        self.plot_cmts()
        plt.title("Inversion statistics for %d earthquakes" % (self.N))

        # Create axis for map
        fig.add_subplot(GS[2:4, :3], projection=PlateCarree(0.0))
        self.plot_map()
        self.plot_stations()
        plt.title("Stations used in the inversion")

        # table axes
        # fig.add_subplot(GS[2:4, 3])
        # self.plot_table()

        # change in cmttime
        fig.add_subplot(GS[2, 3])
        self.plot_histogram(self.dCMT[:, 10], self.dtbins)
        plt.xlabel(r"$\delta t$")

        # MT
        counter = 1
        for _i in range(2):
            for _j in range(3):
                fig.add_subplot(GS[0 + _i, 3 + _j])
                self.plot_histogram(self.dCMT[:, counter],
                                    self.dmbins, facecolor=(0.8, 0.8, 0.8))
                plt.xlabel("%s" % (self.dlabels[counter]))
                counter += 1

        # loc_ax
        fig.add_subplot(GS[3, 4])
        self.plot_histogram(self.dCMT[:, 8], self.ddegbins)
        plt.xlabel("$\\delta$Lat [$^{\\circ}$]")
        fig.add_subplot(GS[3, 5])
        self.plot_histogram(self.dCMT[:, 9], self.ddegbins)
        plt.xlabel("$\\delta$Lon [$^{\\circ}$]")
        fig.add_subplot(GS[2, 4])
        self.plot_histogram(self.dCMT[:, 7], self.dzbins)
        plt.xlabel("$\\delta z$ [km]")
        fig.add_subplot(GS[2, 5])
        self.plot_histogram(self.dCMT[:, 0], self.dmbins)
        plt.xlabel("$\\delta M_0$")

        vmin = 10**(25.75)
        vmax = 10**(26.5)
        msize = 10

        # Change of parameter as function of depth
        fig.add_subplot(GS[4:, 0:2])  # moment vs depth
        sc = plt.scatter(self.dCMT[:, 0], self.ocmt[:, 7], c=self.ocmt[:, 0],
                         s=msize, marker='o', cmap=cm.rainbow,
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sc, orientation="horizontal")
        cbar.ax.set_ylabel(r"$M_0$")
        plt.xlim([-0.5, 0.5])
        plt.gca().invert_yaxis()
        plt.xlabel("$\\delta M_0$")
        plt.ylabel("$z$ [km]")

        # ddepth vs depth
        fig.add_subplot(GS[4:, 2:4])
        sc1 = plt.scatter(self.dCMT[:, 7], self.ocmt[:, 7], c=self.ocmt[:, 0],
                          s=msize, marker='o', cmap=cm.rainbow,
                          norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sc1, orientation="horizontal")
        cbar.ax.set_ylabel(r"$M_0$")
        plt.xlim([-20, 10])
        plt.xlabel(r"$\delta z$ [km]")
        plt.ylabel(r"$z$ [km]")
        plt.gca().invert_yaxis()

        # ddepth vs dM0
        fig.add_subplot(GS[4:, 4:])
        sc2 = plt.scatter(self.dCMT[:, 0], self.dCMT[:, 7], c=self.ocmt[:, 7],
                          s=msize, marker='o', cmap=cm.rainbow,
                          norm=colors.LogNorm())
        cbar = plt.colorbar(sc2, orientation="horizontal")
        cbar.ax.set_ylabel(r"$z$ [km]")
        plt.xlim([-0.5, 0.5])
        plt.ylim([-20, 10])
        plt.ylabel(r"$\delta z$ [km]")
        plt.xlabel(r"$\delta M_0$")
        plt.gca().invert_yaxis()

        # Finally plot shot
        plt.tight_layout()
        if self.savedir is not None:
            plt.savefig(os.path.join(self.savedir, "statfigure.pdf"))
        else:
            plt.show()

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
                       linewidth=0.5, facecolor=(0.9, 0.9, 0.9))

    def plot_cmts(self):
        ax = plt.gca()
        for (lon, lat, m) in zip(self.longitude, self.latitude,
                                 self.ncmt[:, 1:7]):
            try:
                b = beach(m, linewidth=0.25, facecolor='k', bgcolor='w',
                          edgecolor='k', alpha=1, xy=(lon, lat), width=10,
                          size=10, nofill=False, zorder=100,
                          axes=ax)

                ax.add_collection(b)
            except Exception as e:
                if self.verbose:
                    print(e)

    def plot_stations(self):
        """Plots stations into a map
        """

        slat = [station[0] for station in self.stations]
        slon = [station[1] for station in self.stations]

        ax = plt.gca()
        ax.scatter(slon, slat, s=20, marker='v', c=((0.85, 0.2, 0.2),),
                   edgecolors='k', linewidths=0.25, zorder=20)

    def plot_histogram(self, ddata, n_bins, facecolor=(0.8, 0.3, 0.3)):
        """Plots histogram of input data."""

        # the histogram of the data
        ax = plt.gca()
        n, bins, patches = ax.hist(ddata, n_bins, facecolor=facecolor, alpha=1)

    def plot_xcorr_matrix(self):
        """Plots Corrlation matrix with approximate correlation bars
        """

        fig = plt.figure(figsize=(12, 11))

        ax = fig.subplots(10, 10, sharex="col", sharey='row', squeeze=True,
                          gridspec_kw={'hspace': 0, 'wspace': 0})

        for _i in range(10):
            for _j in range(10):

                plt.sca(ax[_i][_j])
                if _j == _i:
                    shay = ax[_i][_j].get_shared_y_axes()
                    shay.remove(ax[_i][_j])
                    self.plot_histogram(self.dCMT[:, _i], self.nbins)
                else:
                    ax[_i][_j].plot(self.dCMT[:, _j],  self.dCMT[:, _i],
                                    'ko', markersize=2)

                    # OLS fit
                    A = np.vstack([self.dCMT[:, _j],
                                   np.ones(len(self.dCMT[:, _j]))]).T

                    m, c = np.linalg.lstsq(A, self.dCMT[:, _i], rcond=None)[0]

                    res = np.sqrt(np.sum(((c + m * self.dCMT[:, _j])
                                          - self.dCMT[:, _j]) ** 2))

                    print(res, np.sqrt(self.mean_mat[_j]**2
                                       + self.mean_mat[_i]**2))

                    if res < 0.25 * self.N*np.sqrt(self.mean_mat[_j]**2
                                                   + self.mean_mat[_i]**2):
                        # Different option to compute the OLS fit by computing
                        # the perpendicular distance
                        # m, c = fit_xy(self.dCMT[:, _j], self.dCMT[:, _i])

                        # Plot polyline
                        ax[_i][_j].plot(self.dCMT[:, _j],
                                        c + m * self.dCMT[:, _j],
                                        '-', c=(0.85, 0.2, 0.2))

        plt.tight_layout()
        for _i in range(10):
            for _j in range(10):
                if _i == 9:
                    ax[_i][_j].set_xlabel(self.dlabels[_j])
                if _j == 0:
                    ax[_i][_j].set_ylabel(self.dlabels[_i])
                    ax[_i][_j].yaxis.set_label_coords(-0.3, 0.5)

                    # Label magic happens here
                    if _i in [0, 1, 2, 3, 4, 5, 6]:
                        # Change the ticklabel format to scientific format
                        ax[_i][_j].ticklabel_format(axis="y", style='sci')
                        offset_text = \
                            ax[_i][_j].yaxis.get_offset_text().get_text()
                        ax[_i][_j].text(-.375, 0.85, offset_text,
                                        rotation='vertical',
                                        ha='center', va='center',
                                        transform=ax[_i][_j].transAxes)
                        ax[_i][_j].yaxis.get_offset_text().set_visible(False)

        # Finally plot shot
        plt.tight_layout()
        if self.savedir is not None:
            plt.savefig(os.path.join(self.savedir, "xcorr.pdf"))
        else:
            plt.show()

    def plot_measurement_changes(self):
        """ Uses the stat_dict to compute measurement statistics.

        """

        # Create figure
        if self.stat_dict is None:
            logger.info("No statistics dictionary here...")
            return

        # Get number of rows and columns
        nrows = len(self.stat_dict)
        ncols = len(self.vtype_dict)

        stats = ["mean", "std"]
        # Create figure

        for _stat in stats:

            fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
            G = GridSpec(nrows, ncols)

            for irow, cat in enumerate(self.stat_dict.keys()):
                for icol, (label, vtype) in enumerate(self.vtype_dict.items()):
                    ax = fig.add_subplot(G[irow, icol])
                    if irow == 0 and icol == 0:
                        fontsize = 20
                        ax.text(0.025, 0.975, _stat,
                                fontsize=fontsize,
                                horizontalalignment='left',
                                verticalalignment='top',
                                transform=ax.transAxes)

                    self._hist_sub(self.stat_dict[cat][vtype], cat, vtype,
                                   label, _stat, self.nbins)

            plt.tight_layout()
            plt.savefig(os.path.join(self.savedir,
                                     "measurement_changes_" + _stat + ".pdf"))
            plt.close(fig)

    def plot_mean_measurement_change_stats(self):
        """ Uses the stat_dict to compute measurement statistics.

        """

        # Create figure
        if self.stat_dict is None:
            logger.info("No statistics dictionary here...")
            return

        # Get number of rows and columns
        nrows = len(self.stat_dict)
        ncols = len(self.vtype_dict)

        # Create figure
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
        G = GridSpec(nrows, ncols)

        for irow, cat in enumerate(self.stat_dict.keys()):
            for icol, (label, vtype) in enumerate(self.vtype_dict.items()):
                ax = fig.add_subplot(G[irow, icol])
                if irow == 0 and icol == 0:
                    fontsize = 20
                    ax.text(0.025, 0.975, r"$\Delta$Mean",
                            fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=ax.transAxes)

                self._hist_sub_change(self.stat_dict[cat][vtype], cat, vtype,
                                      label, self.nbins)

        plt.tight_layout()
        plt.savefig(os.path.join(self.savedir,
                                 "mean_measurement_change_stats.pdf"))
        plt.close(fig)

    @staticmethod
    def _hist_sub(measurement_dict, cat, vtype, label, stat, num_bin):
        """

        :param measurement_dict: {after: {mean: list, std: list},
                                  before: {mean: list, std: list}}
        :param label:
        :param stat:
        :return:
        """
        # Get axes
        ax = plt.gca()

        plt.xlabel(label, fontsize=15)
        plt.ylabel(r"%s" % cat.replace("_", r"\_"), fontsize=15)

        data_b = measurement_dict['before'][stat]
        data_a = measurement_dict['after'][stat]

        if vtype == "cc":
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
        elif vtype == "chi":
            ax_min = 0.0
            ax_max = max(max(data_b), max(data_a))
        else:
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
            abs_max = max(abs(ax_min), abs(ax_max))
            ax_min = -abs_max
            ax_max = abs_max

        if stat == "std":
            ax_min = 0.0

        binwidth = (ax_max - ax_min) / num_bin

        # Stats
        a_mean = np.mean(data_a)
        a_std = np.std(data_a)
        b_mean = np.mean(data_b)
        b_std = np.std(data_b)

        nb, _, _ = ax.hist(
            data_b, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
            facecolor='blue', alpha=0.3)
        nb_max = np.max(nb)
        ax.plot([b_mean, b_mean], [0, nb_max], "b--")
        ax.plot([b_mean - b_std, b_mean + b_std],
                [nb_max / 2, nb_max / 2], "b--")
        na, _, _ = ax.hist(
            data_a, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
            facecolor='red', alpha=0.5)
        na_max = np.max(na)
        ax.plot([a_mean, a_mean], [0, na_max], "r-")
        ax.plot([a_mean - a_std, a_mean + a_std],
                [na_max / 2, na_max / 2], "r-")

    @staticmethod
    def _hist_sub_change(measurement_dict, cat, vtype, label, num_bin):
        """

        :param measurement_dict: {after: {mean: list, std: list},
                                  before: {mean: list, std: list}}
        :param label:
        :param stat:
        :return:
        """
        # Get axes
        ax = plt.gca()

        plt.xlabel(label, fontsize=15)
        plt.ylabel(r"%s" % cat.replace("_", r"\_"), fontsize=15)

        data_b = measurement_dict['before']["mean"]
        data_a = measurement_dict['after']["mean"]

        data = (np.array(data_a) - np.array(data_b)).tolist()

        if vtype == "cc":
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
        elif vtype == "chi":
            ax_min = 0.0
            ax_max = max(max(data_b), max(data_a))
        else:
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
            abs_max = max(abs(ax_min), abs(ax_max))
            ax_min = -abs_max
            ax_max = abs_max
        binwidth = (ax_max - ax_min) / num_bin

        # Stats
        d_mean = np.mean(data)
        d_std = np.std(data)

        nd, _, _ = ax.hist(
            data, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
            facecolor='black', alpha=0.3)
        nd_max = np.max(nd)
        ax.plot([d_mean, d_mean], [0, nd_max], "b-")
        ax.plot([d_mean - d_std, d_mean + d_std],
                [nd_max / 2, nd_max / 2], "b-")

    def plot_xcorr_heat(self):
        """Plots correlatio n heatmap.
        """
        fig = plt.figure(figsize=(12, 10))

        ax = fig.add_subplot(111)

        x = np.arange(10)
        y = np.arange(10)
        xx, yy = np.meshgrid(x, y)

        cmap = plt.get_cmap("coolwarm")
        size_scale = 750
        scat = ax.scatter(xx, yy, s=(np.abs(self.xcorr_mat[:10, :10])
                                     * size_scale),
                          c=self.xcorr_mat[:10, :10], cmap=cmap, marker='s',
                          alpha=0.75)

        cbar = plt.colorbar(scat, aspect=50, pad=0)
        cbar.ax.tick_params(labelsize=12)

        # Mapping from column names to integer coordinates
        plt.xticks(np.arange(10), self.dlabels[:10], fontsize=13)
        plt.yticks(np.arange(10), self.dlabels[:10], fontsize=13)

        # Relocate Grid
        ax.grid(False, 'major')
        ax.grid(True, 'minor')
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

        # Set limits
        ax.set_xlim([-0.5, max(x) + 0.5])
        ax.set_ylim([-0.5, max(x) + 0.5])

        # Invert y and put xaxis on top
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # Finally plot shot
        plt.tight_layout()
        if self.savedir is not None:
            plt.savefig(os.path.join(self.savedir, "xcorrheat.pdf"))
        else:
            plt.show()

    def plot_table(self):
        """Plots minimal summary"""

        columns = (r'$\mu$', r'$\sigma$')
        rows = [r'$\delta t$', r'$\delta$Lat', r'$\delta$Lon', r'$\delta z$',
                r'$\delta M_0$', r"$\delta M_{rr}$", r"$\delta M_{tt}$",
                r"$\delta M_{pp}$", r"$\delta M_{rt}$", r"$\delta M_{rp}$",
                r"$\delta M_{tp}$"]

        cell_text = []

        # dt
        cell_text.append(["%3.3f" % (self.mean_mat[10]),
                          "%3.3f" % (self.std_mat[10])])
        # dLat
        cell_text.append(["%3.3f" % (self.mean_mat[8]),
                          "%3.3f" % (self.std_mat[8])])
        # dLon
        cell_text.append(["%3.3f" % (self.mean_mat[9]),
                          "%3.3f" % (self.std_mat[9])])
        # dz
        cell_text.append(["%3.3f" % (self.mean_mat[7]),
                          "%3.3f" % (self.std_mat[7])])
        # M0
        cell_text.append(["%3.3f" % (self.mean_mat[0]),
                          "%3.3f" % (self.std_mat[0])])
        for _j in range(6):
            cell_text.append(["%3.3f" % (self.mean_mat[1 + _j]),
                              "%3.3f" % (self.std_mat[1 + _j])])

        # Plot table
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
                 loc='center', edges='horizontal', fontsize=13)

    def save_table(self):
        """Uses plot_table function to save figure"""

        # Create figure handle
        fig = plt.figure(figsize=(4, 2))

        # Create subplot layout
        self.plot_table()

        plt.savefig(os.path.join(self.savedir, "summary_table.pdf"))
        plt.close(fig)


if __name__ == "__main__":
    pass
