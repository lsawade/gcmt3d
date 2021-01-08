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
from copy import deepcopy
from matplotlib.gridspec import GridSpec
from cartopy.crs import PlateCarree
import cartopy
from obspy.imaging.beachball import beach
import matplotlib
from matplotlib import cm
from matplotlib import colors
import datetime as dt
from obspy import UTCDateTime
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
from scipy.odr import RealData, ODR, Model
import lwsspy as lpy

from .plot_util import remove_topright, remove_all
from .plot_util import create_colorbar
from .plot_util import get_color
from .plot_util import confidence_ellipse
from .plot_util import figletter
from ..log_util import modify_logger

logger = logging.getLogger(__name__)
modify_logger(logger)
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
lpy.updaterc()

# Define a function (quadratic in our case) to fit the data with.


def linear_func(p, x):
    m, c = p
    return m*x + c


def rgb2rgba(rgb, alpha):
    """Changes rgb tuple or list of rgbtuples
    to rgba tuple or list of rgba tuples.

    Args:
    rgb: tuple or list of tuples or numpy.ndarray
    RGB values
    alpha: float or int
    alpha value between 0 and 1

    """

    if (type(rgb) is tuple) or ((type(rgb) is np.ndarray) and (rgb.size is 3)):
        rgba = (rgb[0], rgb[1], rgb[2], alpha)

    else:
        rgba = []
        for _rgb in rgb:
            if type(rgb[0]) is tuple:
                rgba.append((rgb[0], rgb[1], rgb[2], alpha))
            else:
                rgba.append([rgb[0], rgb[1], rgb[2], alpha])

        if type(rgb) is tuple:
            rgba = tuple(rgba)
        elif type(rgb) is np.ndarray:
            rgba = np.array(rgba)

    return rgba


def get_total_measurements(loglist):
    """Get measurements per event"""

    total_measurements = []
    for _d in loglist:
        if _d is None:
            total_measurements.append(0)
        else:
            for _wave, _mdict in _d.items():
                sub_total = 0
                if _mdict is not None:
                    sub_total += _mdict["overall"]["windows"]

            total_measurements.append(sub_total)
    return np.array(total_measurements)


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


# class PlotStats(object):
#     """Plots statistics given the necessary variables."""

#     def __init__(self, ocmt=None, ncmt=None, dCMT=None, xcorr_mat=None,
#                  mean_mat=None, std_mat=None, stat_dict=None, labels=None,
#                  dlabels=None, stations=None, nbins=20, npar=9, savedir=None):
#         """
#         Parameters:
#             ocmt: old cmt matrix
#             ncmt: new cmt matrix
#             dCMT: diff cmt matrix
#             xcorr_mat: cross correlation matrix
#             mean_mat: mean vector
#             std_mat: std vector
#             labels: labels
#             dlabels: delta label
#             nbins: bins in the histograms
#             savedir: directory to save the figure output
#             verbose: verbosity

#         The matrices below should have following columns:
#             M0, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp,
#             depth, lat, lon, CMT, hdur, t_shift

#         Station list rows have following content
#             [network station latitude longitude elevation]
#         """

#         self.ocmt = ocmt
#         self.depth = ocmt[:, 7]
#         self.latitude = ocmt[:, 8]
#         self.longitude = ocmt[:, 9]
#         self.N = ocmt.shape[0]
#         self.ncmt = ncmt
#         self.dCMT = dCMT
#         self.xcorr_mat = xcorr_mat
#         self.mean_mat = mean_mat
#         self.std_mat = std_mat
#         self.labels = labels
#         self.dlabels = dlabels
#         self.stations = stations
#         self.stat_dict = stat_dict

#         # Save destination
#         self.savedir = savedir

#         # Fix depth
#         self.ocmt[:, 7] = self.ocmt[:, 7]/1000
#         self.ncmt[:, 7] = self.ncmt[:, 7]/1000
#         self.dCMT[:, 7] = self.dCMT[:, 7]/1000
#         self.mean_mat[7] = self.mean_mat[7]/1000
#         self.std_mat[7] = self.std_mat[7]/1000
#         # Min max ddepth for cmt plotting
#         self.maxddepth = np.max(self.dCMT[:, 7])
#         self.minddepth = np.min(self.dCMT[:, 7])
#         self.dd_absmax = np.max(np.abs(
#             [np.quantile(np.min(self.dCMT[:, 7]), 0.30),
#              np.quantile(np.min(self.dCMT[:, 7]), 0.70)]))
#         self.maxdepth = np.max(self.ocmt[:, 7])
#         self.mindepth = np.min(self.ocmt[:, 7])

#         self.nbins = nbins
#         self.dmbins = np.linspace(-0.5, 0.5 + 0.5 / self.nbins, self.nbins)
#         self.ddegbins = np.linspace(-0.1, 0.1 + 0.1 / self.nbins, self.nbins)
#         self.dzbins = np.linspace(-self.dd_absmax,
#                                   2 * self.dd_absmax / self.nbins, self.nbins)
#         self.dtbins = np.linspace(-10, 10 + 10 / self.nbins, self.nbins)

#         # Map characteristics
#         self.cl = 180.0  # central_longitude
#         self.cmt_cmap = matplotlib.colors.ListedColormap(
#             [(0.9, 0.9, 0.9), (0.7, 0.7, 0.7), (0.5, 0.5, 0.5),
#              (0.3, 0.3, 0.3), (0.1, 0.1, 0.1)])
#         self.depth_cmap = matplotlib.colors.ListedColormap(
#             ['#601A4A', '#EE442F', '#63ABCE', '#63ABCE'])
#         self.depth_cmap = matplotlib.colors.ListedColormap(
#             [(1.0, 96.0/255.0, 0.0), (0.0, 1.0, 1.0), (0.35, 0.35, 0.35),
#              (0.35, 0.35, 0.35)])
#         self.depth_bounds = [0, 70, 300, 800]
#         self.depth_norm = matplotlib.colors.BoundaryNorm(self.depth_bounds,
#                                                          self.depth_cmap.N)

#         # Measurement label and tag dictionary
#         self.vtype_dict = {r'Time-shift: ${\Delta t}_{CC}$': "tshift",
#                            r'$CC_{max} = _{max}|\frac{(d \star s)(\tau)}'
#                            r'{\sqrt{(d_i^2) * (s_i^2)}}|$': "cc",
#                            r'$P{L_1} = 10\log\left(\frac{d_i}'
#                            r'{s_i}\right)$ [dB]':
#                                "power_l1",
#                            r'$P{L_2} = 10\log\left(\frac{d_i^2}'
#                            r'{s_i^2}\right)$ [dB]':
#                                "power_l2",
#                            r'$P_{CC} = 10\log\frac{(d_i s_i)}'
#                            r'{s_i^2}$ [dB]': "cc_amp",
#                            r'$\frac{1}{2}\left|  d_i - s_i \right|^2$': "chi"}

#         # set_mpl_params_stats()

#     def plot_main_stats(self):
#         """Plots summary of the main statistics"""

#         # Create figure handle
#         fig = plt.figure(figsize=(11, 6))

#         # Create subplot layout
#         GS = GridSpec(3, 4)

#         cbar_dict = {"orientation": "horizontal",
#                      "shrink": 0.5,
#                      "spacing": 'proportional',
#                      "fraction": 0.025,
#                      "pad": 0.025}

#         # Create axis for map
#         ax = fig.add_subplot(GS[:2, :2],
#                              projection=PlateCarree(central_longitude=self.cl))
#         plt.subplots_adjust(left=0.1, bottom=0.1, right=0.1, top=0.1,
#                             wspace=0.1, hspace=0.1)
#         self.plot_map()
#         self.plot_cmts()
#         sm = matplotlib.cm.ScalarMappable(norm=self.depth_norm,
#                                           cmap=self.depth_cmap)
#         sm.set_array(self.depth_bounds)
#         sm.autoscale()
#         c = plt.colorbar(sm, ticks=self.depth_bounds, **cbar_dict)
#         c.set_label(r'Change in Depth [km]')
#         fontsize = 9
#         text_dict = {"fontsize": fontsize,
#                      "verticalalignment": 'top',
#                      "backgroundcolor": "white",
#                      "bbox": {"facecolor": 'white',
#                               "edgecolor": "black"},
#                      "zorder": 100}

#         ax.text(0.005, 0.995, "%d EQs" % self.N,
#                 **text_dict, horizontalalignment='left',
#                 transform=ax.transAxes)
#         self.print_figure_letter("a")

#         # Create axis for map
#         ax = fig.add_subplot(GS[:2, 2:],
#                              projection=PlateCarree(central_longitude=self.cl))
#         self.plot_map()
#         self.plot_stations()
#         # This is only done, so that both maps have the same aspect ratio
#         c = create_colorbar(vmin=0,
#                             vmax=800,
#                             cmap=self.depth_cmap,
#                             norm=self.depth_norm, **cbar_dict)
#         c.ax.set_visible(False)

#         ax.text(0.995, 0.995, "%d Stations" % len(self.stations),
#                 **text_dict, horizontalalignment='right',
#                 transform=ax.transAxes)
#         self.print_figure_letter("b")
#         # Change of parameter as function of depth
#         msize = 15

#         # Depth vs. change in depth
#         ax = fig.add_subplot(GS[2:3, 0])
#         plt.scatter(
#             self.dCMT[:, 7], self.ocmt[:, 7],
#             c=self.depth_cmap(self.depth_norm(self.ocmt[:, 7])),
#             s=msize, marker='o', alpha=0.5, edgecolors='none'
#             # markeredgecolor=rgb2rgba(self.depth_cmap(
#             #     self.depth_norm(self.ocmt[:, 7])), 0.5)
#         )
#         plt.plot([0, 0], [0, np.max(self.ocmt[:, 7])],
#                  "k--", lw=1.5)
#         plt.ylim(([0, np.max(self.ocmt[:, 7])]))
#         plt.xlim(([np.min(self.dCMT[:, 7]), np.max(self.dCMT[:, 7])]))
#         ax.invert_yaxis()
#         plt.xlabel("Depth Change [km]")
#         plt.ylabel("Depth [km]")
#         self.print_figure_letter("c")

#         fig.add_subplot(GS[2, 1])
#         self.plot_histogram(self.dCMT[:, 7], self.nbins)
#         remove_topright()
#         plt.xlabel("Depth Change [km]")
#         plt.ylabel("$N$", rotation=0, horizontalalignment='right')
#         self.print_figure_letter("d")

#         fig.add_subplot(GS[2, 2])
#         self.plot_histogram(self.dCMT[:, -1], self.nbins)
#         remove_topright()
#         plt.xlabel("Centroid Time Change [sec]")
#         plt.ylabel("$N$", rotation=0, horizontalalignment='right')
#         self.print_figure_letter("e")

#         fig.add_subplot(GS[2, 3])
#         self.plot_histogram(self.dCMT[:, 0]*100, self.nbins)
#         remove_topright()
#         plt.xlabel("Scalar Moment Change [%]")
#         plt.ylabel("$N$", rotation=0, horizontalalignment='right')
#         self.print_figure_letter("f")

#         # Finally plot shot
#         # plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.25)
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "main_stats.pdf"))
#         else:
#             plt.show()

#     def print_figure_letter(self, letter, fontsize=14):
#         ax = plt.gca()

#         text_dict = {"fontsize": fontsize,
#                      "verticalalignment": 'bottom',
#                      "horizontalalignment": 'left',
#                      "bbox": {'facecolor': 'white',
#                               'alpha': 0.0}
#                      }
#         ax.text(0.0, 1.025, letter, **text_dict, transform=ax.transAxes)

#     @staticmethod
#     def plot_scatter_hist(x, y, nbins, z=None, cmap=None, histc='grey',
#                           zmin=None, zmax=None, norm=None, r=True,
#                           xlog=False, ylog=False):
#         """

#         :param x: Data type x-axis
#         :param y: Data type y-axis
#         :params nbins: number of bins
#         :param z: Data type to color xy datapoints. Default None. Datapoints in
#                   color of histogram.
#         :param cmap: name
#         :param zmin: Colorbar min
#         :param zmin: Colorbar max

#         :return:
#         """

#         # definitions for the axes
#         left, width = 0.125, 0.65
#         bottom, height = 0.125, 0.65
#         spacing = 0.000

#         # Create dimensions
#         rect_scatter = [left, bottom, width, height]
#         rect_histx = [left, bottom + height + spacing, width, 0.15]
#         rect_histy = [left + width + spacing, bottom, 0.15, height]

#         # Create Axes
#         ax_scatter = plt.axes(rect_scatter)
#         ax_scatter.tick_params(direction='in', top=True, right=True)
#         ax_histx = plt.axes(rect_histx)
#         ax_histx.tick_params(direction='in', labelbottom=False)
#         ax_histy = plt.axes(rect_histy)
#         ax_histy.tick_params(direction='in', labelleft=False)

#         cax = ax_scatter.inset_axes([0.05, 0.96, 0.25, 0.03],
#                                     zorder=100)

#         # scatterplot with color
#         if cmap is not None and z is not None:
#             # the scatter plot:
#             if zmin is not None:
#                 vminz = zmin
#             else:
#                 vminz = np.min(z)

#             if zmax is not None:
#                 vmaxz = zmax
#             else:
#                 vmaxz = np.max(z)

#             if norm is not None and cmap is not None:
#                 ax_scatter.scatter(x, y, c=cmap(norm(z)),
#                                    s=20, marker='o', edgecolor='k',
#                                    linewidths=0.5)

#                 # Colorbar
#                 cbar_dict = {"orientation": "horizontal"}
#                 plt.colorbar(matplotlib.cm.ScalarMappable(
#                     cmap=cmap, norm=norm),
#                     cax=cax, **cbar_dict)
#                 cax.tick_params(left=False, right=False, bottom=True, top=True,
#                                 labelleft=False, labelright=False,
#                                 labeltop=False,
#                                 labelbottom=True, which='both',
#                                 labelsize=6)

#             else:
#                 ax_scatter.scatter(x, y, c=get_color(z, vmin=vminz, vmax=vmaxz,
#                                                      cmap=cmap, norm=norm),
#                                    s=20, marker='o', edgecolor='k',
#                                    linewidths=0.5)

#                 # Colorbar
#                 cbar_dict = {"orientation": "horizontal"}
#                 create_colorbar(vminz, vmaxz, cmap=cmap, cax=cax, **cbar_dict)
#                 cax.tick_params(left=False, right=False, bottom=True, top=True,
#                                 labelleft=False, labelright=False,
#                                 labeltop=False, labelbottom=True, which='both',
#                                 labelsize=6)

#         # scatterplot without color
#         else:
#             # the scatter plot:
#             ax_scatter.scatter(x, y, c=histc, s=15, marker='s')
#             cax = None

#         if r:
#             if xlog:
#                 xfix = np.log10(x)
#             else:
#                 xfix = x
#             if ylog:
#                 yfix = np.log10(y)
#             else:
#                 yfix = y
#             # Write out correlation coefficient in the top right
#             corr_coeff = np.corrcoef(xfix, yfix)
#             text_dict = {"fontsize": 6, "verticalalignment": 'top',
#                          "zorder": 100}
#             ax_scatter.text(0.97, 0.97, "R = %1.2f" % corr_coeff[0, 1],
#                             horizontalalignment='right', **text_dict,
#                             transform=ax_scatter.transAxes)

#         # now determine nice limits by hand:
#         ax_scatter.set_xlim((np.min(x), np.max(x)))
#         ax_scatter.set_ylim((np.min(y), np.max(y)))

#         # Histogram settings
#         binsx = np.linspace(np.min(x), np.max(x), nbins + 1)
#         binsy = np.linspace(np.min(y), np.max(y), nbins + 1)
#         ax_histx.hist(x, bins=binsx,
#                       color=histc, ec=None)
#         ax_histx.set_xlim(ax_scatter.get_xlim())
#         ax_histy.hist(y, bins=binsy, orientation='horizontal',
#                       color=histc, ec=None)
#         ax_histy.set_ylim(ax_scatter.get_ylim())

#         # Remove boundaries
#         remove_all(ax=ax_histx, bottom=True)
#         remove_all(ax=ax_histy, left=True)

#         if cax is not None:
#             return ax_scatter, ax_histx, ax_histy, cax
#         else:
#             return ax_scatter, ax_histx, ax_histy

#     def plot_dM_dz_nz(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.dCMT[:, 7], self.dCMT[:, 0] * 100, self.nbins,
#             z=self.ncmt[:, 7], cmap=self.depth_cmap, histc='grey',
#             zmin=None, zmax=None, norm=self.depth_norm)
#         ax_scatter.set_xlabel(r"Depth change [km]")
#         ax_scatter.set_ylabel(r"Scalar Moment Change [%]")
#         ax_scatter.plot([np.min(self.dCMT[:, 7]),
#                          np.max(self.dCMT[:, 7])],
#                         [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.plot([0, 0], [np.min(self.dCMT[:, 0] * 100),
#                                  np.max(self.dCMT[:, 0] * 100)],
#                         'k', zorder=0.1, lw=0.75)

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dM_dz_nz.pdf"))
#         else:
#             plt.show()

#     def plot_dM_dz_oz(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.dCMT[:, 7], self.dCMT[:, 0] * 100, self.nbins,
#             z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
#             zmin=None, zmax=None, norm=self.depth_norm)
#         ax_scatter.set_xlabel(r"Depth change [km]")
#         ax_scatter.set_ylabel(r"Scalar Moment Change [%]")
#         ax_scatter.plot([np.min(self.dCMT[:, 7]),
#                          np.max(self.dCMT[:, 7])],
#                         [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.plot([0, 0], [np.min(self.dCMT[:, 0] * 100),
#                                  np.max(self.dCMT[:, 0] * 100)],
#                         'k', zorder=0.1, lw=0.75)

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dM_dz_oz.pdf"))
#         else:
#             plt.show()

#     def plot_dM_oz_dz(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.ocmt[:, 7], self.dCMT[:, 0] * 100, self.nbins,
#             z=self.dCMT[:, 7], cmap=self.cmt_cmap, histc='grey',
#             zmin=-self.dd_absmax, zmax=self.dd_absmax,
#             xlog=False)
#         ax_scatter.plot([1, 800], [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.set_xlabel(r"New Depth [km]")
#         ax_scatter.set_ylabel(r"Scalar Moment Change [%]")
#         ax_scatter.set_xscale('log')
#         ax_scatter.set_xlim([10, self.maxdepth])
#         # ax_scatter.invert_xaxis()
#         ax_histx.set_xscale(ax_scatter.get_xscale())
#         ax_histx.set_xlim(ax_scatter.get_xlim())

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dM_oz_dz.pdf"))
#         else:
#             plt.show()

#     def plot_dM_nz_dz(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.ncmt[:, 7], self.dCMT[:, 0] * 100, self.nbins,
#             z=self.dCMT[:, 7], cmap=self.cmt_cmap, histc='grey',
#             zmin=-self.dd_absmax, zmax=self.dd_absmax,
#             xlog=False)
#         ax_scatter.plot([1, 800], [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.set_xlabel(r"New Depth [km]")
#         ax_scatter.set_ylabel(r"Scalar Moment Change [%]")
#         ax_scatter.set_xscale('log')
#         ax_scatter.set_xlim([10, self.maxdepth])
#         # ax_scatter.invert_xaxis()
#         ax_histx.set_xscale(ax_scatter.get_xscale())
#         ax_histx.set_xlim(ax_scatter.get_xlim())

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dM_nz_dz.pdf"))
#         else:
#             plt.show()

#     def plot_dt_oz_dz(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.ocmt[:, 7], self.dCMT[:, -1], self.nbins,
#             z=self.dCMT[:, 7], cmap=self.cmt_cmap, histc='grey',
#             zmin=-self.dd_absmax, zmax=self.dd_absmax,
#             xlog=False)
#         ax_scatter.set_xlabel(r"Depth [km]")
#         ax_scatter.set_ylabel(r"Centroid Time Change [s]")
#         ax_scatter.plot([1, 800], [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.set_xscale('log')
#         ax_scatter.set_xlim([10, self.maxdepth])
#         ax_histx.set_xscale(ax_scatter.get_xscale())
#         ax_histx.set_xlim(ax_scatter.get_xlim())

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dt_oz_dz.pdf"))
#         else:
#             plt.show()

#     def plot_dt_nz_dz(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.ncmt[:, 7], self.dCMT[:, -1], self.nbins,
#             z=self.dCMT[:, 7], cmap=self.cmt_cmap, histc='grey',
#             zmin=-self.dd_absmax, zmax=self.dd_absmax,
#             xlog=False)
#         ax_scatter.set_xlabel(r"New Depth [km]")
#         ax_scatter.set_ylabel(r"Centroid Time Change [s]")
#         ax_scatter.plot([1, 800], [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.set_xscale('log')
#         ax_scatter.set_xlim([10, self.maxdepth])
#         ax_histx.set_xscale(ax_scatter.get_xscale())
#         ax_histx.set_xlim(ax_scatter.get_xlim())

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dt_nz_dz.pdf"))
#         else:
#             plt.show()

#     def plot_dz_nz_dM(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.ncmt[:, 7], self.dCMT[:, 7], self.nbins,
#             z=self.dCMT[:, 0] * 100, cmap=matplotlib.cm.get_cmap("PiYG"),
#             histc='grey', zmin=-7.5, zmax=7.5, xlog=False)
#         ax_scatter.plot([1, 800], [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.set_xlabel(r"New Depth [km]")
#         ax_scatter.set_ylabel(r"Depth Change [km]")
#         ax_scatter.set_xscale('log')
#         ax_scatter.set_xlim([10, self.maxdepth])
#         # ax_scatter.invert_xaxis()
#         ax_histx.set_xscale(ax_scatter.get_xscale())
#         ax_histx.set_xlim(ax_scatter.get_xlim())

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dz_nz_dM.pdf"))
#         else:
#             plt.show()

#     def plot_dz_oz_dM(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.ocmt[:, 7], self.dCMT[:, 7], self.nbins,
#             z=self.dCMT[:, 0] * 100, cmap=matplotlib.cm.get_cmap("PiYG"),
#             histc='grey', zmin=-7.5, zmax=7.5, xlog=False)
#         ax_scatter.plot([1, 800], [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.set_xlabel(r"Old Depth [km]")
#         ax_scatter.set_ylabel(r"Depth Change [km]")
#         ax_scatter.set_xscale('log')
#         ax_scatter.set_xlim([10, self.maxdepth])
#         # ax_scatter.invert_xaxis()
#         ax_histx.set_xscale(ax_scatter.get_xscale())
#         ax_histx.set_xlim(ax_scatter.get_xlim())

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "dz_oz_dM.pdf"))
#         else:
#             plt.show()

#     def plot_z_z_dM(self):
#         """Creates Figure with histograms one the side for two
#         change in depth and change in scalar moment."""

#         # start with a rectangular Figure
#         plt.figure(figsize=(4, 4))

#         ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
#             self.ncmt[:, 7], self.ocmt[:, 7], self.nbins,
#             z=self.dCMT[:, 0] * 100, cmap=matplotlib.cm.get_cmap("PiYG"),
#             histc='grey', zmin=-7.5, zmax=7.5)
#         ax_scatter.plot([1, 800], [0, 0], 'k', zorder=-1, lw=0.75)
#         ax_scatter.set_xlabel(r"New Depth [km]")
#         ax_scatter.set_ylabel(r"Old Depth [km]")
#         ax_scatter.set_xscale('log')
#         ax_scatter.set_xlim([10, self.maxdepth])
#         # ax_scatter.invert_xaxis()
#         ax_histx.set_xscale(ax_scatter.get_xscale())
#         ax_histx.set_xlim(ax_scatter.get_xlim())

#         ax_scatter.set_yscale('log')
#         ax_scatter.set_ylim([10, self.maxdepth])
#         # ax_scatter.invert_xaxis()
#         ax_histy.set_yscale(ax_scatter.get_yscale())
#         ax_histy.set_ylim(ax_scatter.get_ylim())

#         # Finally plot shot
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "nz_oz_dM.pdf"))
#         else:
#             plt.show()

#     def plot_changes(self):
#         """Plots figure with statistics."""

#         # Create figure handle
#         fig = plt.figure(figsize=(11, 10))

#         # Create subplot layout
#         GS = GridSpec(6, 6)

#         # Create axis for map
#         fig.add_subplot(GS[:2, :3],
#                         projection=PlateCarree(central_longitude=180.0))
#         self.plot_map()
#         self.plot_cmts()
#         plt.title("Inversion statistics for %d earthquakes" % (self.N))

#         # Create axis for map
#         fig.add_subplot(GS[2:4, :3],
#                         projection=PlateCarree(central_longitude=180.0))
#         self.plot_map()
#         self.plot_stations()
#         plt.title("Stations used in the inversion")

#         # table axes
#         # fig.add_subplot(GS[2:4, 3])
#         # self.plot_table()

#         # change in cmttime
#         fig.add_subplot(GS[2, 3])
#         self.plot_histogram(self.dCMT[:, 10], self.dtbins)
#         plt.xlabel(r"$\delta t$")

#         # MT
#         counter = 1
#         for _i in range(2):
#             for _j in range(3):
#                 fig.add_subplot(GS[0 + _i, 3 + _j])
#                 self.plot_histogram(self.dCMT[:, counter],
#                                     self.dmbins, facecolor=(0.8, 0.8, 0.8))
#                 plt.xlabel("%s" % (self.dlabels[counter]))
#                 counter += 1

#         # loc_ax
#         fig.add_subplot(GS[3, 4])
#         self.plot_histogram(self.dCMT[:, 8], self.ddegbins)
#         plt.xlabel("$\\delta$Lat [$^{\\circ}$]")
#         fig.add_subplot(GS[3, 5])
#         self.plot_histogram(self.dCMT[:, 9], self.ddegbins)
#         plt.xlabel("$\\delta$Lon [$^{\\circ}$]")
#         fig.add_subplot(GS[2, 4])
#         self.plot_histogram(self.dCMT[:, 7], self.dzbins)
#         plt.xlabel("$\\delta z$ [km]")
#         fig.add_subplot(GS[2, 5])
#         self.plot_histogram(self.dCMT[:, 0], self.dmbins)
#         plt.xlabel("$\\delta M_0$")

#         vmin = 10**(25.75)
#         vmax = 10**(26.5)
#         msize = 10

#         # Change of parameter as function of depth
#         fig.add_subplot(GS[4:, 0:2])  # moment vs depth
#         sc = plt.scatter(self.dCMT[:, 0], self.ocmt[:, 7], c=self.ocmt[:, 0],
#                          s=msize, marker='o', cmap=cm.rainbow,
#                          norm=colors.LogNorm(vmin=vmin, vmax=vmax))
#         cbar = plt.colorbar(sc, orientation="horizontal")
#         cbar.ax.set_ylabel(r"$M_0$")
#         plt.xlim([-0.5, 0.5])
#         plt.gca().invert_yaxis()
#         plt.xlabel("$\\delta M_0$")
#         plt.ylabel("$z$ [km]")

#         # ddepth vs depth
#         fig.add_subplot(GS[4:, 2:4])
#         sc1 = plt.scatter(self.dCMT[:, 7], self.ocmt[:, 7], c=self.ocmt[:, 0],
#                           s=msize, marker='o', cmap=cm.rainbow,
#                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))
#         cbar = plt.colorbar(sc1, orientation="horizontal")
#         cbar.ax.set_ylabel(r"$M_0$")
#         plt.xlim([-20, 10])
#         plt.xlabel(r"$\delta z$ [km]")
#         plt.ylabel(r"$z$ [km]")
#         plt.gca().invert_yaxis()

#         # ddepth vs dM0
#         fig.add_subplot(GS[4:, 4:])
#         sc2 = plt.scatter(self.dCMT[:, 0], self.dCMT[:, 7], c=self.ocmt[:, 7],
#                           s=msize, marker='o', cmap=cm.rainbow,
#                           norm=colors.LogNorm())
#         cbar = plt.colorbar(sc2, orientation="horizontal")
#         cbar.ax.set_ylabel(r"$z$ [km]")
#         plt.xlim([-0.5, 0.5])
#         plt.ylim([-20, 10])
#         plt.ylabel(r"$\delta z$ [km]")
#         plt.xlabel(r"$\delta M_0$")
#         plt.gca().invert_yaxis()

#         # Finally plot shot
#         plt.tight_layout()
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "statfigure.pdf"))
#         else:
#             plt.show()

#     def plot_map(self):

#         ax = plt.gca()
#         ax.set_global()
#         ax.frameon = True
#         ax.outline_patch.set_linewidth(0.75)

#         # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
#         # function around 180deg
#         gl = ax.gridlines(crs=PlateCarree(central_longitude=180.0),
#                           draw_labels=False,
#                           linewidth=1, color='lightgray', alpha=0.5,
#                           linestyle='-', zorder=-1.5)
#         gl.top_labels = False
#         gl.left_labels = False
#         gl.xlines = True

#         # Add Coastline
#         ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
#                        linewidth=0.5, facecolor=(0.9, 0.9, 0.9))

#     def plot_cmts(self):

#         ax = plt.gca()
#         for idx, (lon, lat, m) in enumerate(zip(self.longitude, self.latitude,
#                                                 self.ncmt[:, 1:7])):
#             try:
#                 # Longitude fix because cartopy is being shitty
#                 if self.cl == 180.0:
#                     if lon <= 0:
#                         lon = lon + 180.0
#                     else:
#                         lon = lon - 180.0
#                 b = beach(m, linewidth=0.25,
#                           facecolor=self.depth_cmap(self.depth_norm(
#                               self.ocmt[idx, 7]
#                           )),
#                           # get_color(self.ocmt[idx, 7],
#                           #                     cmap=self.depth_cmap,
#                           #                     vmin=0,
#                           #                     vmax=800,
#                           #                     norm=self.depth_norm),
#                           bgcolor='w',
#                           edgecolor='k', alpha=1,
#                           xy=(lon, lat), width=10,
#                           size=10, nofill=False, zorder=-1)

#                 ax.add_collection(b)
#             except Exception as e:
#                 for line in e.__str__().splitlines():
#                     logger.error(line)

#     def plot_stations(self):
#         """Plots stations into a map
#         """

#         slat = [station[0] for station in self.stations]
#         # Weird fix because cartopy is weird
#         if self.cl == 180.0:
#             slon = [station[1] + self.cl if station[1] <= 0
#                     else station[1] - self.cl
#                     for station in self.stations]
#         else:
#             slon = [station[1] for station in self.stations]

#         ax = plt.gca()
#         ax.scatter(slon, slat, s=20, marker='v', c=((0.7, 0.2, 0.2),),
#                    edgecolors='k', linewidths=0.25, zorder=-1)

#     def plot_histogram(self, ddata, n_bins, facecolor=(0.7, 0.2, 0.2),
#                        alpha=1):
#         """Plots histogram of input data."""

#         # the histogram of the data
#         ax = plt.gca()
#         ax.hist(ddata, n_bins, facecolor=facecolor, alpha=alpha)

#     def plot_xcorr_matrix(self):
#         """Plots Corrlation matrix with approximate correlation bars
#         """

#         fig = plt.figure(figsize=(12, 11))

#         ax = fig.subplots(10, 10, sharex="col", sharey='row', squeeze=True,
#                           gridspec_kw={'hspace': 0, 'wspace': 0})

#         for _i in range(10):
#             for _j in range(10):

#                 plt.sca(ax[_i][_j])
#                 if _j == _i:
#                     shay = ax[_i][_j].get_shared_y_axes()
#                     shay.remove(ax[_i][_j])
#                     self.plot_histogram(self.dCMT[:, _i], self.nbins)
#                 else:
#                     ax[_i][_j].plot(self.dCMT[:, _j],  self.dCMT[:, _i],
#                                     'ko', markersize=2)

#                     # OLS fit
#                     A = np.vstack([self.dCMT[:, _j],
#                                    np.ones(len(self.dCMT[:, _j]))]).T

#                     m, c = np.linalg.lstsq(A, self.dCMT[:, _i], rcond=None)[0]

#                     res = np.sqrt(np.sum(((c + m * self.dCMT[:, _j])
#                                           - self.dCMT[:, _j]) ** 2))

#                     print(res, np.sqrt(self.mean_mat[_j]**2
#                                        + self.mean_mat[_i]**2))

#                     if res < 0.25 * self.N*np.sqrt(self.mean_mat[_j]**2
#                                                    + self.mean_mat[_i]**2):
#                         # Different option to compute the OLS fit by computing
#                         # the perpendicular distance
#                         # m, c = fit_xy(self.dCMT[:, _j], self.dCMT[:, _i])

#                         # Plot polyline
#                         ax[_i][_j].plot(self.dCMT[:, _j],
#                                         c + m * self.dCMT[:, _j],
#                                         '-', c=(0.85, 0.2, 0.2))

#         plt.tight_layout()
#         for _i in range(10):
#             for _j in range(10):
#                 if _i == 9:
#                     ax[_i][_j].set_xlabel(self.dlabels[_j])
#                 if _j == 0:
#                     ax[_i][_j].set_ylabel(self.dlabels[_i])
#                     ax[_i][_j].yaxis.set_label_coords(-0.3, 0.5)

#                     # Label magic happens here
#                     if _i in [0, 1, 2, 3, 4, 5, 6]:
#                         # Change the ticklabel format to scientific format
#                         ax[_i][_j].ticklabel_format(axis="y", style='sci')
#                         offset_text = \
#                             ax[_i][_j].yaxis.get_offset_text().get_text()
#                         ax[_i][_j].text(-.375, 0.85, offset_text,
#                                         rotation='vertical',
#                                         ha='center', va='center',
#                                         transform=ax[_i][_j].transAxes)
#                         ax[_i][_j].yaxis.get_offset_text().set_visible(False)

#         # Finally plot shot
#         plt.tight_layout()
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "xcorr.pdf"))
#         else:
#             plt.show()

#     def plot_measurement_changes(self):
#         """ Uses the stat_dict to compute measurement statistics.

#         """

#         # Create figure
#         if self.stat_dict is None:
#             logger.info("No statistics dictionary here...")
#             return

#         # Get number of rows and columns
#         nrows = len(self.stat_dict)
#         ncols = len(self.vtype_dict)

#         stats = ["mean", "std"]
#         # Create figure

#         for _stat in stats:

#             fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
#             G = GridSpec(nrows, ncols)

#             for irow, cat in enumerate(self.stat_dict.keys()):
#                 for icol, (label, vtype) in enumerate(self.vtype_dict.items()):
#                     ax = fig.add_subplot(G[irow, icol])
#                     if irow == 0 and icol == 0:
#                         fontsize = 20
#                         ax.text(0.025, 0.975, _stat,
#                                 fontsize=fontsize,
#                                 horizontalalignment='left',
#                                 verticalalignment='top',
#                                 transform=ax.transAxes)

#                     self._hist_sub(self.stat_dict[cat][vtype], cat, vtype,
#                                    label, _stat, self.nbins)

#             plt.tight_layout()
#             plt.savefig(os.path.join(self.savedir,
#                                      "measurement_changes_" + _stat + ".pdf"))
#             plt.close(fig)

#     def plot_mean_measurement_change_stats(self):
#         """ Uses the stat_dict to compute measurement statistics.

#         """

#         # Create figure
#         if self.stat_dict is None:
#             logger.info("No statistics dictionary here...")
#             return

#         # Get number of rows and columns
#         nrows = len(self.stat_dict)
#         ncols = len(self.vtype_dict)

#         # Create figure
#         fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
#         G = GridSpec(nrows, ncols)

#         for irow, cat in enumerate(self.stat_dict.keys()):
#             for icol, (label, vtype) in enumerate(self.vtype_dict.items()):
#                 ax = fig.add_subplot(G[irow, icol])
#                 if irow == 0 and icol == 0:
#                     fontsize = 20
#                     ax.text(0.025, 0.975, r"$\Delta$Mean",
#                             fontsize=fontsize,
#                             horizontalalignment='left',
#                             verticalalignment='top',
#                             transform=ax.transAxes)

#                 self._hist_sub_change(self.stat_dict[cat][vtype], cat, vtype,
#                                       label, self.nbins)

#         plt.tight_layout()
#         plt.savefig(os.path.join(self.savedir,
#                                  "mean_measurement_change_stats.pdf"))
#         plt.close(fig)

#     @staticmethod
#     def _hist_sub(measurement_dict, cat, vtype, label, stat, num_bin):
#         """

#         :param measurement_dict: {after: {mean: list, std: list},
#                                   before: {mean: list, std: list}}
#         :param label:
#         :param stat:
#         :return:
#         """
#         # Get axes
#         ax = plt.gca()

#         plt.xlabel(label, fontsize=15)
#         plt.ylabel(r"%s" % cat.replace("_", r"\_"), fontsize=15)

#         data_b = measurement_dict['before'][stat]
#         data_a = measurement_dict['after'][stat]

#         if vtype == "cc":
#             ax_min = min(min(data_b), min(data_a))
#             ax_max = max(max(data_b), max(data_a))
#         elif vtype == "chi":
#             ax_min = 0.0
#             ax_max = max(max(data_b), max(data_a))
#         else:
#             ax_min = min(min(data_b), min(data_a))
#             ax_max = max(max(data_b), max(data_a))
#             abs_max = max(abs(ax_min), abs(ax_max))
#             ax_min = -abs_max
#             ax_max = abs_max

#         if stat == "std":
#             ax_min = 0.0

#         binwidth = (ax_max - ax_min) / num_bin

#         # Stats
#         a_mean = np.mean(data_a)
#         a_std = np.std(data_a)
#         b_mean = np.mean(data_b)
#         b_std = np.std(data_b)

#         nb, _, _ = ax.hist(
#             data_b, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
#             facecolor='blue', alpha=0.3)
#         nb_max = np.max(nb)
#         ax.plot([b_mean, b_mean], [0, nb_max], "b--")
#         ax.plot([b_mean - b_std, b_mean + b_std],
#                 [nb_max / 2, nb_max / 2], "b--")
#         na, _, _ = ax.hist(
#             data_a, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
#             facecolor='red', alpha=0.5)
#         na_max = np.max(na)
#         ax.plot([a_mean, a_mean], [0, na_max], "r-")
#         ax.plot([a_mean - a_std, a_mean + a_std],
#                 [na_max / 2, na_max / 2], "r-")

#     @staticmethod
#     def _hist_sub_change(measurement_dict, cat, vtype, label, num_bin):
#         """

#         :param measurement_dict: {after: {mean: list, std: list},
#                                   before: {mean: list, std: list}}
#         :param label:
#         :param stat:
#         :return:
#         """
#         # Get axes
#         ax = plt.gca()

#         plt.xlabel(label, fontsize=15)
#         plt.ylabel(r"%s" % cat.replace("_", r"\_"), fontsize=15)

#         # Compute data change
#         data_b = measurement_dict['before']["mean"]
#         data_a = measurement_dict['after']["mean"]
#         data = (np.array(data_a) - np.array(data_b)).tolist()

#         # Compute plot axes and bins
#         ax_min = min(data)
#         ax_max = max(data)
#         abs_max = max(abs(ax_min), abs(ax_max))
#         ax_min = -abs_max
#         ax_max = abs_max
#         binwidth = (ax_max - ax_min) / num_bin

#         # Stats
#         d_mean = np.mean(data)
#         d_std = np.std(data)

#         nd, _, _ = ax.hist(
#             data, bins=np.arange(ax_min, ax_max + binwidth / 2., binwidth),
#             facecolor='black', alpha=0.3)
#         nd_max = np.max(nd)
#         ax.plot([d_mean, d_mean], [0, nd_max], "b-")
#         ax.plot([d_mean - d_std, d_mean + d_std],
#                 [nd_max / 2, nd_max / 2], "b-")

#     def plot_xcorr_heat(self):
#         """Plots correlatio n heatmap.
#         """
#         fig = plt.figure(figsize=(12, 10))

#         ax = fig.add_subplot(111)

#         x = np.arange(10)
#         y = np.arange(10)
#         xx, yy = np.meshgrid(x, y)

#         cmap = plt.get_cmap("coolwarm")
#         size_scale = 750
#         scat = ax.scatter(xx, yy, s=(np.abs(self.xcorr_mat[:10, :10])
#                                      * size_scale),
#                           c=self.xcorr_mat[:10, :10], cmap=cmap, marker='s',
#                           alpha=0.75)

#         cbar = plt.colorbar(scat, aspect=50, pad=0)
#         cbar.ax.tick_params(labelsize=12)

#         # Mapping from column names to integer coordinates
#         plt.xticks(np.arange(10), self.dlabels[:10], fontsize=13)
#         plt.yticks(np.arange(10), self.dlabels[:10], fontsize=13)

#         # Relocate Grid
#         ax.grid(False, 'major')
#         ax.grid(True, 'minor')
#         ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
#         ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

#         # Set limits
#         ax.set_xlim([-0.5, max(x) + 0.5])
#         ax.set_ylim([-0.5, max(x) + 0.5])

#         # Invert y and put xaxis on top
#         ax.invert_yaxis()
#         ax.xaxis.tick_top()

#         # Finally plot shot
#         plt.tight_layout()
#         if self.savedir is not None:
#             plt.savefig(os.path.join(self.savedir, "xcorrheat.pdf"))
#         else:
#             plt.show()

#     def plot_table(self):
#         """Plots minimal summary"""

#         columns = (r'$\mu$', r'$\sigma$')
#         rows = [r'$\delta t$', r'$\delta$Lat', r'$\delta$Lon', r'$\delta z$',
#                 r'$\delta M_0$', r"$\delta M_{rr}$", r"$\delta M_{tt}$",
#                 r"$\delta M_{pp}$", r"$\delta M_{rt}$", r"$\delta M_{rp}$",
#                 r"$\delta M_{tp}$"]

#         cell_text = []

#         # dt
#         cell_text.append(["%3.3f" % (self.mean_mat[10]),
#                           "%3.3f" % (self.std_mat[10])])
#         # dLat
#         cell_text.append(["%3.3f" % (self.mean_mat[8]),
#                           "%3.3f" % (self.std_mat[8])])
#         # dLon
#         cell_text.append(["%3.3f" % (self.mean_mat[9]),
#                           "%3.3f" % (self.std_mat[9])])
#         # dz
#         cell_text.append(["%3.3f" % (self.mean_mat[7]),
#                           "%3.3f" % (self.std_mat[7])])
#         # M0
#         cell_text.append(["%3.3f" % (self.mean_mat[0]),
#                           "%3.3f" % (self.std_mat[0])])
#         for _j in range(6):
#             cell_text.append(["%3.3f" % (self.mean_mat[1 + _j]),
#                               "%3.3f" % (self.std_mat[1 + _j])])

#         # Plot table
#         ax = plt.gca()
#         ax.axis('tight')
#         ax.axis('off')
#         ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns,
#                  loc='center', edges='horizontal', fontsize=13)

#     def save_table(self):
#         """Uses plot_table function to save figure"""

#         # Create figure handle
#         fig = plt.figure(figsize=(4, 2))

#         # Create subplot layout
#         self.plot_table()

#         plt.savefig(os.path.join(self.savedir, "summary_table.pdf"))
#         plt.close(fig)


class PlotCatalogStatistics(object):

    def __init__(self, event, ocmt, ncmt, dcmt, xcorr_mat, angles,
                 ocmtfiles, ncmtfiles, mean_mat, mean_dabs,
                 std_mat, stations, bounds, labels, dlabels, tags, factor,
                 units, measurements=None, outdir="./",
                 prefix: None or str = None,
                 cmttime: bool = False, hdur: bool = False, nbins=40):

        self.event = event
        self.N = len(event)
        self.ocmt = ocmt
        self.ncmt = ncmt
        self.dcmt = dcmt
        self.xcorr_mat = xcorr_mat
        self.angles = angles
        self.ocmtfiles = ocmtfiles
        self.ncmtfiles = ncmtfiles
        self.mean_mat = mean_mat
        self.mean_dabs = mean_dabs
        self.std_mat = std_mat
        self.stations = stations
        self.bounds = bounds
        self.factor = factor
        self.labels = labels
        self.dlabels = dlabels
        self.tags = tags
        self.units = units
        self.measurements = measurements
        self.prefix = prefix
        self.outdir = outdir

        # Min max ddepth for cmt plotting
        self.maxddepth = np.max(self.dcmt[:, 7])
        self.minddepth = np.min(self.dcmt[:, 7])
        self.dd_absmax = np.max(np.abs(
            [np.quantile(np.min(self.dcmt[:, 7]), 0.30),
             np.quantile(np.min(self.dcmt[:, 7]), 0.70)]))
        self.maxdepth = np.max(self.ocmt[:, 7])
        self.mindepth = np.min(self.ocmt[:, 7])

        self.nbins = nbins
        self.dmbins = np.linspace(-0.5, 0.5 + 0.5 / self.nbins, self.nbins)
        self.ddegbins = np.linspace(-0.1, 0.1 + 0.1 / self.nbins, self.nbins)
        self.dzbins = np.linspace(-self.dd_absmax,
                                  2 * self.dd_absmax / self.nbins, self.nbins)
        self.dtbins = np.linspace(-10, 10 + 10 / self.nbins, self.nbins)

        # Map characteristics
        self.cl = 180.0  # central_longitude
        self.cmt_cmap = matplotlib.colors.ListedColormap(
            [(0.9, 0.9, 0.9), (0.7, 0.7, 0.7), (0.5, 0.5, 0.5),
             (0.3, 0.3, 0.3), (0.1, 0.1, 0.1)])
        self.depth_cmap = matplotlib.colors.ListedColormap(
            [(0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.35, 0.35, 0.35),
             (0.35, 0.35, 0.35)])
        self.depth_bounds = [0, 70, 300, 800]
        self.depth_norm = matplotlib.colors.BoundaryNorm(self.depth_bounds,
                                                         self.depth_cmap.N)

        self.lettergen = figletter()
        self.figletter = lambda: next(self.lettergen)
        self.abc = 'abcdefghijklmopqrstuvw'

    def plot_main_stats(self):
        """Plots summary of the main statistics"""

        # Create figure handle
        fig = plt.figure(figsize=(11, 5))

        # Create subplot layout
        GS = GridSpec(3, 4)

        # cbar_dict = {"orientation": "horizontal",
        #              "shrink": 0.5,
        #              "spacing": 'proportional',
        #              "fraction": 0.025,
        #              "pad": 0.025}

        # Create axis for map
        ax = fig.add_subplot(GS[:2, :2],
                             projection=PlateCarree(central_longitude=self.cl))
        plt.subplots_adjust(left=0.06, bottom=0.1, right=0.94, top=0.94,
                            wspace=0.35, hspace=0.25)
        self.plot_map()
        self.plot_cmts()
        # sm = matplotlib.cm.ScalarMappable(norm=self.depth_norm,
        #                                   cmap=self.depth_cmap)
        # sm.set_array(self.depth_bounds)
        # sm.autoscale()
        # c = plt.colorbar(sm, ticks=self.depth_bounds, **cbar_dict)
        # c.set_label(r'Change in Depth [km]')

        text_dict = {"fontsize": 'small',
                     "verticalalignment": 'top',
                     "backgroundcolor": "white",
                     "bbox": {"facecolor": 'white',
                              "edgecolor": "black"},
                     "zorder": 100}

        ax.text(0.005, 0.995, "%d EQs" % self.N,
                **text_dict, horizontalalignment='left',
                transform=ax.transAxes)
        self.print_figure_letter(self.figletter())

        # Create axis for map
        ax = fig.add_subplot(GS[:2, 2:],
                             projection=PlateCarree(central_longitude=self.cl))
        self.plot_map()
        self.plot_stations()
        # # This is only done, so that both maps have the same aspect ratio
        # c = create_colorbar(vmin=0,
        #                     vmax=800,
        #                     cmap=self.depth_cmap,
        #                     norm=self.depth_norm, **cbar_dict)
        # c.ax.set_visible(False)

        ax.text(0.995, 0.995, "%d Stations" % len(self.stations),
                **text_dict, horizontalalignment='right',
                transform=ax.transAxes)
        self.print_figure_letter(self.figletter())
        # Change of parameter as function of depth
        msize = 15

        # Depth vs. change in depth
        ax = fig.add_subplot(GS[2:3, 0])
        plt.scatter(
            self.dcmt[:, 7], self.ocmt[:, 7],
            c=self.depth_cmap(self.depth_norm(self.ocmt[:, 7])),
            s=msize, marker='o', alpha=0.5, edgecolors='none')

        # Custom legend
        classes = ['  <70 km', ' ', '>300 km']
        colors = [(0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.35, 0.35, 0.35)]
        for cla, col in zip(classes, colors):
            plt.scatter([], [], c=[col], s=msize, label=cla, alpha=0.5,
                        edgecolors='none')
        plt.legend(loc='lower left', frameon=False, fancybox=False,
                   numpoints=1, scatterpoints=1, fontsize='xx-small',
                   borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
                   labelspacing=0.2, handlelength=0.5,
                   bbox_to_anchor=(0.0, 0.0))

        # Zero line
        plt.plot([0, 0], [0, np.max(self.ocmt[:, 7])],
                 "k--", lw=1.5)
        plt.ylim(([np.min(self.ocmt[:, 7]), np.max(self.ocmt[:, 7])]))
        plt.xlim(([np.min(self.dcmt[:, 7]), np.max(self.dcmt[:, 7])]))
        ax.invert_yaxis()
        plt.xlabel("Depth Change [km]")
        plt.ylabel("Depth [km]")
        self.print_figure_letter(self.figletter())

        fig.add_subplot(GS[2, 1])
        self.plot_histogram(self.dcmt[:, 7], self.nbins, facecolor='lightgray',
                            statsleft=True)
        remove_topright()
        plt.xlabel("Depth Change [km]")
        plt.ylabel("N", rotation=0, horizontalalignment='right')
        self.print_figure_letter(self.figletter())

        fig.add_subplot(GS[2, 2])
        self.plot_histogram(
            self.dcmt[:, -1], self.nbins, facecolor='lightgray')
        remove_topright()
        plt.xlabel("Centroid Time Change [sec]")
        plt.ylabel("N", rotation=0, horizontalalignment='right')
        self.print_figure_letter(self.figletter())

        fig.add_subplot(GS[2, 3])
        self.plot_histogram(self.dcmt[:, 0]*100,
                            self.nbins, facecolor='lightgray')
        remove_topright()
        plt.xlabel("Scalar Moment Change [%]")
        plt.ylabel("N", rotation=0, horizontalalignment='right')
        self.print_figure_letter(self.figletter())

        # Finally plot shot
        # plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.25)

        filename = "main_stats.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def print_figure_letter(self, letter, fontsize='large'):
        ax = plt.gca()

        text_dict = {"fontsize": fontsize,
                     "verticalalignment": 'bottom',
                     "horizontalalignment": 'left',
                     "bbox": {'facecolor': 'white',
                              'alpha': 0.0}
                     }
        ax.text(0.0, 1.025, letter, **text_dict, transform=ax.transAxes)

    def print_figure_letter_inside(self, letter, fontsize=14):
        ax = plt.gca()

        text_dict = {"fontsize": fontsize,
                     "verticalalignment": 'bottom',
                     "horizontalalignment": 'left',
                     "bbox": {'facecolor': 'white',
                              'alpha': 0.0,
                              'edgecolor': 'k'}
                     }
        ax.text(0.0125, 0.90, letter, **text_dict, transform=ax.transAxes)

    def plot_topo_dM0(self):

        # Reading Etopo
        topo = lpy.read_etopo()

        # finding topography values
        cmt_topo = []
        for (_lon, _lat) in zip(self.ncmt[:, 9], self.ncmt[:, 8]):
            lonpos = np.argmin(np.abs(topo.longitude.values - _lon))
            latpos = np.argmin(np.abs(topo.latitude.values - _lat))
            cmt_topo.append(topo.bedrock[latpos, lonpos].values)
        # return
        cmt_topo = np.array(cmt_topo)

        # start with a rectangular Figure
        plt.figure(figsize=(4, 4))

        ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
            self.dcmt[:, 0] * 100, cmt_topo/1000, self.nbins,
            z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
            zmin=None, zmax=None, norm=self.depth_norm)
        ax_scatter.set_xlabel(r"Scalar Moment Change [%]")
        ax_scatter.set_ylabel(r"Topography [km]")
        ax_scatter.plot([np.min(np.min(self.dcmt[:, 0] * 100)),
                         np.max(self.dcmt[:, 0] * 100)],
                        [0, 0], 'k', zorder=-1, lw=0.75)
        ax_scatter.plot([0, 0], [np.min(cmt_topo/1000),
                                 np.max(cmt_topo/1000)],
                        'k', zorder=0.1, lw=0.75)

        # Finally plot shot
        filename = "topo_dM0.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_topo_dz(self):

        # Reading Etopo
        topo = lpy.read_etopo()

        # finding topography values
        cmt_topo = []
        for (_lon, _lat) in zip(self.ncmt[:, 9], self.ncmt[:, 8]):
            lonpos = np.argmin(np.abs(topo.longitude.values - _lon))
            latpos = np.argmin(np.abs(topo.latitude.values - _lat))
            cmt_topo.append(topo.bedrock[latpos, lonpos].values)
        # return
        cmt_topo = np.array(cmt_topo)

        # start with a rectangular Figure
        plt.figure(figsize=(4, 4))

        ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
            self.dcmt[:, 7], cmt_topo/1000, self.nbins,
            z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
            zmin=None, zmax=None, norm=self.depth_norm)
        ax_scatter.set_xlabel(r"Change in Depth [km]")
        ax_scatter.set_ylabel(r"Topography [km]")
        ax_scatter.plot([np.min(self.dcmt[:, 7]),
                         np.max(self.dcmt[:, 7])],
                        [0, 0], 'k', zorder=-1, lw=0.75)
        ax_scatter.plot([0, 0], [np.min(cmt_topo/1000),
                                 np.max(cmt_topo/1000)],
                        'k', zorder=0.1, lw=0.75)

        # Finally plot shot
        filename = "topo_dz.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_crust_dM0(self):

        # Reading Etopo
        crust = lpy.read_litho()
        bottom = 'lower_crust_bottom_depth'

        # finding Moho Topography values
        cmt_crust = []
        for (_lon, _lat) in zip(self.ncmt[:, 9], self.ncmt[:, 8]):
            lonpos = np.argmin(np.abs(crust.longitude.values - _lon))
            latpos = np.argmin(np.abs(crust.latitude.values - _lat))
            cmt_crust.append(getattr(crust, bottom)[latpos, lonpos].values)

        # return
        cmt_crust = np.array(cmt_crust)

        # start with a rectangular Figure
        plt.figure(figsize=(4, 4))

        ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
            self.dcmt[:, 0] * 100, cmt_crust, self.nbins,
            z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
            zmin=None, zmax=None, norm=self.depth_norm)
        ax_scatter.set_xlabel(r"Scalar Moment Change [%]")
        ax_scatter.set_ylabel(r"Moho depth [km]")
        ax_scatter.plot([np.min(np.min(self.dcmt[:, 0] * 100)),
                         np.max(self.dcmt[:, 0] * 100)],
                        [0, 0], 'k', zorder=-1, lw=0.75)
        ax_scatter.plot([0, 0], [np.min(cmt_crust),
                                 np.max(cmt_crust)],
                        'k', zorder=0.1, lw=0.75)
        ax_scatter.invert_yaxis()
        ax_histy.set_ylim(ax_scatter.get_ylim())

        # Finally plot shot
        filename = "crust_dM0.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_crust_dz(self):

        # Reading Etopo
        crust = lpy.read_litho()
        bottom = 'lower_crust_bottom_depth'

        # finding Moho Topography values
        cmt_crust = []
        for (_lon, _lat) in zip(self.ncmt[:, 9], self.ncmt[:, 8]):
            lonpos = np.argmin(np.abs(crust.longitude.values - _lon))
            latpos = np.argmin(np.abs(crust.latitude.values - _lat))
            cmt_crust.append(getattr(crust, bottom)[latpos, lonpos].values)

        # return
        cmt_crust = np.array(cmt_crust)

        # start with a rectangular Figure
        plt.figure(figsize=(4, 4))

        ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
            self.dcmt[:, 7], cmt_crust, self.nbins,
            z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
            zmin=None, zmax=None, norm=self.depth_norm)
        ax_scatter.set_xlabel(r"Change in Depth [km]")
        ax_scatter.set_ylabel(r"Moho Depth [km]")
        ax_scatter.plot([np.min(self.dcmt[:, 7]),
                         np.max(self.dcmt[:, 7])],
                        [0, 0], 'k', zorder=-1, lw=0.75)
        ax_scatter.plot([0, 0], [np.min(cmt_crust),
                                 np.max(cmt_crust)],
                        'k', zorder=0.1, lw=0.75)
        ax_scatter.invert_yaxis()
        ax_histy.set_ylim(ax_scatter.get_ylim())

        # Finally plot shot
        filename = "crust_dz.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_thick_dM0(self):

        # Reading Etopo
        topo = lpy.read_etopo()

        # Reading Crust
        crust = lpy.read_litho()
        bottom = 'lower_crust_bottom_depth'

        # finding Moho Topography values
        cmt_crust = []
        cmt_topo = []
        for (_lon, _lat) in zip(self.ncmt[:, 9], self.ncmt[:, 8]):
            clonpos = np.argmin(np.abs(crust.longitude.values - _lon))
            clatpos = np.argmin(np.abs(crust.latitude.values - _lat))
            cmt_crust.append(getattr(crust, bottom)[clatpos, clonpos].values)

            tlonpos = np.argmin(np.abs(topo.longitude.values - _lon))
            tlatpos = np.argmin(np.abs(topo.latitude.values - _lat))
            cmt_topo.append(topo.bedrock[tlatpos, tlonpos].values)

        # return
        cmt_crust = np.array(cmt_crust)
        cmt_topo = np.array(cmt_topo)
        cmt_thickness = cmt_topo/1000 + cmt_crust

        # start with a rectangular Figure
        plt.figure(figsize=(4, 4))

        ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
            self.dcmt[:, 0] * 100, cmt_thickness, self.nbins,
            z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
            zmin=None, zmax=None, norm=self.depth_norm)
        ax_scatter.set_xlabel(r"Scalar Moment Change [%]")
        ax_scatter.set_ylabel(r"Crustal Thickness [km]")
        ax_scatter.plot([np.min(np.min(self.dcmt[:, 0] * 100)),
                         np.max(self.dcmt[:, 0] * 100)],
                        [0, 0], 'k', zorder=-1, lw=0.75)
        ax_scatter.plot([0, 0], [np.min(cmt_thickness),
                                 np.max(cmt_thickness)],
                        'k', zorder=0.1, lw=0.75)

        # Finally plot shot
        filename = "thickness_dM0.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_thick_dz(self):

        # Reading Etopo
        topo = lpy.read_etopo()

        # Reading Crust
        crust = lpy.read_litho()
        bottom = 'lower_crust_bottom_depth'

        # finding Moho Topography values
        cmt_crust = []
        cmt_topo = []
        for (_lon, _lat) in zip(self.ncmt[:, 9], self.ncmt[:, 8]):
            clonpos = np.argmin(np.abs(crust.longitude.values - _lon))
            clatpos = np.argmin(np.abs(crust.latitude.values - _lat))
            cmt_crust.append(getattr(crust, bottom)[clatpos, clonpos].values)

            tlonpos = np.argmin(np.abs(topo.longitude.values - _lon))
            tlatpos = np.argmin(np.abs(topo.latitude.values - _lat))
            cmt_topo.append(topo.bedrock[tlatpos, tlonpos].values)

        # return
        cmt_crust = np.array(cmt_crust)
        cmt_topo = np.array(cmt_topo)
        cmt_thickness = cmt_topo/1000 + cmt_crust

        # start with a rectangular Figure
        plt.figure(figsize=(4, 4))

        ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
            self.dcmt[:, 7], cmt_thickness, self.nbins,
            z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
            zmin=None, zmax=None, norm=self.depth_norm)
        ax_scatter.set_xlabel(r"Change in Depth [km]")
        ax_scatter.set_ylabel(r"Crustal Thickness [km]")
        ax_scatter.plot([np.min(self.dcmt[:, 7]),
                         np.max(self.dcmt[:, 7])],
                        [0, 0], 'k', zorder=-1, lw=0.75)
        ax_scatter.plot([0, 0], [np.min(cmt_thickness),
                                 np.max(cmt_thickness)],
                        'k', zorder=0.1, lw=0.75)

        # Finally plot shot
        filename = "thickness_dz.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_map(self):

        ax = plt.gca()
        ax.set_global()
        ax.frameon = True
        ax.outline_patch.set_linewidth(1)

        # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
        # function around 180deg
        gl = ax.gridlines(crs=PlateCarree(central_longitude=180.0),
                          draw_labels=False,
                          linewidth=1, color='lightgray', alpha=0.5,
                          linestyle='-', zorder=-1.5)
        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = True

        # Add Coastline
        ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
                       linewidth=0.5, facecolor=(0.9, 0.9, 0.9))

    def plot_cmts(self):

        ax = plt.gca()
        for idx, (lon, lat, m) in enumerate(zip(self.ncmt[:, 9], self.ncmt[:, 8],
                                                self.ncmt[:, 1:7])):
            try:
                # Longitude fix because cartopy is being shitty
                if self.cl == 180.0:
                    if lon <= 0:
                        lon = lon + 180.0
                    else:
                        lon = lon - 180.0
                b = beach(m, linewidth=0.25,
                          facecolor=self.depth_cmap(self.depth_norm(
                              self.ocmt[idx, 7]
                          )),
                          bgcolor='w',
                          edgecolor='k', alpha=1,
                          xy=(lon, lat), width=10,
                          size=10, nofill=False, zorder=-1)

                ax.add_collection(b)
            except Exception as e:
                for line in e.__str__().splitlines():
                    logger.error(line)

    def plot_stations(self):
        """Plots stations into a map
        """

        slat = [station[0] for station in self.stations]
        # Weird fix because cartopy is weird
        if self.cl == 180.0:
            slon = [station[1] + self.cl if station[1] <= 0
                    else station[1] - self.cl
                    for station in self.stations]
        else:
            slon = [station[1] for station in self.stations]

        ax = plt.gca()
        ax.scatter(slon, slat, s=20, marker='v', c=((0.7, 0.2, 0.2),),
                   edgecolors='k', linewidths=0.25, zorder=-1)

    def plot_histogram(self, ddata, n_bins, facecolor=(0.7, 0.2, 0.2),
                       alpha=1, chi=False, wmin=None, statsleft: bool = False,
                       label: str or None = None, stats: bool = True,
                       CI: bool = True):
        """Plots histogram of input data."""

        if wmin is not None:
            logger.info(f"Datamin: {np.min(ddata)}")
            ddata = ddata[np.where(ddata >= wmin)]
            logger.info(f"Datamin: {np.min(ddata)}")
        # the histogram of the data
        ax = plt.gca()
        n, bins, _ = ax.hist(ddata, n_bins, facecolor=facecolor,
                             edgecolor=facecolor, alpha=alpha, label=label)
        _, _, _ = ax.hist(ddata, n_bins, color='k', histtype='step')
        text_dict = {
            "fontsize": 'x-small',
            "verticalalignment": 'top',
            "horizontalalignment": 'right',
            "transform": ax.transAxes,
            "zorder": 100,
            'family': 'monospace'
        }

        if stats:
            if statsleft:
                text_dict["horizontalalignment"] = 'left'
                posx = 0.03
            else:
                posx = 0.97

            ax.text(posx, 0.97,
                    f"$\mu$ = {np.mean(ddata):5.2f}\n"
                    f"$\sigma$ = {np.std(ddata):5.2f}",
                    **text_dict)
        if CI:
            ci_norm = {
                "80": 1.282,
                "85": 1.440,
                "90": 1.645,
                "95": 1.960,
                "99": 2.576,
                "99.5": 2.807,
                "99.9": 3.291
            }
            if chi:
                Zval = ci_norm["90"]
            else:
                Zval = ci_norm["95"]

            mean = np.mean(ddata)
            pmfact = Zval * np.std(ddata)
            CI = [mean - pmfact, mean + pmfact]
            # if we are only concerned about the lowest values the more the better:
            if wmin is not None:
                CI[1] = np.max(ddata)
                if CI[0] < wmin:
                    CI[0] = wmin
            minbox = [np.min(bins), 0]
            minwidth = (CI[0]) - minbox[0]
            maxbox = [CI[1], 0]
            maxwidth = np.max(bins) - maxbox[0]
            height = np.max(n)*1.05

            boxdict = {
                "facecolor": 'w',
                "edgecolor": None,
                "alpha": 0.6,
            }
            minR = Rectangle(minbox, minwidth, height, **boxdict)
            maxR = Rectangle(maxbox, maxwidth, height, **boxdict)
            ax.add_patch(minR)
            ax.add_patch(maxR)

            return CI
        else:
            return None

    def plot_magnitude_comp(self):

        # Mi chose this yellow
        yellow = np.array((0.93,  0.75,  0.20))
        blue = np.array((0.25, 0.25, 0.9))

        # This converts dt.datetime.fromtimestamp function
        # into numpy function making a for loop unnecessary. It is not faster
        # however
        dateconv = np.vectorize(dt.datetime.fromtimestamp)

        # Get CMT Times
        sortpos = np.argsort(deepcopy(self.ncmt[:, 10]))
        M0 = self.ncmt[sortpos, 0]

        Mw = 2/3 * np.log10(M0) - 10.7
        cmt_times = dateconv(self.ncmt[sortpos, 10])

        # Get cumulative values Note that the cmt solution is in dyne*cm
        cum_M0_dyne_cm = np.cumsum(M0, dtype=np.float)

        # Make figure
        plt.figure(figsize=(11.5, 2.8))
        plt.subplots_adjust(bottom=0.16, top=0.9, left=0.06, right=0.94,
                            wspace=0.4)

        plt.subplot(131)
        bin_edges = np.arange(4.0, 9.7, 0.1)

        # Plot my data
        self.plot_histogram(
            Mw, bin_edges, facecolor='lightgray', label='GCMT3D',
            stats=False, CI=False)

        # Load period before 2004
        x0, y0 = lpy.load_1976_2004_mag()
        plt.plot(x0, y0, 'k', label='1976-2004 GCMT')
        plt.yscale('log')

        # Load period after 2004
        x1, y1 = lpy.load_2004_2010_mag()
        plt.plot(x1, y1, 'r', label='2004-2010 GCMT')

        plt.legend(loc='upper right', frameon=False, fancybox=False,
                   numpoints=1, scatterpoints=1, fontsize='x-small',
                   borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
                   labelspacing=0.2, handlelength=1.0,
                   bbox_to_anchor=(1.0, 1.0))

        #  Plot params
        plt.ylim((0.7, 5000.0))
        plt.xlabel("Moment Magnitude $M_W$")
        plt.ylabel("N (per $0.1$ mag. bin)")
        self.print_figure_letter('a')

        # Cumulative magnitude in the catalog
        ax2 = plt.subplot(132)

        x, y = lpy.load_cum_mag()

        # Create conversion function!
        convy2dt = np.vectorize(lpy.year2date)

        # Plot GCMT Data
        plt.fill_between(
            convy2dt(x), y, facecolor=(*yellow, 1.0), edgecolor='k',
            linewidth=0.75, label="GCMT")

        # Plot GCMT3D Data
        plt.fill_between(
            cmt_times, cum_M0_dyne_cm * (1e-30), facecolor=(*blue, 1.0),
            edgecolor='k', linewidth=0.75, label="GCMT3D")

        plt.legend(loc='upper left', frameon=False, fancybox=False,
                   numpoints=1, scatterpoints=1, fontsize='x-small',
                   borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
                   labelspacing=0.2, handlelength=1.0,
                   bbox_to_anchor=(0.0, 1.0))

        locator = matplotlib.dates.YearLocator(10)
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

        # Plot labels
        self.print_figure_letter('b')
        plt.ylim((0.0, 3.1))
        plt.xlim((dt.datetime(year=1976, month=1, day=1), dt.datetime.now()))
        plt.xlabel('Year')
        plt.ylabel('$10^{30}$ dyne-cm')

        # Plot Yearly distribution of earthquakes
        ax3 = plt.subplot(133)

        # Load the gcmt data
        x2, y2 = lpy.load_num_events()

        # # Set bin dates
        # def add_years(date: dt.datetime, years: int) -> dt.datetime:
        #     try:
        #         return date.replace(year=date.year + years)
        #     except ValueError:
        #         return date + (dt.datetime(date.year + years, 1, 1)
        #                        - dt.datetime(date.year, 1, 1))

        # Create bin edges
        bine = [lpy.add_years(dt.datetime(year=1976, month=1, day=1), i)
                for i in range(dt.datetime.now().year-1976+1)]

        # Plot histogram GCMT
        plt.hist(convy2dt(np.round(x2) + 0.5), bins=bine, edgecolor='k',
                 facecolor=(*yellow, 1.0), linewidth=0.75,
                 label='GCMT', weights=y2, histtype='stepfilled')

        # Plot histogram GCMT3D
        plt.hist(cmt_times, bins=bine, edgecolor='k',
                 facecolor=(*blue, 1.0), linewidth=0.75,
                 label='GCMT3D', histtype='stepfilled')

        plt.legend(loc='upper left', frameon=False, fancybox=False,
                   numpoints=1, scatterpoints=1, fontsize='x-small',
                   borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
                   labelspacing=0.2, handlelength=1.0,
                   bbox_to_anchor=(0.0, 1.0))
        ax3.xaxis.set_major_locator(locator)
        ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

        # labels
        self.print_figure_letter('c')
        plt.xlabel('Year')
        plt.ylabel('Number of Events')
        plt.xlim((dt.datetime(year=1976, month=1, day=1), dt.datetime.now()))

        # Saving
        filename = f"mag_freq_comp.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_spatial_change(self):

        # Interpolate the map
        # Creating a kdtree, and use it to interp
        # SNN = lpy.SphericalNN(lat, lon)
        # rad = SNN.interp(rad, llat, llon, no_weighting=True)
        # val = SNN.interp(val, llat, llon, no_weighting=True)
        # per = SNN.interp(per, llat, llon, no_weighting=True)
        # dif = SNN.interp(dif, llat, llon, no_weighting=True)
        # dis = SNN.interp(dis, llat, llon, no_weighting=True)

        for _i in range(10):

            fig = plt.figure(figsize=(8.0, 3.0))
            ax = plt.subplot(111, projection=PlateCarree())

            # Map
            self.plot_map()

            # Cmap
            norm = matplotlib.colors.TwoSlopeNorm(vmin=self.bounds[_i][0],
                                                  vmax=self.bounds[_i][1],
                                                  vcenter=0.0)
            # # Data
            # self.plot_2d_histogram(ax, self.ncmt[:, 9], self.ncmt[:, 8],
            #                        data=self.factor[_i] * self.dcmt[:, _i],
            #                        cmap='coolwarm', norm=norm,
            #                        label=self.units[_i])

            # Letters
            self.print_figure_letter_inside(self.dlabels[_i])
            self.print_figure_letter(self.abc[_i] + ")")

            # Saving
            filename = f"space_{self.tags[_i]}.pdf"
            if self.prefix is not None:
                filename = self.prefix + "_" + filename
            plt.savefig(os.path.join(self.outdir, filename))
            plt.close()

        # Figure and axes
        fig = plt.figure(figsize=(8.0, 3.0))
        ax = plt.subplot(111, projection=PlateCarree())

        # Map
        self.plot_map()

        # Cmap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=360)

        # Data
        data = (np.pi/2 - np.arctan2(self.dcmt[:, 8],
                                     self.dcmt[:, 9]))/np.pi*180 + 180
        self.plot_2d_histogram(ax, self.ncmt[:, 9], self.ncmt[:, 8],
                               data=data, cmap='twilight_shifted', norm=norm,
                               label="deg")

        # Letters
        self.print_figure_letter_inside("Angle")
        self.print_figure_letter(self.abc[10] + ")")

        # Saving
        filename = "space_angle.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

        # Figure and axes
        fig = plt.figure(figsize=(8.0, 3.0))
        ax = plt.subplot(111, projection=PlateCarree())

        # Map
        self.plot_map()

        # Cmap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)

        # Data
        self.plot_2d_histogram(ax, self.ncmt[:, 9], self.ncmt[:, 8],
                               data=None, norm=norm, label="#")

        # Letters
        self.print_figure_letter_inside("Counts")
        self.print_figure_letter(self.abc[11] + ")")

        # Saving
        filename = "space_counts.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_dM_dz(self):
        """Creates Figure with histograms one the side for two
        change in depth and change in scalar moment."""

        # start with a rectangular Figure
        fig = plt.figure(figsize=(4, 4))

        ax_scatter, ax_histx, ax_histy, cax = self.plot_scatter_hist(
            self.dcmt[:, 7], self.dcmt[:, 0] * 100, self.nbins,
            z=self.ocmt[:, 7], cmap=self.depth_cmap, histc='grey',
            zmin=None, zmax=None, norm=self.depth_norm)
        ax_scatter.set_xlabel(r"Depth change [km]")
        ax_scatter.set_ylabel(r"Scalar Moment Change [%]")
        ax_scatter.plot([np.min(self.dcmt[:, 7]),
                         np.max(self.dcmt[:, 7])],
                        [0, 0], 'k', zorder=-1, lw=0.75)
        ax_scatter.plot([0, 0], [np.min(self.dcmt[:, 0] * 100),
                                 np.max(self.dcmt[:, 0] * 100)],
                        'k', zorder=0.1, lw=0.75)

        # Finally plot shot
        filename = "dM_dz_nz.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    @staticmethod
    def plot_2d_histogram(ax, lon, lat, data=None, cmap=None, norm=None, alpha=None,
                          label=None):

        ax = plt.gca()

        # Create binning
        res = 4
        binlon = np.linspace(-180, 180, int(1/res*181))
        binlat = np.linspace(-90, 90, int(1/res*91))
        counts, xx, yy = np.histogram2d(lon, lat, bins=(binlon, binlat))
        if data is not None:
            zz, _, _ = np.histogram2d(lon, lat, bins=(binlon, binlat),
                                      weights=data)
            # Workaround for zero count values tto not get an error.
            # Where counts == 0, zi = 0, else zi = zz/counts
            zi = np.zeros_like(zz)
            zi[counts.astype(bool)] = zz[counts.astype(bool)] / \
                counts[counts.astype(bool)]
            zi = np.ma.masked_equal(zi, 0)
            pl = ax.pcolormesh(xx, yy, zi.T, linewidth=0.0, cmap=cmap,
                               norm=norm, alpha=None)
        else:
            zi = counts
            zi = np.ma.masked_equal(zi, 0)
            pl = ax.pcolormesh(xx, yy, zi.T, linewidth=0.0, cmap='magma',
                               norm=norm, alpha=None)

        cbar = plt.colorbar(pl, pad=0.01, aspect=25)
        cbar.set_label(label, rotation=0)

        return pl

    @staticmethod
    def plot_scatter_hist(x, y, nbins, z=None, cmap=None,
                          histc=((0.35, 0.35, 0.35),),
                          zmin=None, zmax=None, norm=None, r=True,
                          xlog=False, ylog=False, ellipses=True):
        """

        :param x: Data type x-axis
        :param y: Data type y-axis
        :params nbins: number of bins
        :param z: Data type to color xy datapoints. Default None. Datapoints in
                  color of histogram.
        :param cmap: name
        :param zmin: Colorbar min
        :param zmin: Colorbar max

        :return:
        """

        # definitions for the axes
        left, width = 0.125, 0.65
        bottom, height = 0.125, 0.65
        spacing = 0.000

        # Create dimensions
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.15]
        rect_histy = [left + width + spacing, bottom, 0.15, height]

        # Create Axes
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labelbottom=False)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)

        # scatterplot with color
        if cmap is not None and z is not None:

            cax = ax_scatter.inset_axes([0.05, 0.96, 0.25, 0.03],
                                        zorder=10000)

            zpos = np.argsort(z)
            # the scatter plot:
            if zmin is not None:
                vminz = zmin
            else:
                vminz = np.min(z)

            if zmax is not None:
                vmaxz = zmax
            else:
                vmaxz = np.max(z)

            if norm is not None and cmap is not None:
                ax_scatter.scatter(x[zpos], y[zpos], c=cmap(norm(z[zpos])),
                                   s=20, marker='o', edgecolor='none',
                                   linewidths=0.5, alpha=0.25)

                # Colorbar
                cbar_dict = {"orientation": "horizontal"}
                plt.colorbar(matplotlib.cm.ScalarMappable(
                    cmap=cmap, norm=norm),
                    cax=cax, **cbar_dict)
                cax.tick_params(left=False, right=False, bottom=True, top=True,
                                labelleft=False, labelright=False,
                                labeltop=False,
                                labelbottom=True, which='both',
                                labelsize=6)

            else:
                ax_scatter.scatter(
                    x[zpos], y[zpos],
                    c=get_color(z[zpos], vmin=vminz, vmax=vmaxz,
                                cmap=cmap, norm=norm),
                    s=20, marker='o', edgecolor='none', linewidths=0.5,
                    alpha=0.25)

                # Colorbar
                cbar_dict = {"orientation": "horizontal"}
                create_colorbar(vminz, vmaxz, cmap=cmap, cax=cax, **cbar_dict)
                cax.tick_params(left=False, right=False, bottom=True, top=True,
                                labelleft=False, labelright=False,
                                labeltop=False, labelbottom=True, which='both',
                                labelsize=6)

        # scatterplot without color
        else:
            # the scatter plot:
            ax_scatter.scatter(x, y, c=histc, s=15, marker='o', alpha=0.25,
                               edgecolor='none')
            cax = None

        if r:
            if xlog:
                xfix = np.log10(x)
            else:
                xfix = x
            if ylog:
                yfix = np.log10(y)
            else:
                yfix = y
            # Write out correlation coefficient in the top right
            corr_coeff = np.corrcoef(xfix, yfix)
            text_dict = {"fontsize": 6, "verticalalignment": 'top',
                         "zorder": 100}
            ax_scatter.text(0.97, 0.97,
                            "R = %5.2f\n"
                            "$\mu_x$ = %5.2f\n"
                            "$\mu_y$ = %5.2f" % (
                                corr_coeff[0, 1], np.mean(x), np.mean(y)),
                            horizontalalignment='right', **text_dict,
                            transform=ax_scatter.transAxes)

        if ellipses:
            ls = ["-", "--", ":"]
            for _i in np.arange(1, 4):
                el = confidence_ellipse(
                    x, y, ax_scatter, n_std=_i, label=r'$%d\sigma$' % _i,
                    edgecolor='k', linestyle=ls[_i-1])
                ax_scatter.add_patch(el)
            ax_scatter.legend(loc='lower left', ncol=3)

        # now determine nice limits by hand:
        ax_scatter.set_xlim((np.min(x), np.max(x)))
        ax_scatter.set_ylim((np.min(y), np.max(y)))

        ax_scatter.xaxis.label.set_size('x-small')
        ax_scatter.yaxis.label.set_size('x-small')

        # Histogram settings
        binsx = np.linspace(np.min(x), np.max(x), nbins + 1)
        binsy = np.linspace(np.min(y), np.max(y), nbins + 1)
        ax_histx.hist(x, bins=binsx,
                      color=histc, ec=None)
        ax_histx.set_xlim(ax_scatter.get_xlim())
        ax_histy.hist(y, bins=binsy, orientation='horizontal',
                      color=histc, ec=None)
        ax_histy.set_ylim(ax_scatter.get_ylim())

        # Remove boundaries
        remove_all(ax=ax_histx, bottom=True)
        remove_all(ax=ax_histy, left=True)

        if cax is not None:
            return ax_scatter, ax_histx, ax_histy, cax
        else:
            return ax_scatter, ax_histx, ax_histy

    def selection_histograms(self):

        # Create figure handle
        fig = plt.figure(figsize=(9, 6))

        # Create subplot layout
        GS = GridSpec(2, 3)

        # Create axis for map
        ax = fig.add_subplot(GS[0, 0])
        ci_angle = self.plot_histogram(self.angles/np.pi*180, self.nbins)
        remove_topright()
        plt.xlabel("Angular Change [$^\circ$]")
        plt.ylabel("$N$", rotation=0, horizontalalignment='right')
        self.print_figure_letter("a")

        if self.measurements is not None:
            fig.add_subplot(GS[0, 1])
            total_measurements = get_total_measurements(self.measurements)
            ci_measurements = self.plot_histogram(
                total_measurements, self.nbins, wmin=200)
            logger.info(
                f"Measurement CI: {ci_measurements[0]}, {ci_measurements[1]}")
            remove_topright()
            plt.xlabel("# of windows")
            plt.ylabel("$N$", rotation=0, horizontalalignment='right')
            self.print_figure_letter("b")
            measurement_select = np.where(
                ((ci_measurements[0] < total_measurements)
                 & (total_measurements < ci_measurements[1])))[0]

            # measurement_select = np.where(200 < total_measurements)[0]

        fig.add_subplot(GS[1, 0])
        ci_depth = self.plot_histogram(self.dcmt[:, 7], self.nbins)
        remove_topright()
        plt.xlabel("Depth Change [km]")
        plt.ylabel("$N$", rotation=0, horizontalalignment='right')
        self.print_figure_letter("d")

        fig.add_subplot(GS[1, 1])
        ci_t0 = self.plot_histogram(self.dcmt[:, -1], self.nbins)
        remove_topright()
        plt.xlabel("Centroid Time Change [sec]")
        plt.ylabel("$N$", rotation=0, horizontalalignment='right')
        self.print_figure_letter("e")

        fig.add_subplot(GS[1, 2])
        ci_m0 = self.plot_histogram(self.dcmt[:, 0]*100, self.nbins)
        remove_topright()
        plt.xlabel("Scalar Moment Change [%]")
        plt.ylabel("$N$", rotation=0, horizontalalignment='right')
        self.print_figure_letter("f")

        # SELECTION
        m0_select = np.where(
            ((ci_m0[0] < self.dcmt[:, 0]*100)
             & (self.dcmt[:, 0]*100 < ci_m0[1])))[0]
        t0_select = np.where(
            ((ci_t0[0] < self.dcmt[:, -1])
             & (self.dcmt[:, -1] < ci_t0[1])))[0]
        z_select = np.where(
            ((ci_depth[0] < self.dcmt[:, 7])
             & (self.dcmt[:, 7] < ci_depth[1])))[0]
        angle_select = np.where(
            ((ci_angle[0] < self.angles/np.pi*180)
             & (self.angles/np.pi*180 < ci_angle[1])))[0]

        # Get intersection
        m0_select = set([int(x) for x in m0_select])
        t0_select = set([int(x) for x in t0_select])
        z_select = set([int(x) for x in z_select])
        angle_select = set([int(x) for x in angle_select])
        if self.measurements is not None:
            measurement_select = set([int(x) for x in measurement_select])

            selection = m0_select.intersection(t0_select, z_select, angle_select,
                                               measurement_select)
        else:
            selection = m0_select.intersection(
                t0_select, z_select, angle_select)
            measurement_select = ""

        selection = list(selection)
        selection.sort()
        print("{0:-^72}".format(" Header "))
        print("Number of selected in each category")
        print(f"M0_____________: {len(m0_select):=5}")
        print(f"t0_____________: {len(t0_select):=5}")
        print(f"z______________: {len(z_select):=5}")
        print(f"angle__________: {len(angle_select):=5}")
        print(f"# of meas______: {len(measurement_select):=5}")
        print(" ")
        print(f"Intersection___: {len(selection):=5}")
        print(72 * "-")
        print(" ")
        print(f"\nM0 within confidence interval [{len(m0_select)}]:\n")
        for m0_ind in m0_select:
            print(os.path.abspath(self.ncmtfiles[m0_ind]))
        print(f"\nt0 within confidence interval [{len(t0_select)}]:\n")
        for t0_ind in t0_select:
            print(os.path.abspath(self.ncmtfiles[t0_ind]))
        print(f"\nDepth within confidence interval [{len(z_select)}]:\n")
        for z_ind in z_select:
            print(os.path.abspath(self.ncmtfiles[z_ind]))
        print(f"\nAngle within confidence interval [{len(angle_select)}]:\n")
        for angle_ind in angle_select:
            print(os.path.abspath(self.ncmtfiles[angle_ind]))
        print(f"\nAbove number of measurements [{len(measurement_select)}]:\n")
        for measure_ind in measurement_select:
            print(os.path.abspath(self.ncmtfiles[measure_ind]))

        # Intersection
        print(f"\nIntersection [{len(selection)}/{len(self.ncmtfiles)}]:\n")
        for sel in selection:
            print(os.path.abspath(self.ncmtfiles[sel]))

        # Finally plot shot
        plt.tight_layout(pad=2, w_pad=2.5, h_pad=2.25)

        filename = "selection_histograms.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()


if __name__ == "__main__":
    pass
