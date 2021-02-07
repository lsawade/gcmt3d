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
import datetime as dt
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

    if (type(rgb) is tuple) or ((type(rgb) is np.ndarray) and (rgb.size == 3)):
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

        lpy.plot_map(labelstopright=False, labelsbottomleft=False)

    def plot_cmts(self):

        ax = plt.gca()
        for idx, (lon, lat, m) in enumerate(zip(self.ncmt[:, 9],
                                                self.ncmt[:, 8],
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
                    f"$\\mu$ = {np.mean(ddata):5.2f}\n"
                    f"$\\sigma$ = {np.std(ddata):5.2f}",
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
            # if we are only concerned about the lowest values the more
            # the better:
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
        filename = "mag_freq_comp.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

    def plot_spatial_change(self):

        # Interpolate the map
        # Creating a kdtree, and use it to interp
        SNN = lpy.SphericalNN(self.ncmt[:, 8], self.ncmt[:, 9])

        # Create desired grid for plotting
        res = 1
        llon, llat = np.meshgrid(np.arange(-180.0, 180.0 + res, res),
                                 np.arange(-90.0, 90.0 + res, res))

        plt.figure(figsize=(11.0, 5.5))

        # ########## Mw ######################
        ax = plt.subplot(221, projection=PlateCarree())
        plt.subplots_adjust(bottom=0.05, top=0.95, left=0.01, right=0.95,
                            wspace=0.1)

        # Map
        self.plot_map()

        # Cmap
        norm = matplotlib.colors.Normalize(vmin=5.75, vmax=6.75)
        val = SNN.interp(lpy.m0_2_mw(self.ncmt[:, 0]), llat, llon,
                         maximum_distance=3.0, no_weighting=False,
                         k=200)

        pmesh = plt.pcolormesh(
            llon, llat, lpy.smooth_nan_image(val, sigma=1.0, truncate=1.5),
            transform=cartopy.crs.PlateCarree(),
            cmap='magma_r', norm=norm, rasterized=True)

        cbar = lpy.nice_colorbar(pmesh, fraction=0.05, pad=0.02)
        cbar.set_label('$M_W$')

        # Letters
        self.print_figure_letter_inside('$M_W$')
        self.print_figure_letter(self.abc[0] + ")")

        # ########## dM0/M0 ######################
        ax = plt.subplot(222, projection=PlateCarree())

        # Map
        self.plot_map()

        # Cmap
        norm = matplotlib.colors.TwoSlopeNorm(vmin=self.bounds[0][0],
                                              vmax=self.bounds[0][1],
                                              vcenter=0.0)
        val = SNN.interp(self.factor[0] * self.dcmt[:, 0], llat, llon,
                         maximum_distance=3.0, no_weighting=False,
                         k=200)

        pmesh = plt.pcolormesh(
            llon, llat, lpy.smooth_nan_image(val, sigma=1.0, truncate=1.5),
            transform=cartopy.crs.PlateCarree(),
            cmap='coolwarm', norm=norm, rasterized=True)

        cbar = lpy.nice_colorbar(pmesh, fraction=0.05, pad=0.02)
        cbar.set_label(self.units[0])

        # Letters
        self.print_figure_letter_inside(self.dlabels[0])
        self.print_figure_letter(self.abc[1] + ")")

        # ########## Mw ######################
        ax = plt.subplot(223, projection=PlateCarree())

        # Map
        self.plot_map()

        # Cmap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=500)
        val = SNN.interp(self.ncmt[:, 7], llat, llon,
                         maximum_distance=3.0, no_weighting=False,
                         k=200)

        pmesh = plt.pcolormesh(
            llon, llat, lpy.smooth_nan_image(val, sigma=1.0, truncate=1.5),
            transform=cartopy.crs.PlateCarree(),
            cmap='magma_r', norm=norm, rasterized=True)

        cbar = lpy.nice_colorbar(pmesh, fraction=0.05, pad=0.02)
        cbar.set_label(self.units[7])
        cbar.ax.invert_yaxis()

        # Letters
        self.print_figure_letter_inside(self.labels[7])
        self.print_figure_letter(self.abc[2] + ")")

        # ########## dM0/M0 ######################
        ax = plt.subplot(224, projection=PlateCarree())

        # Map
        self.plot_map()

        # Cmap
        norm = matplotlib.colors.TwoSlopeNorm(vmin=self.bounds[7][0],
                                              vmax=self.bounds[7][1],
                                              vcenter=0.0)
        val = SNN.interp(self.factor[7] * self.dcmt[:, 7], llat, llon,
                         maximum_distance=3.0, no_weighting=False,
                         k=200)

        pmesh = plt.pcolormesh(
            llon, llat, lpy.smooth_nan_image(val, sigma=1.0, truncate=1.5),
            transform=cartopy.crs.PlateCarree(),
            cmap='coolwarm', norm=norm, rasterized=True)

        cbar = lpy.nice_colorbar(pmesh, fraction=0.05, pad=0.02)
        cbar.set_label(self.units[7])

        # Letters
        self.print_figure_letter_inside(self.dlabels[7])
        self.print_figure_letter(self.abc[3] + ")")

        # Saving
        filename = "space_overview.pdf"
        if self.prefix is not None:
            filename = self.prefix + "_" + filename
        plt.savefig(os.path.join(self.outdir, filename))
        plt.close()

        print("Done M0.")

        return None
        # Figure and axes
        plt.figure(figsize=(8.0, 3.0))
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
        plt.figure(figsize=(8.0, 3.0))
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
        plt.figure(figsize=(4, 4))

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
    def plot_2d_histogram(ax, lon, lat, data=None, cmap=None, norm=None,
                          alpha=None, label=None):

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
                            "$\\mu_x$ = %5.2f\n"
                            "$\\mu_y$ = %5.2f" % (
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
        fig.add_subplot(GS[0, 0])
        ci_angle = self.plot_histogram(self.angles/np.pi*180, self.nbins)
        remove_topright()
        plt.xlabel("Angular Change [$^\\circ$]")
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

            selection = m0_select.intersection(
                t0_select, z_select, angle_select, measurement_select)
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
