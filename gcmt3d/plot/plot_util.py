"""

Plot utilities not to modify plots or base plots.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2020


"""

import scipy.ndimage
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from obspy.imaging.beachball import beach
from obspy.geodetics.base import gps2dist_azimuth
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from ..source import CMTSource

# Colors for the optically less capable.
COLORBLIND = ['#000000', '#ffffff', '#ff1e00', '#ffb49d',
              '#22a8be', '#36fa0e', '#df0095']

def figletter():
    for i in 'abcdefghijklmnopqrstuvwxyz':
        yield i

def remove_topright(ax=None):
    """Removes top and right border and ticks from input axes."""

    # Get current axis if none given.
    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def remove_all(ax=None, top=False, bottom=False, left=False, right=False,
               xticks='none', yticks='none'):
    """Removes all frames and ticks."""
    # Get current axis if none given.
    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)
    ax.spines['right'].set_visible(right)
    ax.spines['top'].set_visible(top)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position(yticks)
    ax.xaxis.set_ticks_position(xticks)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def get_color(x, cmap="seismic", vmin=0, vmax=100, norm=None):
    n = vmax - vmin
    val = (np.clip(x, vmin, vmax) - vmin) / n

    if type(cmap) not in [ListedColormap, LinearSegmentedColormap]:
        cmap = getattr(cm, cmap)

    if norm is not None:
        val = norm(val)

    return cmap(val)


def sns_to_mpl(sns_cmap, linear=False):
    """Converts sns colormap to matplotlib colormap. If the ``as_cmap``
    flag is on for seaborn cmap a linear list is returned. we don't want
    that."""
    if linear:
        return LinearSegmentedColormap(sns_cmap.as_hex())
    else:
        return ListedColormap(sns_cmap.as_hex())


def create_colorbar(vmin, vmax, cmap="seismic", norm=None, cax=None, **kwargs):
    """Creates Colorbar given certain inputs."""

    if type(cmap) not in [ListedColormap, LinearSegmentedColormap]:
        cmap = getattr(cm, cmap)

    if norm is not None:
        values = np.linspace(vmin, vmax, cmap.N)
        norm = colors.BoundaryNorm(values, cmap.N)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    ax = plt.gca()
    if cax is None:
        c = plt.colorbar(sm, ax=ax, **kwargs)
    else:
        c = plt.colorbar(sm, cax=cax, **kwargs)
    return c


# def set_mpl_params_section():
#     params = {
#         'font.weight': 'bold',
#         'axes.labelweight': 'bold',
#         'axes.labelsize': 15,
#         'axes.titlesize': 17,
#         'axes.titleweight': "bold",
#         'axes.outline.linewidth': 2,
#         'xtick.labelsize': 13,
#         'xtick.direction': 'in',
#         'xtick.top': True,  # draw label on the top
#         'xtick.bottom': True,  # draw label on the bottom
#         'xtick.minor.visible': True,
#         'xtick.major.top': True,  # draw x axis top major ticks
#         'xtick.major.bottom': True,  # draw x axis bottom major ticks
#         'xtick.minor.top': True,  # draw x axis top minor ticks
#         'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
#         'ytick.labelsize': 13,
#         'ytick.direction': 'in',
#         'ytick.left': True,  # draw label on the top
#         'ytick.right': True,  # draw label on the bottom
#         'ytick.minor.visible': True,
#         'ytick.major.left': True,  # draw x axis top major ticks
#         'ytick.major.right': True,  # draw x axis bottom major ticks
#         'ytick.minor.left': True,  # draw x axis top minor ticks
#         'ytick.minor.right': True,  # draw x axis bottom minor tick
#     }
#     matplotlib.rcParams.update(params)


# def set_mpl_params_stats():
#     params = {
#         'font.weight': 'bold',
#         'axes.labelweight': 'bold',
#         'axes.labelsize': 7,
#         'axes.linewidth': 1,
#         'xtick.labelsize': 7,
#         'xtick.direction': 'in',
#         'xtick.top': True,  # draw label on the top
#         'xtick.bottom': True,  # draw label on the bottom
#         'xtick.minor.visible': True,
#         'xtick.major.top': True,  # draw x axis top major ticks
#         'xtick.major.bottom': True,  # draw x axis bottom major ticks
#         'xtick.minor.top': True,  # draw x axis top minor ticks
#         'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
#         'ytick.labelsize': 7,
#         'ytick.direction': 'in',
#         'ytick.left': True,  # draw label on the top
#         'ytick.right': True,  # draw label on the bottom
#         'ytick.minor.visible': True,
#         'ytick.major.left': True,  # draw x axis top major ticks
#         'ytick.major.right': True,  # draw x axis bottom major ticks
#         'ytick.minor.left': True,  # draw x axis top minor ticks
#         'ytick.minor.right': True,  # draw x axis bottom minor ticks
#         'legend.fancybox': False,
#         'legend.frameon': False,
#         'legend.loc': 'upper left',
#         'legend.numpoints': 2,
#         'legend.fontsize': 7,
#         'legend.framealpha': 1,
#         'legend.scatterpoints': 3,
#         'legend.edgecolor': 'inherit'
#     }
#     matplotlib.rcParams.update(params)


# def set_mpl_params_summary():
#     params = {
#         'font.weight': 'bold',
#         'axes.labelweight': 'normal',
#         'axes.labelsize': 12,
#         'axes.titlesize': 14,
#         'axes.titleweight': "bold",
#         'axes.linewidth': 0.5,
#         'xtick.labelsize': 6,
#         'xtick.direction': 'in',
#         'xtick.top': True,  # draw label on the top
#         'xtick.bottom': True,  # draw label on the bottom
#         'xtick.minor.visible': False,
#         'xtick.major.top': False,  # draw x axis top major ticks
#         'xtick.major.bottom': True,  # draw x axis bottom major ticks
#         'xtick.minor.top': False,  # draw x axis top minor ticks
#         'xtick.minor.bottom': False,  # draw x axis bottom minor ticks
#         'ytick.labelsize': 6,
#         'ytick.direction': 'in',
#         'ytick.left': True,  # draw label on the top
#         'ytick.right': True,  # draw label on the bottom
#         'ytick.minor.visible': False,
#         'ytick.major.left': True,  # draw x axis top major ticks
#         'ytick.major.right': False,  # draw x axis bottom major ticks
#         'ytick.minor.left': False,  # draw x axis top minor ticks
#         'ytick.minor.right': False,  # draw x axis bottom minor ticks
#     }
#     matplotlib.rcParams.update(params)


def plot_bounds():
    ax = plt.gca()
    lw = 1.0
    ax.plot([0, 1], [0.0, 0.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0, 1], [1.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0, 0], [0.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([1, 1], [0.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)


def plot_bottomline(lw=0.5):
    ax = plt.gca()
    ax.plot([0.1, 0.9], [0.0, 0.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0, 1], [1.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0, 0], [0.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([1, 1], [0.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)


def plot_topline(lw=0.5):
    ax = plt.gca()
    ax.plot([0.1, 0.9], [0.0, 0.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)


def get_new_locations(lat1, lon1, lat2, lon2, padding):
    """ Get new plodding locations

    :param lat1: Old cmt lat
    :param lon1: Old cmt lon
    :param lat2: New cmt lat
    :param lon2: New cmt lon
    :param az: azimuth
    :param padding: padding
    :return: newlat1, newlon1, newlat2, newlon2
    """

    # Get aziumth
    _, az, _ = gps2dist_azimuth(lat1, lon1,
                                lat2, lon2)

    # length of corner placement:
    dx = 1.2 * padding

    if 0 <= az < 90:
        newlat1 = lat1 - dx
        newlon1 = lon1 - dx
        newlat2 = lat1 + dx
        newlon2 = lon1 + dx

    elif 90 <= az < 180:
        newlat1 = lat1 - dx
        newlon1 = lon1 + dx
        newlat2 = lat1 + dx
        newlon2 = lon1 - dx

    elif 180 <= az < 270:
        newlat1 = lat1 + dx
        newlon1 = lon1 + dx
        newlat2 = lat1 - dx
        newlon2 = lon1 - dx

    else:
        newlat1 = lat1 + dx
        newlon1 = lon1 - dx
        newlat2 = lat1 - dx
        newlon2 = lon1 + dx

    return newlat1, newlon1, newlat2, newlon2


def plot_beachballs(oldcmt: CMTSource, ocolor: str or tuple,
                    newcmt: CMTSource, ncolor: str or tuple,
                    minlon, maxlon, minlat, maxlat, padding):
    """Plots beachballs onto a map using their relative location"""

    ax = plt.gca()

    # Beachball width
    width_beach = min((maxlon + 2 * padding - minlon) / (5 * padding),
                      (maxlat + 2 * padding - minlat) / (5 * padding))

    # Get CMT location
    cmt_lat = oldcmt.latitude
    cmt_lon = oldcmt.longitude

    # Get new location
    new_cmt_lat = newcmt.latitude
    new_cmt_lon = newcmt.longitude

    # Correct plotting locations
    oldlat, oldlon, newlat, newlon, = \
        get_new_locations(cmt_lat, cmt_lon,
                          new_cmt_lat, new_cmt_lon,
                          padding)

    # Plot points
    markersize = 7.5
    ax.plot(cmt_lon, cmt_lat, "ko", zorder=200, markeredgecolor='k',
            markerfacecolor=ocolor, markersize=markersize)
    ax.plot(new_cmt_lon, new_cmt_lat, "ko", zorder=200, markeredgecolor='k',
            markerfacecolor=ncolor, markersize=markersize)

    # Plot lines
    ax.plot([cmt_lon, oldlon], [cmt_lat, oldlat], "k", zorder=199)
    ax.plot([new_cmt_lon, newlon], [new_cmt_lat, newlat], "k", zorder=199)

    # Old CMT
    ax = plt.gca()
    focmecs = oldcmt.tensor
    bb = beach(focmecs, xy=(oldlon, oldlat), facecolor=ocolor,
               width=width_beach, linewidth=1, alpha=1.0, zorder=250)
    ax.add_collection(bb)

    # New CMT
    new_focmecs = newcmt.tensor
    new_bb = beach(new_focmecs, facecolor=ncolor,
                   xy=(newlon, newlat), width=width_beach,
                   linewidth=1, alpha=1.0, zorder=250)
    ax.add_collection(new_bb)


def unique_locations(latitude, longitude):
    """Returns to lists of corresponding latitude and longitude. But only
    unique entries"""

    # Magic line
    stations = list(set([(lat, lon) for lat, lon in zip(latitude, longitude)]))

    # Lat list, Lon lisst
    return [sta[0] for sta in stations], [sta[1] for sta in stations]


def extract_locations_from_comp(complist: list):
    """Collects latitudes and longitudes in form on number of
    windows. if a trace has 3 windows the location will be added 3 times.
    Later when plotting the scattered stations, use a set comprehension.

    Argument:
        complist: is a list of all traces on one component
    """
    lat = []
    lon = []
    for trace in complist:
        for _i in range(trace["nwindows"]):
            lat.append(trace["lat"])
            lon.append(trace["lon"])

    return lat, lon


def extract_stations_from_traces(wave_dict):
    """Get all stations"""
    lat = []
    lon = []

    for wave in wave_dict.keys():
        for comp, complist in wave_dict[wave]["traces"].items():
            for trace in complist:
                lat.append(trace["lat"])
                lon.append(trace["lon"])

    return lat, lon


def get_azimuth(elat, elon, latitude, longitude):
    """ computes the azimuth for multiple stations

    Args:
        elat: event latitude
        elon: event longitude
        latitude: station latitudes
        lon: station longitudes
    Returns:

    """

    azi = []
    for lat, lon in zip(latitude, longitude):
        azi.append(gps2dist_azimuth(elat, elon, lat, lon)[1])
    return azi


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse
