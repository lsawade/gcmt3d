"""

Plot utilities not to modify plots or base plots.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2020


"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


params = {
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'xtick.direction': 'in',
    'xtick.top': True,   # draw label on the top
    'xtick.bottom': True,    # draw label on the bottom
    'xtick.minor.visible': True,
    'xtick.major.top': True,  # draw x axis top major ticks
    'xtick.major.bottom': True,  # draw x axis bottom major ticks
    'xtick.minor.top': True,  # draw x axis top minor ticks
    'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
    'ytick.labelsize': 7,
    'ytick.direction': 'in',
    'ytick.left': True,  # draw label on the top
    'ytick.right': True,  # draw label on the bottom
    'ytick.minor.visible': True,
    'ytick.major.left': True,  # draw x axis top major ticks
    'ytick.major.right': True,  # draw x axis bottom major ticks
    'ytick.minor.left': True,  # draw x axis top minor ticks
    'ytick.minor.right': True,  # draw x axis bottom minor ticks
    # 'text.usetex': True,
    # 'font.family': 'STIXGeneral',
    # 'mathtext.fontset': 'cm',
}
matplotlib.rcParams.update(params)


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


def set_mpl_params_section():
    params = {
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': "bold",
        'xtick.labelsize': 10,
        'xtick.direction': 'in',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'ytick.labelsize': 10,
        'ytick.direction': 'in',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        # 'text.usetex': True,
        # 'font.family': 'STIXGeneral',
        # 'mathtext.fontset': 'cm',
    }
    matplotlib.rcParams.update(params)
