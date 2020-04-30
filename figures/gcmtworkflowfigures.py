"""

This directory contains function to reproduce figures that show parameters
for the Lamont Global CMT workflow, such as the data weighting and the
filtering


:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.orcopyleft/gpl.html)

Last Update: April 2020

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import seaborn as sns
from gcmt3d.plot.plot_util import remove_topright
from gcmt3d.data.management.process_classifier import filter_scaling
from gcmt3d.data.management.process_classifier import ProcessParams
from matplotlib.colors import ListedColormap

params = {
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.linewidth': 2,
    'xtick.labelsize': 11,
    'xtick.direction': 'in',
    'xtick.top': True,  # draw label on the top
    'xtick.bottom': True,  # draw label on the bottom
    'xtick.minor.visible': True,
    'xtick.major.top': True,  # draw x axis top major ticks
    'xtick.major.bottom': True,  # draw x axis bottom major ticks
    'xtick.minor.top': True,  # draw x axis top minor ticks
    'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
    'ytick.labelsize': 11,
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

def sac_cosine_taper(freqs, flimit):
    """ SAC style cosine taper.

    Args:
        freqs: vector to find a taper for
        flimit: corner frequencies

    Returns:

    """
    fl1, fl2, fl3, fl4 = flimit
    taper = np.zeros_like(freqs)

    # Beginning
    a = (fl1 <= freqs) & (freqs <= fl2)
    taper[a] = 0.5 * (1.0 - np.cos(np.pi * (freqs[a] - fl1) / (fl2 - fl1)))

    # Flat part
    b = (fl2 < freqs) & (freqs < fl3)
    taper[b] = 1.0

    # Ending
    c = (fl3 <= freqs) & (freqs <= fl4)
    taper[c] = 0.5 * (1.0 + np.cos(np.pi * (freqs[c] - fl3) / (fl4 - fl3)))

    return taper


def plot_tapers(outdir=None):
    """Plots the weighting as a function of Period and magnitude."""

    # Magnitudes
    m = np.linspace(7.0, 8.0, 500)

    # Periods
    p = np.linspace(75.0, 450.0, 600)

    # Config values
    startcorners = [125, 150, 300, 350]
    endcorners = [200, 400]
    startmag = 7.0
    endmag = 8.0

    # Preallocate
    tapers = np.zeros((len(p), len(m)))
    corners = np.zeros((4, len(m)))

    for _i, mm in enumerate(m):
        # Compute corners
        corners[:, _i] = filter_scaling(startcorners, startmag,
                                        endcorners, endmag, mm)
        # Compute tapers
        tapers[:, _i] = sac_cosine_taper(p, corners[:, _i])

    plt.figure()
    pc = plt.pcolormesh(p, m, tapers.T,
                        cmap=ListedColormap(sns.color_palette("Blues")))
    plt.colorbar(pc, orientation="horizontal")
    for _i in range(4):
        plt.plot(corners[_i, :], m, 'k')
    plt.xlabel("Period [s]")
    plt.ylabel(r"$M_w$")

    if outdir is not None:
        plt.savefig(os.path.join(outdir, "taperperiodmagnitude.png"), dpi=600)
    else:
        plt.show()


def plot_weighting(outdir=None):
    """Plot Weighting of the different wavetypes."""

    # Momentmanitude vecotr
    mw = np.linspace(5.0, 9.0, 100)

    # Weights
    bodywaveweights = np.zeros_like(mw)
    surfacewaveweights = np.zeros_like(mw)
    mantlewaveweights = np.zeros_like(mw)

    # Corners
    bodywavecorners = np.zeros((len(mw), 4))
    surfacewavecorners = np.zeros((len(mw), 4))
    mantlewavecorners = np.zeros((len(mw), 4))

    for _i, m in enumerate(mw):

        P = ProcessParams(m, 150000)
        P.determine_all()
        # assign
        bodywaveweights[_i] = P.bodywave_weight
        surfacewaveweights[_i] = P.surfacewave_weight
        mantlewaveweights[_i] = P.mantlewave_weight

        # Corners
        bodywavecorners[_i, :] = P.bodywave_filter
        surfacewavecorners[_i, :] = P.surfacewave_filter
        mantlewavecorners[_i, :] = P.mantlewave_filter

    fig = plt.figure(figsize=(10, 4))

    ax1 = plt.subplot(121)
    plt.plot(bodywaveweights, mw, "r", label="Body")
    plt.plot(surfacewaveweights, mw, "b:", label="Surface")
    plt.plot(mantlewaveweights, mw, "k", label="Mantle")
    plt.legend()
    plt.title("Weighting")
    plt.ylabel(r"$M_w$")
    plt.xlabel("Weight Contribution")

    ax2 = plt.subplot(122, sharey=ax1)

    for _i in range(4):
        if _i == 0:
            labelb = "Body"
            labels = "Surface"
            labelm = "Mantle"
        else:
            labels, labelb, labelm = None, None, None
        plt.plot(bodywavecorners[:, _i], mw, 'r', label=labelb)
        plt.plot(surfacewavecorners[:, _i], mw, 'b:', label=labels)
        plt.plot(mantlewavecorners[:, _i], mw, 'k', label=labelm)
    plt.title("Filter Corners")
    plt.xlabel("Period [s]")

    if outdir is not None:
        plt.savefig(os.path.join(outdir, "waveweighting.pdf"))
    else:
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="outdir", default=None, type=str,
                        help="Output directory", required=False)
    args = parser.parse_args()

    # Plot tapers.
    plot_tapers(outdir=args.outdir)

    plot_weighting(outdir=args.outdir)







