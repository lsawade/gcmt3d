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
import lwsspy as lpy
from gcmt3d.plot.plot_util import remove_topright
from gcmt3d.data.management.process_classifier import filter_scaling
from gcmt3d.data.management.process_classifier import ProcessParams
from matplotlib.colors import ListedColormap

lpy.updaterc()

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

def plot_tapers(ax=None, outdir=None):
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

    if ax is None:
        ax = plt.gca()
    pc = ax.pcolormesh(p, m, tapers.T,
                       cmap=ListedColormap(plt.get_cmap(
                           'PuBu')(np.linspace(0, 1, 5)), N=5),
                       zorder=-10)
    lpy.nice_colorbar(pc, orientation="vertical", fraction=0.05, aspect=20,
                      pad=0.05)
    for _i in range(4):
        plt.plot(corners[_i, :], m, 'k')
    # plt.xlabel("Period [s]")
    # plt.ylabel(r"$M_w$")

    if outdir is not None:
        plt.savefig(os.path.join(outdir, "taperperiodmagnitude.png"), dpi=300)


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

    fig = plt.figure(figsize=(8, 3))
    gs = fig.add_gridspec(1, 8)
    ax1 = fig.add_subplot(gs[0, :3])
    lpy.plot_label(ax1, "a", aspect=1, location=6, dist=0.025, box=False)
    plt.subplots_adjust(bottom=0.175, left=0.075, right=0.925, wspace=0.75)
    plt.plot(bodywaveweights, mw, "r", label="Body")
    plt.plot(surfacewaveweights, mw, "b:", label="Surface")
    plt.plot(mantlewaveweights, mw, "k", label="Mantle")
    plt.legend()
    plt.ylabel(r"$M_w$")
    plt.xlabel("Weight Contribution")

    ax2 = fig.add_subplot(gs[0, 3:], sharey=ax1)
    ax2.tick_params(labelleft=False)
    lpy.plot_label(ax2, "Filter Corners", aspect=1,
                   location=4, dist=0.025, box=False)
    lpy.plot_label(ax2, "b", aspect=1, location=7, dist=0.025, box=False)

    # Inset axis for the taper visualization.
    axins = ax2.inset_axes([-0.05, 0.8, 0.4, 0.25])
    axins.set_rasterization_zorder(-5)
    plot_tapers(ax=axins)

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
        axins.plot(bodywavecorners[:, _i], mw, 'r', label=labelb)
        axins.plot(surfacewavecorners[:, _i], mw, 'b:', label=labels)
        axins.plot(mantlewavecorners[:, _i], mw, 'k', label=labelm)
        
    plt.xlabel("Period [s]")

    # Set axes for the insett
    axins.set_ylim((7.0, 8.0))
    axins.set_xlim((75.0, 450.0))
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax2.indicate_inset_zoom(axins)

    if outdir is not None:
        plt.savefig(os.path.join(outdir, "waveweighting.pdf"), dpi=900)
    else:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="outdir", default=None, type=str,
                        help="Output directory", required=False)
    args = parser.parse_args()

    # Plot tapers.
    # plot_tapers(outdir=args.outdir)

    plot_weighting(outdir=args.outdir)

