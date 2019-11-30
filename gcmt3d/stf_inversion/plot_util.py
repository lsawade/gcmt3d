"""

This file contains functions and classes to plot stf inversion results.

:copyright:
    Lucas Sawade (lsawade@princeton.edu, 2019)

:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

import numpy as np
from numpy.fft import fftshift, fftfreq
from .stf import compute_error
import matplotlib.pyplot as plt


def plot_convergence(t, dt, sft, G, u):
    # Compute error
    euc, relerr = compute_error(t, dt, sft, G, u)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(euc, label="Euclidean norm error")
    ax[0].set_xlabel("Iteration #")
    ax[0].set_ylim([0, np.max(euc)])
    ax[1].plot(relerr, label="Euclidean norm error")
    ax[1].set_xlabel("Iteration #")
    plt.tight_layout()


def plot_stf(t, ft, fwt, iT, fT, skip=5):

    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.75))

    # STF plotting
    for _i, f in enumerate(ft[::skip]):
        ax[0].plot(t, f, c=(0.25, 0.25, 0.25, 1),
                   alpha=(len(ft[::skip])*0.05+_i)/(len(ft[::skip])*5))
    ax[0].plot(t, ft[-1], c=(0.8, 0.2, 0.2, 1))
    # ax[0].set_xlim([iT, fT])
    ax[0].set_xlabel('Time in [s]')

    # Frequency spectrum
    freq = fftshift(fftfreq(len(fwt[0]), (t[1]-t[0])))
    for _i, fw in enumerate(fwt[::skip]):
        ax[1].plot(freq, np.real(np.abs(fftshift(fw))),
                   c=(0.25, 0.25, 0.25, 1),
                   alpha=(len(fwt[::skip])*0.05+_i)/(len(fwt[::skip])*10))

    ax[1].plot(freq, np.real(np.abs(fftshift(fwt[-1]))), c=(0.8, 0.2, 0.2, 1))
    # ax[1].set_xlim([0, .5*np.max(1/(t[1]-t[0]))])
    # ax[1].set_ylim([0, np.max(fwt)])
    ax[1].set_xlabel('Frequency in [Hz]')
    plt.tight_layout()

    return ax
