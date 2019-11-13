"""

This file contains functions and classes to perform a sourcetime function
inversion using the projected landweber method as shown to work for seismology
by Bertero et al. as well as Valleé et al.

:copyright:
    Lucas Sawade (lsawade@princeton.edu, 2019)

:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

from numpy.fft import fft, ifft, rfft, irfft, fftfreq
from numpy import correlate
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dc


def conj_grad(G, u, t, dt, tau_factor=1, iT=0, fT=120, niter=10):
    """Computing one landweber iteration of the signal.
    param. Refer back to Bertero et al."""

    # Check which is longer
    N = len(G)

    # Positive part of t
    Fs = 1 / (N * dt)
    tshift = -np.min(t)
    freq = fftfreq(N, d=dt)
    freq_shift = np.exp(-1.j * freq * 2 * np.pi * tshift)

    # Fourier Transform stuff
    Gw = fft(G)
    uw = fft(u)

    # Create first iteration
    # Create first iteration
    f = correlate(u, G, 'same')
    f = np.real(ifft(uw/Gw*freq_shift))

    plt.figure()
    plt.plot(f)

    fw = fft(f)

    # Empty list for all iterations
    ft = []
    fwt = []

    # Actual Method
    r = uw - Gw * fw
    p = r
    rsold = np.dot(r, np.conjugate(r))

    for i in range(niter):
        ffp = fft(p)
        Ap = Gw * ffp
        alpha = rsold / (np.conjugate(ffp)* Ap)

        g = np.real(ifft( (f + alpha * p) * freq_shift) )

        f = lw_projection(g, t, iT=iT, fT=fT)
        ft.append(f)
        fw = fft(f)
        fwt.append(fw)

        r = r - alpha * Ap

        rsnew = np.dot(r, np.conjugate(r))

        if np.sqrt(rsnew) < 1e-10:
            break

        p = r + (rsnew / rsold) * p

        rsold = rsnew

    return fwt, ft



def landweber(Gw, uw, tau, f, freq_shift):
    """Computing one landweber iteration of the signal.
    param. Refer back to Bertero et al."""

    # We have to shift fw in time to account for the negative part
    # time values in the signal
    gp1 = f + tau * np.real(ifft((np.conj(Gw) * freq_shift * (uw - Gw * fft(
        f)))))

    return gp1


def lw_projection(g, t, iT=2, fT=120):
    """Projects a trace into the causal non-negative Convex space."""

    # Copy g
    f = dc(g)

    # Apply projection parameters
    f[np.where(t < iT)[0]] = 0
    f[np.where(t > fT)[0]] = 0
    f[np.where(f < 0)[0]] = 0

    return f


def projected_landweber(G, u, t, dt, tau_factor=1, iT=0, fT=120, niter=10):

    # Check which is longer
    N = len(G)

    # Positive part of t
    Fs = 1/(N*dt)
    tshift = -np.min(t)
    freq = fftfreq(N, d=dt)
    freq_shift = np.exp(-1.j*freq*2*np.pi*tshift)

    # Fourier Transform stuff
    Gw = fft(G)
    uw = fft(u)

    # Create first iteration
    fp1 = np.zeros(Gw.size)

    # Compute tau
    tau = 0.05/np.max(np.abs(Gw))

    # Empty list for all iterations
    ft = []
    fwt = []

    for i in range(niter):

        # Compute the landweber iteration
        g = landweber(Gw, uw, tau, fp1, freq_shift)

        # Compute the projection onto the convex set.
        fp1 = lw_projection(g, t, iT=iT, fT=fT)

        # Normalize by integrated sum
        fp1 = fp1
        # Save STF iteration
        ft.append(fp1)

        # Save STF Frequency spectrum
        fwt.append(fft(fp1))

    return fwt, ft


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def compute_error(t, dt, stf_list, green, disp):
    """Compute both euclidean norm for every iteration as
    well as relative error between iterations"""


    N = len(green)

    # Positive part of t
    tshift = np.min(t)

    # Get Frequencies
    freq = fftfreq(N, dt)
    freq_shift = np.exp(-1.j*freq*2*np.pi*tshift)

    euc = np.zeros(len(stf_list))
    relerr = np.zeros(len(stf_list))
    for _i, stf in enumerate(stf_list):

        Ag = np.real(ifft(fft(green) * fft(stf) * freq_shift))
        # Frobenius norm between data and simulation
        euc[_i] = l2_norm(np.abs(np.real(Ag) - disp)) / l2_norm(disp)

        # Frobenius norm between iterations
        if _i > 0:
            relerr[_i] = l2_norm(stf_list[_i] - stf_list[_i - 1]) / l2_norm(
                stf_list[_i - 1])

    relerr[0] = 1

    return euc, relerr


class Operator(object):
    """Linear operator"""

    def __init__(self, t, wavelet):

        # Time Vector
        self.t = t
        self.dt = t[1] - t[0]
        self.N = len(t)

        # Corresponding frequency
        self.freq = fftfreq(self.N, self.dt)

        # possible timeshift due to change in from zero
        if np.min(t) != 0:
            self.tshift = -np.min(t)
            self.freq_shift = np.exp(-1.j*self.freq*2*np.pi*self.tshift)
        else:
            self.freq_shift = 1

        # Time Vector
        self.wavelet = wavelet


    def forward(self, m):
        """Forward operator

        :param m: model
        :return: conveolved data
        """
        return np.real(ifft(fft(self.wavelet)*fft(self.m)*self.freq_shift))

    def adjoint(self, m):
        """Adjoint operator

        :param model: model
        :return: convolved data

        """
        return np.real(ifft(fft(self.wavelet) * np.conjugate(fft(self.m)) *
                            self.freq_shift))


class stf_operator(object):
    """Linear operator"""

    def __init__(self, t, g):
        """

        :param t: time vector
        :param g: greens function
        """

        # Time Vector
        self.t = t
        self.dt = t[1] - t[0]
        self.N = len(t)

        # Corresponding frequency
        self.freq = fftfreq(self.N, self.dt)

        # possible timeshift due to change in from zero
        if np.min(t) != 0:
            self.tshift = -np.min(t)
            self.freq_shift = np.exp(-1.j*self.freq*2*np.pi*self.tshift)
        else:
            self.freq_shift = 1

        # Time Vector
        self.g = g


    def forward(self, wavelet):
        """Forward operator

        :param m: model
        :return: conveolved data
        """
        return np.real(ifft(fft(wavelet)*fft(self.g)*self.freq_shift))

    def adjoint(self, wavelet):
        """Adjoint operator

        :param model: model
        :return: convolved data

        """
        return np.real(ifft(fft(wavelet) * np.conjugate(fft(self.g)) *
                            self.freq_shift))

class stf_inversion(object):
    """Defines inversion operation."""

    def __init__(self, t, d, g, iT, fT):
        """ Initialize the stf_inversion class

        :param d: data
        :param g: green's function
        """
        self.t = t
        self.dt = t[1] - t[0]
        self.iT = iT
        self.fT = fT
        self.d = d
        self.g = g
        self.n = len(d)
        self.F = stf_operator(t, g)

    def proj_lw(self):
        """Inverts data using projected Landweber method"""

        return projected_landweber(self.g, self.d, self.t, self.dt,
                                   tau_factor=1, iT=0, fT=10, niter=20)


    def conjugate_gradient(self):
        """inverts data"""

        # List to save iterations
        stf_list = []
        fft_list = []
        cost_list = []

        # First Model estimate
        stf = np.zeros_like(self.d)

        # Compute residual (cost function)
        r = self.d - self.F.forward(stf)

        # gradient direction
        s = np.zeros_like(self.d)

        # Initialize beta
        beta = 0

        for i in range(215):

            # Compute gradient
            g = self.F.adjoint(r)

            if i != 0:
                beta = np.dot(g, g) / gamma

            #
            gamma = np.dot(g, g)

            # Compute decent direction
            s = g + beta * s

            # delta (cost function)
            deltar = self.F.forward(s)

            # Scaling factor
            alpha = gamma / np.dot(deltar, deltar)

            # decent
            stf = lw_projection(stf + alpha * s,  self.t,
                                iT=self.iT, fT=self.fT)

            # stf = stf + alpha * s
            fft_list.append(stf)

            # cost
            r = r - alpha * deltar

            stf_list.append(stf)
            cost_list.append(r)

        return fft_list, stf_list, cost_list
