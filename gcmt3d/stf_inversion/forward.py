"""

This file contains functions and classes to perform forward and adjoint
modelling of waveforms

:copyright:
    Lucas Sawade (lsawade@princeton.edu, 2019)

:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift

import matplotlib.pyplot as plt


def gaussian(t, to, sig, amp):
    """Create Gaussian pulse.

    :param t: time vector
    :param to: time in the center
    :param sig: standard deviation
    :param amp: amplitude
    :return: vector with same size of t containing the corresponding pulse

    """
    return amp * np.exp(-0.5 * (t-to)**2 / (sig * sig))


def bimodal_gaussian(t, amp1=2.5, to1=20, sig1=5, amp2=1.75, to2=32, sig2=3):
    """ Creates bimodal gaussian from gaussian function.

    :param t: time vector
    :param amp1: amplitude of first gaussian
    :param to1: center time of first gaussian
    :param sig1: standard deviation of first gaussian
    :param amp2: amplitude of second gaussian
    :param to2: center time of second gaussian
    :param sig2: standard deviation of second gaussian
    :return:
    """

    g1 = gaussian(t, to1, sig1, amp1)
    g2 = gaussian(t, to2, sig2, amp2)

    return g1 + g2


def single_greens_function(t, dt, delta_loc_t, delta_amp):
    """Create a single, simple Green's function from time locations and
    amplitudes of delta functions.

    :param t: time vector
    :param dt: sampling time
    :param delta_loc_t: time locations of Green's function peaks
    :param delta_amp: corresponding amplitudes
    :return: Green's function corresponding to time vector t

    """

    # Create empty Green's function
    green = np.zeros(len(t))

    # Populate it using the list
    for _i, (delta_t, delta_a) in enumerate(zip(delta_loc_t, delta_amp)):
        it = np.argmin(np.abs(delta_t-t))
        green[it] = delta_a

    return green


def synthetic_seismogram(green, wavelet):
    """Takes in a Gaussian and a Green's function and computes the
    seismogram.

    :param green: Green's function
    :param wavelet: source wavelet
    :return: seismogram
    """
    return np.real(ifft(fft(wavelet) * fft(green)))


def noisy_seismogram(t, seismogram, noise_amp=5):
    """ Takes in t and a seismogram to compute the

    :param t:
    :param seismogram:
    :return:
    """

    # Noise
    signoise = 2 * np.sqrt(3)

    # Create filter to take out high frequency noise
    filtgauss = gaussian(t, 75, signoise, 1.)
    filtgauss = filtgauss / sum(filtgauss)

    # Amplitude for the noise
    amp = noise_amp
    noise = 2 * (np.random.uniform(size=t.shape) - 0.5) * amp

    # Compute filtered noise
    filtnoise = np.real(ifft(fft(noise) * fft(filtgauss)))

    # Add noise to original seismogram
    noisemogram = seismogram + filtnoise

    return noisemogram


def synthetic_traces(nr=20, dx=10, nt=3001, dt=0.01, green_vel=None,
                     delta_loc_t=None, delta_amp=None,
                     amp1=2.5, to1=20, sig1=5, amp2=1.75, to2=32, sig2=3,
                     noise_amp=5):
    """
    :param nr: Number of receivers
    :param dx: receiver spacing
    :param nt: number of samples
    :param dt: sampling interval
    :param green_vel: green's function velocities between the peaks
                      Note that green_vel, delta_loc_t, delta_amp have to
                      have the same number of elements
    :param delta_loc_t: initial locations of the green's function things
    :param delta_amp: corresponding amplitudes
    :param amp1: amplitude of first gaussian
    :param to1: center time of first gaussian
    :param sig1: standard deviation of first gaussian
    :param amp2: amplitude of second gaussian
    :param to2: center time of second gaussian
    :param sig2: standard deviation of second gaussian
    :param noise_amp: amplitude of the noise, the noise is filtered with a
                      gaussian to avoid high frequency noise.
    :return: time vector, space vector and set of green's functions, synthetic
    traces and their
    noisified
             corresponding observed data --> (t, xr, Garray, Sarray, Oarray)
    """

    # Time parameters
    t = np.arange(nt)*dt

    # Multiple seismograms
    obs = np.zeros((nr, nt))
    syn = np.zeros((nr, nt))
    G = np.zeros((nr, nt))
    xr = []

    # Source-time function
    wavelet = bimodal_gaussian(t, amp1, to1, sig1, amp2, to2, sig2)

    # iterate over receivers
    for ir in range(nr):

        # Receiver distance from 0
        xr.append(ir * dx)

        # Compute velocity dependent time shift for Green's functions
        tshift = xr[ir] / green_vel

        # Change locations for initial location of the delta functions in trace
        delta_loc_t_tmp = delta_loc_t + tshift

        # Compute green's function using artificial damping
        idecay = np.arange(delta_amp.shape[0])
        delta_amp_tmp = delta_amp * (1-0.99*idecay/nr)

        # Compute green's functions
        G[ir, :] = single_greens_function(t, dt,
                                          delta_loc_t_tmp,
                                          delta_amp_tmp)

        # Compute seismograms
        syn[ir, :] = np.real(ifft(fft(wavelet)*fft(G[ir, :])))

        # Compute noisy, 'observed' seismograms
        obs[ir, :] = noisy_seismogram(t, syn[ir, :], noise_amp=noise_amp)

    return t, np.array(xr), G, syn, obs, wavelet


def plot_one_trace_set(t, wavelet, G, syn, obs):
    """Plots synthetic data.
    :param t: time vector
    :param wavelet: stf
    :param G: green's function
    :param syn: synthetic
    :param obs: observed
    :return:
    """

    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 6))

    # True STF
    ax[0].plot(t, wavelet)
    ax[0].set_xlabel('Time in s')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('True source wavelet')

    # Green function
    ax[1].plot(t, G)
    ax[1].set_xlabel('Time in s')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('Green function')

    # Seismograms
    ax[2].plot(t, syn, label='Synthetic')
    ax[2].plot(t, obs, label='Observed')
    ax[2].set_xlabel('Time in s')
    ax[2].set_ylabel('Amplitude')
    ax[2].set_title('Seismograms')
    plt.legend(loc='best')
    ax[2].set_xlim([np.min(t), np.max(t)])

    plt.tight_layout()

    plt.show(block=False)


def plot_section(t, syn, obs):
    """Plots section of data of the synthetic and observed data"""

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 8))

    for ir in range(syn.shape[0]):

        # Synthetic
        ax[0].plot(t, ir + syn[ir, :], 'k')
        ax[0].set_xlabel('Time in s')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_title('Synthetic')

        # Noisy observed data
        ax[1].plot(t, ir + obs[ir, :], 'k')
        ax[1].set_xlabel('Time in s')
        ax[1].set_title('Observed')
        ax[1].set_xlim([np.min(t), np.max(t)])
        ax[1].set_ylim([-1, syn.shape[0]+1.5])

    plt.tight_layout()

    plt.show(block=False)


def compute_synth(green, src):
    """ Convolution of set of Green's functions

    :param green: Green's function
    :param src:
    :return:
    """
    # Get frequency spectrum of source time function
    SRC = fft(src)

    # Get frequency spectra of Green's functions
    GRE = fft(green, axis=1)

    # Convolve the two and return matrix containing the synthetic
    syn = np.real(ifft(GRE*SRC, axis=1))

    return syn


def deconvolution(obs, green, lambd):
    """ Takes in the observed data and the green's function to obtain the
    source wavelet estimate.

    :param obs:
    :param green:
    :return:
    """

    nr, nt = obs.shape
    num = np.zeros(nt)
    den = np.zeros(nt)

    for ir in range(len(obs)):

        OBS = fft(obs[ir, :])
        GRE = fft(green[ir, :])

        # Sum all
        num = num + np.conj(GRE) * OBS
        den = den + np.conj(GRE) * GRE

    # Get maximum value of denominator
    maxden = np.max(np.abs(den))

    # Waterlevel
    wl = lambd * maxden

    # Deconvolution using the waterlevel
    src = np.real(ifft(num / (den+wl).T))

    # Compute fit to original data
    res = obs
    chi0 = 0.5 * np.sum(np.sum(res ** 2))

    syn = compute_synth(green, src)
    res = obs - syn
    chi = 0.5 * np.sum(np.sum(res ** 2))

    print(chi/chi0)

    return src, syn


def plot_source_comparison(t, wavelet, inversions, labels):
    """

    :param t: time vector
    :param wavelet: original wavelet
    :param inversions: list of inverted
    :params labels: labels corresponding to the inverted stfs
    :return:
    """

    plt.figure(figsize=(10, 6))
    plt.plot(t, wavelet, label="Original")
    for stf, label in zip(inversions, labels):
        plt.plot(t, stf, label=label)
    plt.legend()
    plt.show(block=False)


def landweber(obs, G, dt, maxT, crit, lamb, type="2"):
    """

    :param obs: observed traces
    :param G: Green's functions
    :param maxT: time after which STF is forced to zero
    :param crit: critical value for stopping the iteration
    :param dt: time sampling
    :param lamb: waterlevel for deconvolution if type 2 is chosen. Unused if
                 type is "1".
    :param type: string defining the type of landweber method. Type 1 is the
                 method using the steepest decent; type 2 is using a Newton
                 step.
    :return:
    """

    if type == "1":
        compute_gradient = compute_gradient_sd
    elif type == "2":
        compute_gradient = compute_gradient_newton
    else:
        raise ValueError("Wrong type!")

    # Get informations about size and initialize src
    nr, nt = obs.shape
    src = np.zeros(nt)

    # Compute objective function and gradient
    syn = compute_synth(G, src)
    res = obs - syn
    chi = 0.5 * np.sum(np.sum(res ** 2))
    chi0 = chi
    grad, alpha = compute_gradient(res, G, lamb)

    # Manage windowing tapered window in the future? e.g., tukey?
    itstop = int(np.floor(maxT / dt) + 1)
    Ptime = np.ones(nt)
    Ptime[itstop + 1: itstop + 31] = np.arange(30, 0, -1) / 30.
    Ptime[itstop + 31:] = 0.

    # Perform iterative deconvolution (inverse problem)
    it = 1
    perc = 0.05
    chip = chi0
    # llb = 0.1

    # Initialize list to save iterations
    src_list = []
    chi_list = []

    while chi > crit * chi0 and it <= nt:

        # Regularized gradient
        gradreg = grad

        if type == "1":
            srctmp = src + gradreg
        else:
            srctmp = src + perc * gradreg

        # Window source time function --> zero after some time T
        srctmp = srctmp * Ptime

        # Enforce positivity
        srctmp[np.where(srctmp < 0)[0]] = 0

        # Compute misfit function and gradient
        syn = compute_synth(G, srctmp)
        res = obs - syn
        chi = 0.5 * np.sum(np.sum(res ** 2))
        grad, _ = compute_gradient(res, G, lamb)
        it = it + 1

        # Check convergence
        if chi > chip:
            print("NOT CONVERGING")
            break

        # Update
        # chi / chi0
        chip = chi
        src = srctmp

        chi_list.append(chi)
        src_list.append(src)

    # Final misfit function
    print(chi / chi0)
    print(it)

    return src, src_list, chi_list


def compute_gradient_newton(resid, green, lamb):
    """ Compute Gradient using the waterlevel deconvolution which computes
    the Newton Step.

    :param resid: residual
    :param green: green's function
    :param lamb: waterlevel scaling
    :return:
    """
    # Get infos
    nr, nt = green.shape

    # FFT of residuals and green functions
    RES = fft(resid, axis=1)
    GRE = fft(green, axis=1)

    # Compute gradient (full wavelet estimation)
    num = np.sum(RES * np.conj(GRE), axis=0)
    den = np.sum(GRE * np.conj(GRE), axis=0)

    # Waterlevel?
    wl = lamb * np.max(np.abs(den))
    grad = np.real(ifft(num / (den + wl)))

    # Step value
    hmax = 1

    return grad, hmax


def compute_gradient_sd(resid, green, lamb):
    """ Compute the Gradient using the steepest decent method
    :param resid:
    :param green:
    :return:
    """

    # FFT of residuals and green functions
    RES = fft(resid, axis=1)
    GRE = fft(green, axis=1)

    # Compute gradient (full wavelet estimation)
    num = np.sum(RES * np.conj(GRE), axis=0)
    den = np.sum(GRE * np.conj(GRE), axis=0)

    # Waterlevel?
    tau = 1/np.max(np.abs(den))
    grad = np.real(ifft(num * tau))

    # Step value
    hmax = 1

    return grad, hmax


def plot_comparison_section(t, obs, syn, syn_decon, syn_lw):
    """Plot section to compare convolution results

    :param t: time vector
    :param obs: observed data
    :param syn_decon: synthetics using waterlevel decon
    :param syn_lw: synthetics using landweber iterative.
    :return: plot
    """

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 8))

    for ir in range(obs.shape[0]):
        # Observed + Standard deconvolution
        if ir == 0:
            label_obs = 'Obs'
            label_syn = 'Syn'
            label_syn_decon = 'Inv'
        else:
            label_obs = None
            label_syn = None
            label_syn_decon = None

        ax[0].plot(t, ir + obs[ir, :], 'k', label=label_obs)
        ax[0].plot(t, ir + syn[ir, :], c=(0.85, 0.2, 0.2), label=label_syn)
        ax[0].plot(t, ir + syn_decon[ir, :], c=(0.2, 0.2, 0.85),
                   label=label_syn_decon)
        ax[0].set_xlabel('Time in [s]')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_title('Waterlevel')

        # Observed + Landweber iterative deconvolution
        ax[1].plot(t, ir + obs[ir, :], 'k')
        ax[1].plot(t, ir + syn[ir, :], c=(0.85, 0.2, 0.2))
        ax[1].plot(t, ir + syn_lw[ir, :], c=(0.2, 0.2, 0.85))
        ax[1].set_xlabel('Time in [s]')
        ax[1].set_title('Landweber')
        ax[1].set_xlim([np.min(t), np.max(t)])
        ax[1].set_ylim([-1, syn_lw.shape[0] + 1.5])

    plt.figlegend(frameon=True, fancybox=False, loc='lower center', ncol=3,
                  framealpha=1, edgecolor='k', facecolor='w',
                  bbox_to_anchor=(0.52, 0.925))
    plt.tight_layout()

    plt.show(block=False)


def plot_stf_evolution(t, stf, stf_list, skip=5):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # STF plotting
    for _i, f in enumerate(stf_list[::skip]):
        ax[0].plot(t, f,
                   c=(0.25, 0.25, 0.25, 1),
                   alpha=((len(stf_list[::skip]) * 0.05 + _i)
                          / (len(stf_list[::skip]) * 2)),
                   label=None)
    ax[0].plot(t, stf_list[-1], c=(0.8, 0.2, 0.2, 1), label="Inversion")
    ax[0].plot(t, stf, c=(0.2, 0.2, 0.8, 1), label="Original")
    # ax[0].set_xlim([iT, fT])
    ax[0].set_xlabel('Time in [s]')
    plt.legend()

    # Frequency spectrum
    freq = fftshift(fftfreq(len(stf_list[0]), (t[1]-t[0])))
    for _i, f in enumerate(stf_list[::skip]):
        ax[1].plot(freq, np.real(np.abs(fftshift(fft(f)))),
                   c=(0.25, 0.25, 0.25, 1),
                   alpha=((len(stf_list[::skip])*0.05+_i)
                          / (len(stf_list[::skip]) * 2)))

    ax[1].plot(freq, np.real(np.abs(fftshift(fft(stf_list[-1])))),
               c=(0.8, 0.2, 0.2, 1), label="Inversion")
    ax[1].plot(freq, np.real(np.abs(fftshift(fft(stf)))),
               c=(0.2, 0.2, 0.8, 1),
               label="Original")
    plt.legend()
    ax[1].set_xlim([0, .5*np.max(1/(t[1]-t[0]))])
    ax[1].set_ylim([0, np.max(stf_list)])
    ax[1].set_xlabel('Frequency in [Hz]')
    plt.tight_layout()

    plt.show(block=False)

    return ax
