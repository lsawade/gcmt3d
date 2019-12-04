"""
This file contains functions and classes to check whether the source
inversion works

:copyright:
    Lucas Sawade (lsawade@princeton.edu, 2019)

:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""

from .. stf_inversion import forward as fw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams['text.usetex'] = True


def main():

    # Start at random seed
    np.random.seed(260789)

    # Create synthetic data
    nt = 3002
    dt = 0.05
    nr = 20
    dx = 10.

    green_vel = np.array([20,  19,  19, 10, 13])

    # Data for Vertical component Z
    delta_loc_tz = np.array([50, 55, 57, 78, 96])
    delta_ampz = 3*dt*np.array([4,  -2,  -1, 3, 0.5])
    t, xrz, Gz, synz, obsz, wavelet = \
        fw.synthetic_traces(nr=nr, dx=dx, nt=nt, dt=0.05,
                            green_vel=green_vel,
                            delta_loc_t=delta_loc_tz,
                            delta_amp=delta_ampz,
                            amp1=2.5, to1=20, sig1=5,
                            amp2=1.75, to2=32, sig2=3)

    # # Data for radial R
    # delta_loc_tr = np.array([45, 53, 55, 75, 95])
    # delta_ampr = 3 * dt * np.array([4, -2.5, -.5, 3.5, -.5])
    # t, xrr, Gr, synr, obsr, waveletr = \
    #     fw.synthetic_traces(nr=nr, dx=dx, nt=nt, dt=0.05,
    #                         green_vel=green_vel,
    #                         delta_loc_t=delta_loc_tr,
    #                         delta_amp=delta_ampr,
    #                         amp1=2.5, to1=20, sig1=5,
    #                         amp2=1.75, to2=32, sig2=3)
    #
    # # Data for transverse T
    # delta_loc_tt = np.array([48, 53, 60, 75, 97])
    # delta_ampt = 3 * dt * np.array([3.5, -1.5, -2.5, 1, 1.5])
    # t, xrt, Gt, synt, obst, wavelet = \
    #     fw.synthetic_traces(nr=nr, dx=dx, nt=nt, dt=0.05,
    #                         green_vel=green_vel,
    #                         delta_loc_t=delta_loc_tt,
    #                         delta_amp=delta_ampt,
    #                         amp1=2.5, to1=20, sig1=5,
    #                         amp2=1.75, to2=32, sig2=3)

    # Create full matrices
    # G = np.concatenate((Gr, Gt, Gz), axis=0)
    # syn = np.concatenate((synr, synt, synz), axis=0)
    # obs = np.concatenate((obsr, obst, obsz), axis=0)

    G = Gz
    syn = synz
    obs = obsz

    # Plot one trace
    fw.plot_one_trace_set(t, wavelet, G[0, :], syn[0, :], obs[0, :])

    # Plot section
    fw.plot_section(t, syn, obs)

    # Compute Source time function with conventional waterlevel deconvolution
    lambd = 0.01  # waterlevel
    stf_decon, syn_decon1 = fw.deconvolution(obs, G, lambd)

    # Compute source time function with iterative landweber
    stf_lw, stf_list, chi_list = fw.landweber(obs, G, dt, maxT=50, crit=0.01,
                                              lamb=0.001, type="2")

    # Compare Source Time functions
    fw.plot_source_comparison(t, wavelet, [stf_decon, stf_lw],
                              ['Estimate LS', 'Landweber'])

    # Compute new synthetics
    syn_decon = fw.compute_synth(G, stf_decon)
    syn_lw = fw.compute_synth(G, stf_lw)

    # Plot section comparison
    fw.plot_comparison_section(t, obs, syn, syn_decon, syn_lw)

    # Plot Stf evolution
    fw.plot_stf_evolution(t, wavelet, stf_list, skip=5)

    # Compute
    plt.show()


if __name__ == "__main__":
    main()
