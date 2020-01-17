"""

This script contains functions to convolve the Green's functions in a Stream
with a source time function.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: November 2019

"""

import numpy as np


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


# class Wavelet(object):
#     """Contains wavelet"""
#
#     def __init__(self, t, data=None, type='gaussian', **kwargs):
#         """ Creates wavelet with parameters
#
#         :param dt: sampling time
#         :param N:
#         :param type: type of automatically created wavelet. Irrelevant if
#                      data is input.
#         :param
#         :param **kwargs: keyword arguments depending on the type of wavelet.
#                          'gaussian' or 'bimodal-gaussian'
#
#         """
#
#         self.t = t
#
#         if data is not None:
#             self.data = data
#             self.type = 'self-defined'
#         else:
#             if type == 'gaussian':
#                 self.wavefun = gaussian
#                 self.data = self.wavefun(t, *kwargs)
#
#     @classmethod
#     def _from_SAC(cls, filename):
#         """ Creates
#
#         :param filename: filename to create wavelet from
#         :type: str
#         :return:
#
#         """
#         pass
#
#     @classmethod
#     def _from_cmtsource(cmtsource, dt):
#         """CMTSource object provides halfduration and time shift,
#         as well as approximation of the Gaussian.
#
#         From Specfem Manual:
#
#             The zero time of the simulation corresponds to the center
#             of the triangle/Gaussian, or the centroid time of the
#             earthquake. The start time of the simulation is
#             t = −1.5 ∗ half duration (the 1.5 is to make sure the
#             moment rate function is very close to zero when starting
#             the simulation). To convert to absolute time tabs, set
#
#                 t_abs = tpde + time shift + tsynthetic
#
#             where tpde is the time given in the first line of
#             the CMTSOLUTION, time shift is the corresponding
#             value from the original CMTSOLUTION file and tsynthetic
#             is the time in the first column of the output seismogram.
#
#         Arguments:
#             cmtsource (CMTSource object): Input CMTSource object inlcuding
#                                           necessary info such as the half
#                                           duration, time shift, and, of
#                                           course, origin time.
#
#         Returns:
#             Wavelet class: [description]
#         """
#
#         return cls()


# def convolve_trace(tr: obspy.trace, wavelet):
#     """Takes in a wavelet and an obspy Trace object and convolves the
#     wavelet with the Trace objectss
#
#     Args:
#         tr (obspy.Trace): Trace object
#         wavelet ([type]): wavelet
#     """
#     pass
#
