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

class Wavelet(object):
    """Contains wavelet"""

    def __init__(self, t, data=None, type='gaussian', **kwargs):
        """ Creates wavelet with parameters

        :param dt: sampling time
        :param N:
        :param type: type of automatically created wavelet. Irrelevant if
                     data is input.
        :param
        :param **kwargs: keyword arguments depending on the type of wavelet.
                         'gaussian' or 'bimodal-gaussian'

        """

        self.t = t

        if data is not None:
            self.data = data
            self.type = 'self-defined'
        else:
            if type=='gaussian':
                self.wavefun = gaussian
                self.data = self.wavefun(t, *args)




    @classmethod
    def wavelet_from_SAC(cls, filename):
        """ Creates

        :param filename: filename to create wavelet from
        :type: str
        :return:

        """
        pass

        # return cls()




def convolve_stream(st, wavelet):
    """Takes in a wavelet"""

    pass