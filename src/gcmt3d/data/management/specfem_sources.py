'''

This script/set of functions is used to create a set of sources for parallel
specfem simulations

'''



from ...source import CMTSource
from copy import deepcopy




class SpecfemSources(object):
    '''
    This class handles the writing of specfem sources in form of CMT solutions
    '''

    def __init__(self, cmt, npar, dm=1*10*22, dx=2, ddeg=0.02, outdir=None):
        '''

        :param cmt: The original CMT source loaded using CMTSource
        :type cmt: CMTSource
        :param npar: Number of parameters to be inverted for
        :type npar: int
        :param dm: magnitude scalar -- we only need a scalar since the
                   derivative is independent of magnitude
        :type dm: float
        :param dx: depth change constant for Frechet derivative
        :type dx: float
        :param ddeg: location change constant for Frechet derivative
        :type ddeg: float
        :param dtshift: change in cmt time shift for Frechet derivative
        :type dtshift: float
        :param dhdur: change in half duration for frechet derivative
        :type dhdur: float
        :param outdir: output directory for sources
        :type outdir: str

        '''

        if type(cmt) != CMTSource:
            raise ValueError('Given CMT parameter no a CMTSource.')

        if npar not in [6,7,8,9]:
            raise ValueError('The parameters to be inverted for must be an '
                             + 'integer between 6 and 9.')
