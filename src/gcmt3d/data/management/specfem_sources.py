'''

This script/set of functions is used to create a set of sources for parallel
specfem simulations

copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)


'''



from ...source import CMTSource
import os
import warnings
from copy import deepcopy



class SpecfemSources(object):
    '''
    This class handles the writing of specfem sources in form of CMT solutions
    '''

    def __init__(self, cmt, npar, dm=10.0*24, dx=2., ddeg=0.02, outdir=None):
        '''

        :param cmt: The original CMT source loaded using CMTSource
        :type cmt: CMTSource
        :param npar: Number of parameters to be inverted for: 6 - only moment
                     tensor, 7 - moment tensor and depth, 9 moment tensor, depth
                     and geolocation.
        :type npar: int
        :param dm: magnitude scalar -- we only need a scalar since the
                   derivative is independent of magnitude
        :type dm: float
        :param dx: depth change constant in m for Frechet derivative
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
            raise ValueError('Given CMT parameter not a CMTSource.')

        self.cmt = cmt

        if npar not in [6, 7, 9]:
            raise ValueError('The parameters to be inverted for must be an '
                             + 'integer between 6 and 9.')

        self.npar = npar

        if type(dm) != float:
            raise ValueError('Change in magnitude needs to be a float.')

        self.dm = dm

        if type(dx) != float:
            raise ValueError("Change in depth should be a float")

        self.dx = dx

        if type(ddeg) != float:
            raise ValueError("Change in degrees should be a float")

        self.ddeg = ddeg

        if type(ddeg) != float:
            raise ValueError("Change in degrees should be a float")

        self.ddeg = ddeg

        if outdir==None:
            raise ValueError("The output directory needs wo be set.")
        elif not os.path.exists(outdir):
            os.makedirs(outdir)
            warnings.warn("The chosen output directory does not exist.\n"
                          + "A new one will be created.")

        self.outdir = outdir

    def write_sources(self):
        """Function to write the CMT Solutions to CMT files
        """
        # Attribute list
        attr = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

        for index in range(6):

            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # make everything zero
            for attribute in attr:
                setattr(new_cmt, attribute, 0)

            # except one variable
            setattr(new_cmt, attr[index], self.dm)

            # write file
            new_cmt.write_CMTSOLUTION_file(os.path.join(self.outdir,
                                                        "CMTSOLUTION_M"
                                                        + attr[index][-2:]))

        if self.npar > 6:

            # Attribute name
            depth_str = "depth_in_m"

            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # change a depth
            setattr(new_cmt, depth_str, new_cmt.depth_in_m + self.dx)

            # write new solution
            new_cmt.write_CMTSOLUTION_file(os.path.join(self.outdir,
                                                        "CMTSOLUTION_"
                                                        + "depth"))

        if self.npar == 9:

            # Attribute name
            lat_str = "latitude"
            lon_str = "longitude"

            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # change a depth
            setattr(new_cmt, lat_str, new_cmt.latitude + self.ddeg)

            # write new solution
            new_cmt.write_CMTSOLUTION_file(os.path.join(self.outdir,
                                                        "CMTSOLUTION_"
                                                        + "lat"))
            # Create new CMT solution
            new_cmt = deepcopy(self.cmt)

            # change a depth
            setattr(new_cmt, lon_str, new_cmt.longitude + self.ddeg)

            # write new solution
            new_cmt.write_CMTSOLUTION_file(os.path.join(self.outdir,
                                                        "CMTSOLUTION_"
                                                        + "lon"))


    def __str__(self):
        string =  "-------- CMT Source Writer --------\n"
        string += "Number of parameters to invert for: %d\n" % self.npar
        string += "dM: %f in Nm\n" % self.dm

        if self.npar > 6:
            string += "dx: %f in m\n" % self.dx
        if self.npar >= 9:
            string += "ddeg: %f in degrees\n" % self.ddeg

        return string