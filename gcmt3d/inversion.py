'''

This script/set of functions is used to create a set of sources for parallel
specfem simulations

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: January 2020

'''
import pycmt3d





class GCMT3D(object):

    def __init__(self, dcon: pycmt3d.data_container.DataContainer,
                 cmt3d_config: pycmt3d.Config,
                 grid3d_config: pycmt3d.grid3d.Grid3dConfig,
                 weight_config: pycmt3d.config.WeightConfig,
                 npar=9):

    pass
