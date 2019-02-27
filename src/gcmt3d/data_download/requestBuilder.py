"""

    This Function contains functions to build an FDSN time series download
    request from an input CMT source object and a station list.

"""

from gcmt3d.source import CMTSource


# Input Errors
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message



class dataRequest(object):
    """

    Class that handles the building of a data request

    """

    def __init__(self,cmt=None,stationlist=None,duration=0.0):
        """

        :param cmt: CMTSource object from cmt Source
        :param stationlist:

        """
        if type(cmt)!='CMTSource':
            raise InputError("No or non-CMTSource chosen as input\n\n",
                             "Choose a different CMT source.\n\n")
        self.origin_time = cmt.origin
        self.stationlist
        self.duration = duration


    def __str__(self):
        """
        key download info printing
        """
        return_str = 'request build'

        return return_str
