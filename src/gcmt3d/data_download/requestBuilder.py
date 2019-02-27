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
        print(self.expression + "\n\n")
        print(self.message + "\n")


class dataRequest(object):
    """

    Class that handles the building of a data request

    """

    def __init__(self, cmt=None, stationlist=None, channels=[], locations=[],
                        duration=0.0):
        """
        :param cmt: CMTSource object from cmt Source
        :param stationlist: list of station in format
                            `[['Net1','Sta1'],['Net2','Sta2'],etc.]`
        :param channels: list of strings containing wanted channels
        :param locations: list of seismometer locations
        :param duration: duration of the requested recording from origin time
        :param :
        """

        # Check if CMT input is a CMT object
        if type(cmt) != CMTSource:
            raise InputError("dataRequest initialization",
                             "None or incorrect CMT input.")

        # CMT parameter input
        self.origin_time = cmt.cmt_time
        self.origin_latitude = cmt.latitude
        self.origin_longitude = cmt.longitude
        self.eventname = cmt.eventname

        # Check if station list is a 2D list of networks and stations
        if type(stationlist) != list:
            raise InputError("dataRequest initialization",
                             "stationlist not a list")

        # Station list parameter setup
        self.stationlist = stationlist


        # Download parameters
        self.duration  = duration
        self.channels  = channels
        self.locations = locations


    @classmethod
    def from_file(cls,cmtfname,
                      stationlistfname,
                      channels=[],
                      locations=[],
                      duration=0.0):
        """Creates a downloader class from an input file

        This downloader class also needs to contain the ouput directory for the
        traces. Take CMT fname
        """

        # First Create CMT Solution
        cmt = CMTSource.from_CMTSOLUTION_file(cmtfname)

        # Read station list file with two columns and whitespace separation
        statfile = open(stationlistfname,'r')

        # For loop to add all stations to the station list
        stationlist = [];
        for line in statfile:
            if line=="NETWORK	STATION":
                continue

            # Read stations into list of stations
            line = line.split()

            # Append the network o
            stationlist.append([line[0],line[1]])

        return cls(cmt=cmt,
                   stationlist=stationlist,
                   channels=channels,
                   locations=locations,
                   duration=duration)


    def request(self):
        """
        :returns str: with the actual url request
        """
        pass

    def download(self,path):
        """
        takes the path of the earthquake (from file)
        """


    def __str__(self):
        """
        String that contains key download info
        """
        return_str  = "Earthquake Parameters\n"
        return_str += "--------------------------------------------------\n"
        return_str += "Earthquake ID: %s\n" % self.eventname
        return_str += "Origin Time: %s\n" % self.origin_time
        return_str += "Origin Latitude: %s\n" % self.origin_latitude
        return_str += "Origin Longitude: %s\n\n" % self.origin_longitude

        return_str += "Download Parameters\n"
        return_str += "--------------------------------------------------\n"
        return_str += "Duration [s]: %s\n" % self.duration
        return_str += "Channels: %s\n" % self.channels
        return_str += "Locations: %s\n" % self.origin_latitude
        return_str += "Origin Longitude: %s\n" % self.origin_longitude


        return return_str

