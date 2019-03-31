"""

    This Function contains functions to build an FDSN time series download
    request from an input CMT source object and List of stations.

"""

from gcmt3d.source import CMTSource
import os
from subprocess import Popen, PIPE, STDOUT
import warnings
import sys


# Input Error
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


class DataRequest(object):
    """

    Class that handles the building of a data request

    """

    def __init__(self, cmt=None, stationlist=None, channels=['BHZ'],
                 locations=['00'], duration=0.0, starttime_offset=0,
                 outputdir="", verbose=1):
        """
        :param cmt: CMTSource object from cmt Source
        :param stationlist: list of station in format
                            `[['Net1','Sta1'],['Net2','Sta2'],etc.]`
        :param channels: list of strings containing wanted channels
        :param locations: list of seismometer locations
        :param duration: duration of the requested recording from origin time
        :param starttime_offset: time to request before the origin time, e.g.,
                                 -10 would create a request starting 10 seconds
                                 before the origin time
        :param resp_format: Response format. Either "resp" (SEED) or "pz"
                            (PolesAndZeros). Default "resp"
        :param outputdir: directory to save time series in, in a subdirectory
                          called `seismograms/`
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
        self.duration = duration
        self.starttime_offset = starttime_offset
        self.starttime = self.origin_time + self.starttime_offset
        self.endtime = self.starttime + self.duration
        self.channels = channels
        self.locations = locations
        self.outputdir = outputdir

        # Verbose output
        self.verbose = verbose

        # Name of the output directory:
        # Can't be also parsed input since it's dependent on the provided
        # filename
        if self.outputdir == "":
            self.outputdir = "."
            warnings.warn("No output directory chosen. Seismograms will be " +
                          "saved\nin current location in subdirectory " +
                          "'seismograms/'.")
        elif not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    @classmethod
    def from_file(cls, cmtfname,
                  stationlistfname,
                  channels=["BHZ"],
                  locations=["00"],
                  starttime_offset=0,
                  outputdir="",
                  verbose=1):
        """Creates a downloader class from an input file

        This downloader class also needs to contain the output directory for the
        traces. Take CMT fname
        """

        # Name of the output directory:
        # Can't be also parsed input since it's dependent on the provided
        # filename
        if outputdir == "":
            outputdir = os.path.abspath(os.path.join(cmtfname, os.pardir))

        # First Create CMT Solution
        cmt = CMTSource.from_CMTSOLUTION_file(cmtfname)

        # Read station list file with two columns and whitespace separation
        statfile = open(stationlistfname, 'r')

        # For loop to add all stations to the station list
        stationlist = []
        for line in statfile:
            # Read stations into list of stations
            line = line.split()

            if line[0] == "NETWORK":
                continue
            else:
                # Append the [network station latitude longitude elevation] to the station list
                stationlist.append(line)

        return cls(cmt=cmt,
                   stationlist=stationlist,
                   channels=channels,
                   locations=locations,
                   duration=duration,
                   starttime_offset=starttime_offset,
                   outputdir=outputdir,
                   verbose=verbose)

    def request(self):
        """
        Writes a request file to the earthquake directory. This request file is
        then used by the download function to download.

        :returns str: Path to requestfile
        """

        # Open file for writing in the earthquake directory
        path_to_file = self.outputdir+'/request.txt'
        requestfile = open(path_to_file, 'w')

        # Writing for each parameter overarching of all is the station parameter
        # of course

        for station in self.stationlist:
            for location in self.locations:
                for channel in self.channels:
                    #                                     joining just the
                    #                                     network   & station
                    requestfile.write(" ".join([" ".join([station[0], station[1]]),
                                                location,
                                                channel,
                                                self.starttime.__str__(),
                                                self.endtime.__str__()]))
                    requestfile.write("\n")

        return path_to_file

    def download(self, selection_file="", download_log_file=""):
        """
        takes self.request and starts the download of the time series data
        """

        # Create request file if none is given
        if selection_file == "":
            selection_file = "\\ ".join(self.request().split())

        # Create download log file in Earthquake directory if no other directory
        # is specified
        if download_log_file == "":
            download_log_file = self.outputdir+"/download_log.txt"

        # If doesn't exist, create directory for responses and seismograms
        seis_path = os.path.join(self.outputdir, "seismograms/obs")

        if not os.path.exists(seis_path):
            os.makedirs(seis_path)
        if not os.path.exists(resp_path):
            os.makedirs(resp_path)

        # Relative path to the resources directory
        path_to_this_file = os.path.abspath(os.path.dirname(__file__))
        path_to_script = "\\ ".join(os.path.join(path_to_this_file,
                                                 "resources").split())

        # Invoke download command depending on the response format
        with open(download_log_file, "w") as out:

            Proc = Popen(" ".join(["%s/FetchData" % path_to_script,
                            "-l", "%s" % selection_file,
                            "-o", "%s.mseed" %
                                 os.path.join("\\ ".join(seis_path.split()),
                                              self.eventname),
                            "-X", "%s" % "\\ ".join(self.outputdir.split()) +
                            "/station_data/"+"station.xml"]),
                  shell=True, stdout=PIPE, stderr=STDOUT)

            for line in Proc.stdout:
                # write to standard out
                if self.verbose:
                    sys.stdout.write(line)

                # write to logfile
                out.write(line.decode('utf-8'))
            Proc.wait()

    def specfem_list(self, specfemfiledir=""):
        """This method takes the station list and its content to create a
        Specfem Station recording location script list.

        The text file should have following format but not contain a header

        ::
            -------------------------------------------------------------
            | Station Network Latitude Longitude Elevation(m) Burial(m) |
            -------------------------------------------------------------

        That is the a sample line would be:

        ::

            HRV   IU   42.5064  -71.5583  200.0  0.0
            :     :       :         :       :     :

        :keyword specfemfiledir: directory where to save the STATIONS file for
                                 specfem3d_globe

        """

        # Setting the Station file name and path
        if specfemfiledir == "":
            specfemfile = self.outputdir+"/station_data/STATIONS"
        else:
            specfemfile = specfemfiledir + "/station_data/STATIONS"


        with open(specfemfile,'w') as file:
            for k,line in enumerate(self.stationlist):
                file.write('{0:11s}{1:4s}{2:12.4f}{3:12.4f}{4:10.1f}{5:8.1f}\n'
                           .format(line[1], line[0], float(line[2]),
                                   float(line[3]), float(line[4]), 0.))

        if self.verbose:
            print("\nSTATIONS FILE WRITTEN.\n")



    def __str__(self):
        """
        String that contains key download info.
        """
        return_str  = "\nEarthquake Parameters\n"
        return_str += "--------------------------------------------------\n"
        return_str += "Earthquake ID: %s\n" % self.eventname
        return_str += "Origin Time: %s\n" % self.origin_time
        return_str += "Origin Latitude: %s\n" % self.origin_latitude
        return_str += "Origin Longitude: %s\n\n" % self.origin_longitude

        return_str += "Download Parameters:\n"
        return_str += "--------------------------------------------------\n"
        return_str += "Starttime: %s\n" % self.starttime
        return_str += "Endtime: %s\n" % self.endtime
        return_str += "Duration [s]: %s\n" % self.duration
        return_str += "Channels: %s\n" % self.channels
        return_str += "Locations: %s\n\n" % self.locations
        return_str += "Response Format: %s\n\n" % self.resp_format

        return_str += "Saving Parameters:\n"
        return_str += "--------------------------------------------------\n"
        return_str += "Output Directory: %s\n" % self.outputdir


        return return_str

