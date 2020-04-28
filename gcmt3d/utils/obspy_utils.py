"""

Help with the conversion between obspy and normal

:copyright:
    Lucas Sawade (lsawade@princeton.edu) 2019

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""

# External imports
import logging
from obspy import Inventory
from obspy.core.event.event import Event

# Internal imports
from ..log_util import modify_logger

logger = logging.getLogger(__name__)
modify_logger(logger)


def get_event_location(event: Event):
    """Takes in an obspy Event and spits out event location.

    Args:
        event:

    Returns:
        lat, lon
    """

    return event.preferred_origin()


def write_stations_file(inv: Inventory, filename="STATIONS"):
    """
    Write station inventory out to a txt file(in SPECFEM FORMAT)

    :param sta_dict:
    :type sta_dict: obspy.Inventory
    :param filename: the output filename for STATIONS file.
    :type filename: str
    """

    logger.verbose("Writing STATIONS file to: %s" % filename)

    with open(filename, 'w') as fh:

        for network in inv:
            for station in network:
                # Get station parameters
                lat = station.latitude
                lon = station.longitude
                elev = station.elevation
                burial = 0.0  # Fixed parameter, now readable by obspys

                # Write line
                fh.write("%-9s %5s %15.4f %12.4f %10.1f %6.1f\n"
                         % (station.code, network.code,
                            lat, lon, elev, burial))
