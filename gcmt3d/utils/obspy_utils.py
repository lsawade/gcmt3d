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
    origin = event.preferred_origin()
    return origin.latitude, origin.longitude


def get_moment_tensor(event: Event):
    """Takes in an obspy Event and spits a focal mechanisms
    This is the fucking worst. I mean look at it. Just to
    get the moment tensor.

    Args:
        event:

    Returns:
        lat, lon
    """
    tensor = event.preferred_focal_mechanism().moment_tensor.tensor
    return [tensor.m_rr, tensor.m_tt, tensor.m_pp,
            tensor.m_rt, tensor.m_rp, tensor.m_tp]


def get_station_locations(inv: Inventory):
    """Get all station locations.

    Args:
        inv: :class:`obspy.Inventory`

    Returns:
        list of station_locations [ [lat1, lon1], ..., [latn, lonn]]
    """
    locations = []
    for network in inv:
        for station in network:
            # Get station parameters
            lat = station.latitude
            lon = station.longitude

            locations.append([lat, lon])

    return locations


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
