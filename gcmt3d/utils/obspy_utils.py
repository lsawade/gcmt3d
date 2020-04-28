"""

Help with the conversion between obspy and normal

:copyright:
    Lucas Sawade (lsawade@princeton.edu) 2019

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""

from obspy.core.event.event import Event


def get_event_location(event: Event):
    """Takes in an obspy Event and spits out event location.

    Args:
        event:

    Returns:
        lat, lon
    """

    return event.preferred_origin()


def create_cumulative_ivnentory(xml_file_list):
    """Create a stations file from the xml """


def write_stations_file(inv, filename="STATIONS"):
    """
    Write station information out to a txt file(in SPECFEM FORMAT)

    :param sta_dict: the dict contains station locations information.
        The key should be "network.station", like "II.AAK".
        The value are the list of
        [latitude, longitude, elevation_in_m, depth_in_m].
    :type sta_dict: dict
    :param filename: the output filename for STATIONS file.
    :type filename: str
    """
    # with open(filename, 'w') as fh:
    #     od = collections.OrderedDict(sorted(sta_dict.items()))
    #     for _sta_id, _sta in od.items():
    #         network, station = _sta_id.split(".")
    #         _lat = _sta[0]
    #         _lon = _sta[1]
    #         check_in_range(_lat, [-90.1, 90.1])
    #         check_in_range(_lon, [-180.1, 180.1])
    #         fh.write("%-9s %5s %15.4f %12.4f %10.1f %6.1f\n"
    #                  % (station, network, _lat, _lon, _sta[2], _sta[3]))
    pass
