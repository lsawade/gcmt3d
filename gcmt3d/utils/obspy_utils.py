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
