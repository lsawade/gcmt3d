"""

Plot event of to asdf files and their windows

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2020


"""

# External imports
import logging
import matplotlib.pyplot as plt
from obspy import Inventory
from obspy.core.event.event import Event
from obspy.imaging.beachball import beach
import cartopy
from cartopy.crs import Robinson

# Internal imports
from ..utils.obspy_utils import get_event_location
from ..utils.obspy_utils import get_moment_tensor
from ..utils.obspy_utils import get_station_locations
from ..plot.plot_util import set_mpl_params_section
from ..log_util import modify_logger

# Get logger
logger = logging.getLogger(__name__)
modify_logger(logger)

# Set Rcparams for the section plot
set_mpl_params_section()


def plot_map(central_longitude):
    ax = plt.gca()
    ax.set_global()
    ax.frameon = True
    ax.outline_patch.set_linewidth(0.75)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=Robinson(central_longitude=central_longitude),
                      draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-', zorder=-1.5)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = True

    # Add Coastline
    ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
                   linewidth=0.5, facecolor=(0.9, 0.9, 0.9))


def plot_event(event: Event, inv: Inventory):
    """Takes in an inventory and plots paths one a map."""

    # Get Event location
    ev_coords = get_event_location(event)
    mt = get_moment_tensor(ev_coords)

    # Get Station locations
    locations = get_station_locations(inv)
    print(locations)

    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=Robinson(central_longitude=ev_coords[1]))

    # Plot Map
    plot_map(ev_coords[1])

    # Plot beachball
    b = beach(mt, linewidth=0.25, facecolor='k', bgcolor='w', edgecolor='k',
              alpha=1, xy=(ev_coords[1], ev_coords[0]), width=10, size=10,
              nofill=False, zorder=10, axes=ax)
    ax.add_collection(b)
