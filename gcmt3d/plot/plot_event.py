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


class LowerThresholdRobinson(cartopy.crs.Robinson):
    @property
    def threshold(self):
        return 1e3


class LowerThresholdPlateCarree(cartopy.crs.PlateCarree):
    @property
    def threshold(self):
        return 0.1


def plot_map():
    ax = plt.gca()
    ax.set_global()
    ax.frameon = True
    ax.outline_patch.set_linewidth(1.5)
    ax.outline_patch.set_zorder(100)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=cartopy.crs.Geodetic(),
                      draw_labels=False,
                      linewidth=1, color='lightgray', alpha=0.5,
                      linestyle='-', zorder=-1.5)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = True

    # Add Coastline
    ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
                   linewidth=0.5, facecolor=(0.9, 0.9, 0.9))


def plot_event(event: Event, inv: Inventory, filename=None,
               projection="azi_equi"):
    """Takes in an inventory and plots paths one a map.

    Args:
        event: Obspy event class, required
        inv: inventory obspy inventory, required
        filename: output filename, Default None
        transform: map projection. Default azi_equi. Other possible values:
                  "stereo", "robinson", "plate"
    Returns:
        map plot as file if filename is set and window if not
    """

    # Get Event location
    ev_coords = get_event_location(event)
    mt = get_moment_tensor(event)

    # Get Station locations
    locations = get_station_locations(inv)

    # Transformations
    geo = cartopy.crs.Geodetic()

    # Define transform
    if projection == "azi_equi":
        trans = cartopy.crs.AzimuthalEquidistant(
            central_longitude=ev_coords[1], central_latitude=ev_coords[0])
        plt.figure(figsize=(6.25, 6.25))
    elif projection == "stereo":
        trans = cartopy.crs.Stereographic(
            central_longitude=ev_coords[1], central_latitude=ev_coords[0])
        plt.figure(figsize=(6.25, 6.25))
    elif projection == "robinson":
        trans = LowerThresholdRobinson(central_longitude=ev_coords[1])
        plt.figure(figsize=(12, 6.25))
    else:
        trans = LowerThresholdPlateCarree(central_longitude=ev_coords[1])
        plt.figure(figsize=(12, 6.25))

    # Create figure
    ax = plt.axes(projection=trans)

    # Plot Map
    plot_map()

    # Plot rays (this coule be ugly, but let's see)
    for lat, lon in locations:
        plt.plot(lon, lat, "v", markerfacecolor=(0.7, 0.15, 0.15),
                 markeredgecolor='k', markersize=10, transform=geo,
                 zorder=15, clip_on=True)
        plt.plot([lon, ev_coords[1]], [lat, ev_coords[0]], c=(0.3, 0.3, 0.45),
                 lw=0.75, transform=geo, zorder=10)

    # Plot beachball
    # Transform the beach ball location to new location according to
    # the central longitude
    mtlon, mtlat = trans.transform_point(ev_coords[1], ev_coords[0], geo)
    b = beach(mt, linewidth=0.75, facecolor='k', bgcolor='w', edgecolor='k',
              alpha=1, xy=(mtlon, mtlat), width=40,
              nofill=False, zorder=16, axes=ax)
    ax.add_collection(b)

    # To set everything correctly
    ax.set_global()

    # Make it smaller
    plt.tight_layout()

    if filename is not None:
        plt.savefig("")
    else:
        plt.show()
