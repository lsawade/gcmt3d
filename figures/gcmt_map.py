from obspy import read_events
from obspy.imaging.beachball import beach

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm



def load_ndk_file(file):
    """Loads the CMT ndk file and outputs the locations of all cmt files. """

    keys = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

    # Load the ndk file
    catalog = read_events(file)

    # Go through
    latitude = []
    longitude = []
    depth = []
    mt = []
    smt = []
    origin_time = []
    for event in catalog:
        latitude.append(event.origins[0].latitude)
        longitude.append(event.origins[0].longitude)
        depth.append(event.origins[0].depth)
        origin_time.append(event.origins[0].time)
        M = []
        for key in keys:
            M.append(event.focal_mechanisms[0].moment_tensor.tensor[key])
        mt.append(M)
        smt.append(event.focal_mechanisms[0].moment_tensor.scalar_moment)

    return latitude, longitude, depth, mt, smt, origin_time



def min_UTC(UTC_list, ind=False):
    """ Get first, earliest time in a list of UTC

    :param UTC_list: list of UTCDateTime stamps
    :param ind: boolean defining whether the index in list is supposed to be
                output. Default False
    :return: earliest UTCDateTime stamp. If ind is True, tuple of (earliest time
             stamp, index) is output.

    Usage:

        .. code-block:: python

            from mermaid_plot import max_UTC

            # Assuming you have a list of UTCDateTimes
            oldest_UTC = min_UTC(<some_UTCDatetime_list>)

    """

    counter = 0
    index = 0
    min_timestamp = UTC_list[0]

    for time in UTC_list:
        if time < min_timestamp:
            min_timestamp = time
            index = counter
        counter +=1

    if ind:
        return min_timestamp, index
    else:
        return min_timestamp


def plot_mts(ax, file_name):


    latitude, longitude, depth, mt, smt, otime = load_ndk_file(file_name)

    print(len(latitude))

    # Get timing
    minUTC = min_UTC(otime)

    elaps = []

    for time in otime:
        elaps.append(time - minUTC)

    # normalized times
    norm_times = np.array(elaps)/np.max(np.array(elaps))

    norm_times = np.where(norm_times < 0.4, 0.4, norm_times)

    # Color as a function of depth
    cmap = cm.get_cmap("inferno")
    vmin = -50000
    vmax = 500000
    colors = Normalize(vmin, vmax, clip=True)(np.array(depth))
    colors = cmap(colors)

    # Moment magnitude from scalar moment
    mmt = np.array((np.log10(smt)-9.1)/1.5)

    maxmmt = np.max(mmt)

    # Normalized Moment magnitude
    nmmt = (mmt/maxmmt)

    for (lon, lat, c, d, m, sm, time) \
            in zip(longitude, latitude, colors,
                   depth, mt, nmmt, norm_times):
        try:
            b = beach(m, linewidth=0.25, facecolor=c, bgcolor='w',
                      edgecolor='k', alpha=1, xy=(lon, lat), width=40 * sm,
                      size=100, nofill=False, zorder=100*time,
                      axes=ax)

            ax.add_collection(b)
        except Exception as e:
            print(e)


def plot_map():
    
    fig = plt.figure(figsize=(12, 7.5))
    ax = plt.axes(projection=ccrs.PlateCarree(0.0))
    ax.set_global()
    ax.frameon = True
    # ax.outline_patch.set_visible(False)

    # Set gridlines. NO LABELS HERE, there is a bug in the gridlines
    # function around 180deg
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                           linewidth=1, color='lightgray', alpha=0.5,
                           linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = True

    # Change fontsize
    fontsize = 12
    font_dict = {"fontsize": fontsize,
                 "weight": "bold"}
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=font_dict)

    # Get WMS map if wanted
    ax.stock_img()
    # ax.add_feature(cfeature.LAND, zorder=10)
    # ax.add_feature(cfeature.OCEAN, zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=10)


    plot_mts(ax, "jan76_dec17.ndk")
    # plot_mts(ax, "qcmt.ndk")
    # plt.show()
    plt.savefig("map.pdf", format='pdf')



if __name__ == "__main__":

    # plot map
    plot_map()

