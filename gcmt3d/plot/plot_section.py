"""

Plot section of to asdf files and their windows

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2020


"""

# External imports
import os
import json
import logging
from numpy import max, array, argsort, arange
from pyasdf import ASDFDataSet
from obspy import Stream, Inventory
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from matplotlib.pyplot import plot, xlabel, ylabel, xlim, ylim, title
from matplotlib.pyplot import figure, axes, show, savefig, tight_layout

from matplotlib.patches import Rectangle

# Internal imports
from .plot_util import set_mpl_params_section
from ..log_util import modify_logger

# Get logger
logger = logging.getLogger(__name__)
modify_logger(logger)

# Set Rcparams for the section plot
set_mpl_params_section()


def plot_windows_on_trace(ax, winlist, epi, smax, timefactor):
    for win in winlist:
        # Get start and endtime
        starttime = win["relative_starttime"] / timefactor
        endtime = win["relative_endtime"] / timefactor

        # Define coordinates
        halfmax = smax / 2
        lowerleft = (epi - halfmax, starttime,)
        width = smax
        height = endtime - starttime

        # Plot
        r = Rectangle(lowerleft, width, height, fill=True, fc="b",
                      alpha=0.15)
        ax.add_patch(r)


def plot_section(obsd_file_name, synt_file_name=None, window_file_name=None,
                 timescale="min", scale=1.5, outputdir=None):
    """ This function takes in and observed and a synthetic asdf file and
    a FLEXWIN sstyle json file to plot a seismic section, that

    Args:
        obsd_file_name: Name of Obsd asdf file
        synt_file_name: Name of Synt asdf file
        window_file_name: name of window file. Default None.
        outputdir: Output directory to save sections

    Returns:
        plots or saves figure depending
    """

    # -------------------------------------------------------------------------
    # Loading the data
    obsdds = ASDFDataSet(obsd_file_name)

    if synt_file_name is not None:
        syntds = ASDFDataSet(synt_file_name)

    if window_file_name is not None:
        with open(window_file_name, 'r') as winfile:
            windows = json.load(winfile)

    # Create empty obsjects
    inv = Inventory()
    obsdst = Stream()

    if synt_file_name is not None:
        syntst = Stream()

    # Fill data objects
    obsdtag = list(obsdds.waveform_tags)[0]

    if synt_file_name is not None:
        synttag = list(syntds.waveform_tags)[0]

    for station in obsdds.waveforms.list():
        try:
            obsdst += getattr(obsdds.waveforms[station], obsdtag)
            inv += obsdds.waveforms[station].StationXML

            if synt_file_name is not None:
                syntst += getattr(syntds.waveforms[station], synttag)

        except Exception as e:
            logger.verbose(e)

    # Loads the only event contained in the catalog
    origin = obsdds.events[0].preferred_origin()

    # -------------------------------------------------------------------------
    # Just printing stuff for debugging in the future
    logger.debug("Event:")
    logger.debug(" ")
    for line in origin.__str__().splitlines():
        logger.debug("    %s" % line)
    logger.debug(" ")
    logger.debug(" ")
    logger.debug("Observed Stream:")
    logger.debug(" ")
    for line in obsdst.__str__().splitlines():
        logger.debug("    %s" % line)
    logger.debug(" ")
    logger.debug(" ")
    logger.debug("Synthetic Stream:")
    logger.debug(" ")
    for line in obsdst.__str__().splitlines():
        logger.debug("    %s" % line)
    logger.debug(" ")

    # -------------------------------------------------------------------------
    # Get spatial stuff such as epicentral ditance and azimuth

    # Define event coordinates:
    ev_coord = (origin.latitude, origin.longitude)

    codes = []
    azimuth = []
    epic = []
    wins = []
    for network in inv:
        for station in network:

            # Create empty station code list and window list
            stacode = dict()
            if window_file_name is not None:
                stawins = dict()

            # get location parameters
            azimuth.append(gps2dist_azimuth(ev_coord[0], ev_coord[1],
                                            station.latitude,
                                            station.longitude)[1])
            epic.append(locations2degrees(ev_coord[0], ev_coord[1],
                                          station.latitude,
                                          station.longitude))

            if window_file_name is not None:
                station_window_dict = windows[
                    "%s.%s" % (network.code, station.code)]

            for channel in station:
                codemap = {"N": "R", "E": "T", "Z": "Z", }

                if window_file_name is not None:
                    try:
                        stawins[codemap[channel.code[-1]]] \
                            = station_window_dict["%s.%s.%s.%s" %
                                                  (network.code, station.code,
                                                   channel.location_code,
                                                   channel.code[:-1]
                                                   + codemap[channel.code[-1]])
                                                  ]
                    except Exception as e:
                        stawins[codemap[channel.code[-1]]] = []
                        template = "Channel %s.%s.%s.%s - " \
                                   "Error type %s - Arguments: %s"
                        message = template % (network.code, station.code,
                                              channel.location_code,
                                              channel.code,
                                              type(e).__name__, e.args)
                        logger.warning(message)

                stacode[codemap[channel.code[-1]]] = \
                    (network.code, station.code,
                     channel.location_code, channel.code)
            codes.append(stacode)

            if window_file_name is not None:
                wins.append(stawins)

    # --------------------------------------------------------------------------
    # Convert to numpy arrays
    codes = array(codes)
    epic = array(epic)
    azimuth = array(azimuth)

    # Sort stuff according to epicentral distance
    pos = argsort(epic)
    epic = epic[pos]
    azimuth = azimuth[pos]
    codes = codes[pos]

    if window_file_name is not None:
        wins = array(wins)
        wins = wins[pos]

    # --------------------------------------------------------------------------
    # Now that everything is sorted into the correct arrays create a data
    # structure for quick plotting

    npts = obsdst[0].stats.npts
    dt = obsdst[0].stats.delta
    t = arange(0, npts, 1) * dt

    # Initialize dictionaries to simplify plotting later on
    o = {"R": {"data": [], "dist": [], "wins": []},
         "T": {"data": [], "dist": [], "wins": []},
         "Z": {"data": [], "dist": [], "wins": []}}

    if synt_file_name is not None:
        s = {"R": {"data": [], "dist": [], "wins": []},
             "T": {"data": [], "dist": [], "wins": []},
             "Z": {"data": [], "dist": [], "wins": []}}

    for _i, station in enumerate(codes):
        for key, channel in station.items():
            try:
                # Add data
                o[key]["data"].append(
                    obsdst.select(network=channel[0], station=channel[1],
                                  component=key)[0].data)

                if synt_file_name is not None:
                    s[key]["data"].append(
                        syntst.select(network=channel[0], station=channel[1],
                                      component=key)[0].data)
                # Add epicentral distance
                o[key]["dist"].append(epic[_i])

                if synt_file_name is not None:
                    s[key]["dist"].append(epic[_i])

                # Add windows
                if window_file_name is not None:
                    o[key]["wins"].append(wins[_i][key])
                    if synt_file_name is not None:
                        s[key]["wins"].append(wins[_i][key])

            except Exception as e:
                template = "Channel %s.%s.%s.%s - " \
                           "Error type %s - Arguments: %s"
                message = template % (channel[0], channel[1],
                                      channel[2], channel[3],
                                      type(e).__name__, e.args)
                logger.warning(message)

    logger.info("Number of traces for R: %d" % len(o["R"]["data"]))
    logger.info("Number of traces for T: %d" % len(o["T"]["data"]))
    logger.info("Number of traces for Z: %d" % len(o["Z"]["data"]))

    # --------------------------------------------------------------------------
    # Since all data is sorted nicely, let's plot stuff.
    # Loop over components to plot each
    titlemap = {"R": "Radial", "T": "Transverse", "Z": "Vertical"}
    timemap = {"sec": 1.0, "min": 60.0, "hours": 3600.0}

    # Loop over components
    for comp, odict in o.items():
        figure(figsize=(8, 12))
        ax = axes()
        for _i, (obs, epi) in enumerate(zip(odict["data"], odict["dist"])):
            if synt_file_name is not None:
                syn = s[comp]["data"][_i]
                smax = max(abs(syn))

                # Plot synthetic data if available
                plot(scale * syn / smax + epi,
                     t / timemap[timescale],
                     "r", lw=0.5)

                # Get common max for both
                commax = max([max(abs(scale * syn / smax)),
                              max(abs(scale * obs / smax))])
            else:
                smax = max(abs(obs))
                commax = smax

            # Plot observed data
            plot(scale * obs / smax + epi,
                 t / timemap[timescale], "k", lw=0.5)

            # Plot windows
            if window_file_name is not None:
                _wins = odict["wins"][_i]
                plot_windows_on_trace(ax, _wins, epi, commax,
                                      timemap[timescale])

        # Put labels
        title("%s Component" % titlemap[comp], fontweight="bold")
        ylabel("Time [%s]" % timescale)
        xlabel(r"$\Delta$ [deg]")

        # Set limits
        ylim((0, max(t / timemap[timescale])))
        xlim((0, 180))
        tight_layout()

        if outputdir is not None:
            if window_file_name is not None:
                winsuffix = ".wins"
            else:
                winsuffix = ""
            if synt_file_name is not None:
                synsuffix = ".comp"
            else:
                synsuffix = ""

            filename = "%s_component%s%s.pdf" % (comp, synsuffix, winsuffix)
            outfile = os.path.join(outputdir, filename)
            logger.info("Saving plot as: %s" % outfile)

            savefig(outfile)

    show()


if __name__ == "__main__":
    pass
