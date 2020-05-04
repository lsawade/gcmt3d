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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from obspy import Inventory
from obspy.core.event.event import Event
from obspy.imaging.beachball import beach
from obspy.geodetics.base import gps2dist_azimuth
import cartopy

# Internal imports
from ..source import CMTSource
from ..utils.io import load_json
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


def plot_bounds():
    ax = plt.gca()
    lw = 1.0
    ax.plot([0, 1], [0.0, 0.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0, 1], [1.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([0, 0], [0.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)
    ax.plot([1, 1], [0.0, 1.0], color='black', lw=lw,
            transform=ax.transAxes, clip_on=False)


def unique_locations(latitude, longitude):
    """Returns to lists of corresponding latitude and longitude. But only
    unique entries"""

    # Magic line
    stations = list(set([(lat, lon) for lat, lon in zip(latitude, longitude)]))

    # Lat list, Lon lisst
    return [sta[0] for sta in stations], [sta[1] for sta in stations]


def extract_locations_from_comp(complist: list):
    """Collects latitudes and longitudes in form on number of
    windows. if a trace has 3 windows the location will be added 3 times.
    Later when plotting the scattered stations, use a set comprehension.

    Argument:
        complist: is a list of all traces on one component
    """
    lat = []
    lon = []
    for trace in complist:
        for _i in range(trace["nwindows"]):
            lat.append(trace["lat"])
            lon.append(trace["lon"])

    return lat, lon


def extract_stations_from_traces(wave_dict):
    """Get all stations"""
    lat = []
    lon = []

    for wave in wave_dict.keys():
        for comp, complist in wave_dict[wave]["traces"].items():
            for trace in complist:
                lat.append(trace["lat"])
                lon.append(trace["lon"])

    return lat, lon


def get_azimuth(elat, elon, latitude, longitude):
    """ computes the azimuth for multiple stations

    Args:
        elat: event latitude
        elon: event longitude
        latitude: station latitudes
        lon: station longitudes
    Returns:

    """
    azi = []
    for lat, lon in zip(latitude, longitude):
        azi.append(gps2dist_azimuth(elat, elon, lat, lon)[1])
    return azi


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


class PlotEventSummary():
    """Gets the either the pycmt3d json or both the pycmt3d and the gridsearch
    summary json. """

    def __init__(self, cmt3d: dict, g3d: dict = None):
        """The function takes in a summary dictionaries and plots info
        accordingly

        Arguments:
            cmt3d: dictionary for CMT3D
            g3d: dictionary for the gridsearch

        Returns:
            plots
        """

        # Main dicts:
        self.cmt3d = cmt3d
        self.g3d = g3d

        # Old location
        self.olat = self.cmt3d["oldcmt"]["latitude"]
        self.olon = self.cmt3d["oldcmt"]["longitude"]

        # New location
        self.clat = self.cmt3d["oldcmt"]["latitude"]
        self.clon = self.cmt3d["oldcmt"]["longitude"]

        # Gridsearch location
        if self.g3d is not None:
            # Get g3d info
            pass

        # Projections
        self.geo = cartopy.crs.Geodetic()
        self.azi_equi = cartopy.crs.AzimuthalEquidistant(
            central_longitude=self.olon, central_latitude=self.olat
        )
        self.robinson = cartopy.crs.Robinson(central_longitude=self.olon)

    @classmethod
    def from_JSON(self, cmt3d_json: str, g3d_json: str = None):
        """The function takes in a summary json files and creates the class

        Arguments:
            cmt3d: dictionary for CMT3D
            g3g: dictionary for the gridsearch

        Returns:
            plots
        """

        # Load cmt3d JSON
        cmt3d = load_json(cmt3d_json)

        # Load g3d json if wanted
        if g3d_json is not None:
            g3d = load_json(g3d_json)
        else:
            g3d = None

        return self(cmt3d, g3d=g3d)

    def plot_table(self):

        ax = plt.gca()
        """Plots the table that shows the CMT3D and the G3D stuff"""
        ocmt = CMTSource.from_dictionary(self.cmt3d["oldcmt"])
        newcmt = CMTSource.from_dictionary(self.cmt3d["newcmt"])

        sta_lat, sta_lon = extract_stations_from_traces(
            self.cmt3d["wave_dict"])
        nwins = len(sta_lat)
        sta_lat, sta_lon = unique_locations(sta_lat, sta_lon)
        nsta = len(sta_lat)

        par_mean = self.cmt3d["bootstrap_mean"]
        par_std = self.cmt3d["bootstrap_std"]
        std_over_mean = np.zeros(len(par_mean))
        for _i in range(len(par_mean)):
            if par_mean[_i] != 0:
                std_over_mean[_i] = par_std[_i] / np.abs(par_mean[_i])
            else:
                std_over_mean[_i] = 0.0
        # Moment Tensors
        format1 = "%15.4e  %15.4e  %15.4e  %15.4e  %10.2f%%\n"
        # CMT/HDR
        # format2 = "%10.3f s  %13.3f s  %13.3f s  %13.3f s  %13.2f%%\n"
        # Depth
        format3 = "%10.3f km  %12.3f km  %12.3f km  %12.3f km  %12.2f%%\n"
        # LatLon
        format4 = "%10.3f deg  %11.3f deg  %11.3f deg  %11.3f deg  %11.2f%%\n"

        text = "Number of stations: %5d    Number of windows: %4d" \
               % (nsta, nwins) + "    Envelope coef: %12.3f\n" \
               % self.cmt3d["config"]["envelope_coef"]

        text += "Number of Parameter: %4d    Zero-Trace: %11s" \
                "    Double-couple: %12s \n" \
                % (self.cmt3d["config"]["npar"],
                   self.cmt3d["config"]["zero_trace"],
                   self.cmt3d["config"]["double_couple"])

        text += "Station Correction: %5s    Norm_by_energy: %7s" \
                "    Norm_by_category: %9s\n" \
                % (self.cmt3d["config"]["station_correction"],
                   self.cmt3d["config"]["weight_config"]
                   ["normalize_by_energy"],
                   self.cmt3d["config"]
                   ["weight_config"]["normalize_by_category"])

        energy_change = (newcmt.M0 - ocmt.M0) / ocmt.M0
        text += "Inversion Damping: %6.3f    Energy Change: %7.2f%%" \
                "    Variance Reduction: %6.2f%%\n" \
                % (self.cmt3d["config"]["damping"], energy_change * 100,
                   self.cmt3d["var_reduction"] * 100)

        text += "-" * 32 + "   Summary Table   " + "-" * 32 + "\n"
        text += "PAR      Global CMT          CMT3D          " \
                "Bootstrap_Mean   Bootstrap_STD  STD/Mean\n"

        text += "Mrr:" + format1 % (ocmt.m_rr, newcmt.m_rr,
                                    par_mean[0], par_std[0],
                                    std_over_mean[0] * 100)

        text += "Mtt:" + format1 % (ocmt.m_tt, newcmt.m_tt, par_mean[1],
                                    par_std[1], std_over_mean[1] * 100)

        text += "Mpp:" + format1 % (ocmt.m_pp, newcmt.m_pp, par_mean[2],
                                    par_std[2], std_over_mean[2] * 100)

        text += "Mrt:" + format1 % (ocmt.m_rt, newcmt.m_rt, par_mean[3],
                                    par_std[3], std_over_mean[3] * 100)
        text += "Mrp:" + format1 % (ocmt.m_rp, newcmt.m_rp, par_mean[4],
                                    par_std[4], std_over_mean[4] * 100)
        text += "Mtp:" + format1 % (ocmt.m_tp, newcmt.m_tp, par_mean[5],
                                    par_std[5], std_over_mean[5] * 100)
        text += "DEP:" + format3 % (ocmt.depth_in_m / 1000,
                                    newcmt.depth_in_m / 1000,
                                    par_mean[6] / 1000, par_std[6] / 1000,
                                    std_over_mean[6] * 100)
        text += "LAT:" + format4 % (ocmt.latitude, newcmt.latitude,
                                    par_mean[8], par_std[8],
                                    std_over_mean[8] * 100)

        text += "LON:" + format4 % (ocmt.longitude, newcmt.longitude,
                                    par_mean[7], par_std[7],
                                    std_over_mean[7] * 100)

        fontsize = 9
        plt.text(0.01, 0.95, text, fontsize=fontsize, fontweight="normal",
                 fontfamily="monospace", transform=ax.transAxes,
                 horizontalalignment="left", verticalalignment="top",
                 zorder=100
                 )
        plot_bounds()
        plt.axis('off')

    def plot_g3d_table(self):
        """Plots the table that shows the CMT3D and the G3D stuff"""
        ax = plt.gca()
        ocmt = CMTSource.from_dictionary(self.cmt3d["oldcmt"])
        newcmt = CMTSource.from_dictionary(self.cmt3d["newcmt"])
        if self.g3d is not None:
            g3dcmt = CMTSource.from_dictionary(self.g3d["newcmt"])

        gbootstrapmean = self.g3d["G"]["bootstrap_mean"]
        gbootstrapstd = self.g3d["G"]["bootstrap_std"]

        # Moment Tensors
        format1 = "%15.4e  %15.4e  %15.4e  %15.4e  %10.2f%%\n"
        # CMT/HDR
        format2 = r"%10.3f s  %13.3f s  " \
                  "%13.3f s  %13.3f s  %13.2f%%\n"

        text = "\nGrid Search Parameters:\n"

        text += "CMT:" + format2 % (
            ocmt.time_shift, g3dcmt.time_shift,
            ocmt.time_shift + gbootstrapmean[-1],
            gbootstrapstd[-1],
            (ocmt.time_shift + gbootstrapmean[-1]) / gbootstrapstd[-1])

        text += "M0: " + format1 % (
            ocmt.M0, newcmt.M0,
            gbootstrapmean[0] * ocmt.M0, gbootstrapstd[0] * ocmt.M0,
            gbootstrapstd[0] / gbootstrapmean[0] * 100)

        fontsize = 9
        plt.text(0.01, 0.05, text, fontsize=fontsize, fontweight="normal",
                 fontfamily="monospace", transform=ax.transAxes,
                 horizontalalignment="left", verticalalignment="bottom",
                 zorder=100)
        plot_bounds()
        plt.axis('off')

    def plot_title(self):
        """This puts a text box in the top left"""
        ax = plt.gca()
        id = self.cmt3d["oldcmt"]["eventname"]
        region_tag = self.cmt3d["oldcmt"]["region_tag"]
        # Put text about the wave and the component
        fontsize = 12
        ax.text(0.01, 0.95, id + " - " + region_tag,
                fontsize=fontsize, transform=ax.transAxes,
                horizontalalignment="left", verticalalignment="top",
                zorder=100)
        plot_bounds()
        ax.axis('off')

    @staticmethod
    def get_cmt_text_from_cmt(cmt: CMTSource, cmttype):
        """Output text"""
        format = r"%16s: Lat=%6.2f, Lon=%7.2f, " \
                 "d=%4.1fkm, dt=%4.1fs, MW=%3.1f\n"
        txt = format % (cmttype.ljust(16, " "), cmt.latitude, cmt.longitude,
                        cmt.depth_in_m/1000.0, cmt.time_shift,
                        cmt.moment_magnitude)
        return txt

    @staticmethod
    def get_PDE_text_from_cmt(cmt: CMTSource):
        """Output text"""
        format = "%16s: Lat=%6.2f, Lon=%7.2f, " \
                 "d=%4.1fkm, dt=%4.1fs, mb=%3.1f, MS=%3.1f\n"
        txt = format % ("Hypocenter (PDE)", cmt.pde_latitude,
                        cmt.pde_longitude, cmt.pde_depth_in_m/1000.0,
                        0.0, cmt.mb, cmt.ms)
        return txt

    def plot_description(self):

        ax = plt.gca()
        ocmt = CMTSource.from_dictionary(self.cmt3d["oldcmt"])
        newcmt = CMTSource.from_dictionary(self.cmt3d["newcmt"])

        # Put together the Text
        txt = "Origin Time: %s\n" \
              % ocmt.origin_time.strftime("%y/%m/%d, %H:%M:%S")
        txt += self.get_PDE_text_from_cmt(ocmt)
        txt += self.get_cmt_text_from_cmt(ocmt, "Global-CMT")
        txt += self.get_cmt_text_from_cmt(newcmt, "GCMT3D")
        if self.g3d is not None:
            g3dcmt = CMTSource.from_dictionary(self.g3d["newcmt"])
            txt += self.get_cmt_text_from_cmt(g3dcmt, "GCMT3D (fix)")
        # Put text about the wave and the component
        fontsize = 10
        ax.text(0.05, 0.75, txt, fontsize=fontsize, fontweight="normal",
                fontfamily="monospace", transform=ax.transAxes,
                horizontalalignment="left", verticalalignment="top",
                zorder=100)
        ax.axis('off')

    def plot_cost(self):
        """Plots graph of the misfit reduction."""

        ax = plt.gca()

        mincost_array = np.array(self.g3d["G"]["mincost_array"])
        maxcost_array = np.array(self.g3d["G"]["maxcost_array"])
        meancost_array = np.array(self.g3d["G"]["meancost_array"])
        stdcost_array = np.array(self.g3d["G"]["stdcost_array"])

        x = np.arange(0, mincost_array.shape[0], 1)
        ax.fill_between(x, mincost_array, maxcost_array,
                        color='lightgray', label="min/max")
        ax.fill_between(x, meancost_array - stdcost_array,
                        meancost_array + stdcost_array,
                        color='darkgray', label=r"$\bar{\chi}\pm\sigma$")
        ax.plot(meancost_array, 'k', label=r"$\bar{\chi}$")

        method = self.g3d["config"]["method"]

        if method == "gn":
            label = "Gauss-Newton"
        else:
            label = "Newton"

        ax.plot(self.g3d["G"]["chi_list"], "r",
                label=r"%s ($C_{min} = %.3f$)"
                      % (label, self.g3d["G"]["chi_list"][-1]))
        plt.legend(prop={'size': 6}, fancybox=False, framealpha=1)
        ax.set_ylim([0, 1])

    def plot_summary(self, outputfilename=None):
        """The """

        plt.figure(figsize=(10, 9), facecolor='w', edgecolor='k',
                   tight_layout=True)

        nwave = len(self.cmt3d["wave_dict"])

        g = GridSpec(4 + nwave, 4)

        # Title
        plt.subplot(g[0, :3])
        self.plot_title()
        self.plot_description()

        if self.g3d is not None:
            plt.subplot(g[3, -1])
            self.plot_cost()

        plt.subplot(g[1:3, :3])
        self.plot_table()
        if self.g3d is not None:
            self.plot_g3d_table()

        #
        # plt.subplot(g[1:2, :2], projection=self.robinson)
        # plot_map()

        # Plot wave distributions
        for _i, wave in enumerate(self.cmt3d["wave_dict"].keys()):
            for _j, comp in enumerate(
                    self.cmt3d["wave_dict"][wave]["traces"].keys()):
                plt.subplot(g[4 + _i, 2 * _j], projection=self.azi_equi)
                self.plot_stationdist(wave, comp=comp)
                plot_bounds()

                plt.subplot(g[4 + _i, 2 * _j - 1], polar=True)
                self.plot_window_distribution(wave, comp=comp)
                plot_bounds()

        # Make it smaller
        plt.tight_layout()

        if outputfilename is not None:
            plt.savefig(outputfilename)
        else:
            plt.show()

    def plot_stationdist(self, wave, comp):
        """ Plots the distribution of stations for a wave and a component
        Args:
            wave:
            comp:

        Returns:

        """
        plot_map()
        lat, lon = unique_locations(*extract_locations_from_comp(
            self.cmt3d["wave_dict"][wave]["traces"][comp]))
        self.plot_stations(lat, lon)
        self.plot_paths(self.olat, self.olon, lat, lon)

        ax = plt.gca()
        # Put text about the wave and the component
        fontsize = 9
        text = "%s-%s: %d" % (wave.capitalize(), comp, len(lat))

        ax.text(-0.1, 1.1, text, fontsize=fontsize, transform=ax.transAxes,
                horizontalalignment="left", verticalalignment="center",
                zorder=100,
                bbox=dict(facecolor='white', edgecolor='black',
                          pad=3))

    def plot_window_distribution(self, wave, comp):
        """
         Plots the distribution of stations for a wave and a component
        Args:
            wave:
            comp:

        Returns:

        """

        # Get locations and azimuths
        lat, lon = extract_locations_from_comp(
            self.cmt3d["wave_dict"][wave]["traces"][comp])
        sta_azi = get_azimuth(self.olat, self.olon, lat, lon)

        # Compute histogram
        bins = np.linspace(0, 2 * np.pi, self.cmt3d["nregions"] + 1)
        naz, bins = np.histogram(np.array(sta_azi) / 180 * np.pi, bins=bins)

        norm_factor = np.max(naz)

        # Compute bin centers for bars
        binc = bins[:-1] + 0.5 * np.diff(bins)

        bars = plt.bar(binc, naz, width=(bins[1] - bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r / norm_factor))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        plt.xticks(fontsize=8)
        plt.yticks([])
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Put text about the wave and the component
        fontsize = 9
        text = "Wins: %d" % len(sta_azi)
        ax.text(-0.1, 1.10, text, fontsize=fontsize, transform=ax.transAxes,
                horizontalalignment="left", verticalalignment="center",
                zorder=100,
                bbox=dict(facecolor='white', edgecolor='black',
                          pad=3))

    @staticmethod
    def plot_stations(lat, lon):
        """
        Args:
            lat: latitudes
            lon: longitudes

        Returns:
        """
        geo = cartopy.crs.Geodetic()
        plt.plot(lon, lat, "v", markerfacecolor=(0.7, 0.15, 0.15),
                 markeredgecolor='k', markersize=10, transform=geo,
                 zorder=15, clip_on=True)

    @staticmethod
    def plot_paths(elat, elon, latitude, longitude):
        """ Plots raypaths

        Args:
            elat: event latitude
            elon: event longitude
            latitude: station latitudes
            longitude: station longitudes

        Returns:

        """
        geo = cartopy.crs.Geodetic()
        for lat, lon in zip(latitude, longitude):
            plt.plot([lon, elon], [lat, elat],
                     c=(0.3, 0.3, 0.45),
                     lw=0.75, transform=geo, zorder=10)


if __name__ == "__main__":

    cmt3d_json = "/Users/lucassawade/inversion_test/9703873.9p_ZT.stats.json"
    g3d_json = "/Users/lucassawade/inversion_test/9703873.grad.stats.json"
    P = PlotEventSummary.from_JSON(cmt3d_json, g3d_json)
    P.plot_summary()
