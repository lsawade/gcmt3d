from typing import List
import os
import sys
import json
from copy import deepcopy
import numpy as np
import matplotlib
import lwsspy as lpy
from glob import glob
from time import localtime, strftime
from obspy import read_inventory, Inventory, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
from gcmt3d.source import CMTSource
import matplotlib.pyplot as plt
import pickle

lpy.updaterc()


def load_winfile_json(flexwin_file: str, inv: Inventory, event: CMTSource,
                      v: bool = False):
    """Read the json format of window file and output epicentral distances and
    times

    Parameters
    ----------
    flexwin_file : str
        window file
    inv : Inventory
        Obspy inventory to get station locations from
    event : CMTSource
        CMTSource
    v : bool
        verbose flag. Default False.

    Returns
    -------
    tuple
        (list of epicentral distances, list of corresponding times)

    Notes
    -----

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.19 12.00

    """
    base_measure_dict = dict(epics=[], wins=[], dlnAs=[], cc_shifts=[],
                             max_ccs=[])
    measurement_dict = dict(R=deepcopy(base_measure_dict),
                            T=deepcopy(base_measure_dict),
                            Z=deepcopy(base_measure_dict))

    with open(flexwin_file, 'r') as fh:
        content = json.load(fh)
        # Loop over stations and channel.
        for _sta, _channel in content.items():
            # Loop over channel values
            for _chan_win in _channel.values():
                num_wins = len(_chan_win)
                if num_wins <= 0:
                    continue
                channel_id = _chan_win[0]["channel_id"]

                # Create empty window arrays
                win_time = np.zeros([num_wins, 2])
                dt = np.zeros(num_wins)
                dlnA = np.zeros(num_wins)
                cc_shift = np.zeros(num_wins)
                max_cc = np.zeros(num_wins)
                for _idx, _win in enumerate(_chan_win):
                    win_time[_idx, 0] = _win["relative_starttime"]
                    win_time[_idx, 1] = _win["relative_endtime"]
                    dt[_idx] = _win["dt"]
                    dlnA[_idx] = _win["dt"]
                    cc_shift[_idx] = _win["cc_shift_in_seconds"]
                    max_cc[_idx] = _win["max_cc_value"]

                # Compute epicentral distance
                try:
                    if 'BHR' in channel_id:
                        channel = 'R'
                        for chan in ['BH1', 'BHE']:
                            try:
                                nc = channel_id[:-3] + chan
                                coords = inv.get_coordinates(
                                    nc, event.cmt_time)
                                break
                            except Exception as e:
                                if v:
                                    print(channel_id, nc, e)

                    elif 'BHT' in channel_id:
                        channel = 'T'
                        for chan in ['BH2', 'BHN']:
                            try:
                                nc = channel_id[:-3] + chan
                                coords = inv.get_coordinates(
                                    nc, event.cmt_time)
                                break
                            except Exception as e:
                                if v:
                                    print(channel_id, nc, e)
                    else:
                        channel = 'Z'
                        coords = inv.get_coordinates(
                            channel_id, event.cmt_time)

                    # Compute epicentral distance
                    epi = kilometer2degrees(gps2dist_azimuth(
                        event.latitude, event.longitude,
                        coords["latitude"], coords["longitude"])[0]/1000.0)

                    # Summarize time values and distances
                    for _idx, (_dt, _dlnA, _cc_shift, _max_cc) in \
                            enumerate(zip(dt, dlnA, cc_shift, max_cc)):
                        times = np.arange(win_time[_idx, 0],
                                          win_time[_idx, 1] + _dt, _dt).tolist()
                        measurement_dict[channel]["wins"].extend(times)
                        measurement_dict[channel]["epics"].extend(
                            np.ones(len(times)) * epi)
                        measurement_dict[channel]["dlnAs"].extend(
                            np.ones(len(times)) * _dlnA)
                        measurement_dict[channel]["cc_shifts"].extend(
                            np.ones(len(times)) * _cc_shift)
                        measurement_dict[channel]["max_ccs"].extend(
                            np.ones(len(times)) * _max_cc)

                except Exception as e:
                    print(channel_id, e)

    return measurement_dict


def load_stations(outputdir: str = "."):
    """Loads a standard set of stations saves it and

    Parameters
    ----------
    outputdir : str, optional
        [description], by default "."

    Returns
    -------
    [type]
        [description]
    """

    # Networks used in the Global CMT Project
    networks = ["AU", "CU", "G", "GE", "GT", "IC", "II", "IU", "MN", "US"]
    networksstring = ",".join(networks)

    # number of legend columns
    ncol = len(networks)

    # Define file
    xml_name = "gcmt3d_full_station.xml"
    invfile = os.path.join(outputdir, xml_name)

    # Write inventory after downloading
    if os.path.exists(invfile):
        inv = lpy.read_inventory(invfile)
    else:
        # Download metadata
        start = UTCDateTime(1976, 1, 1)
        end = UTCDateTime.now()
        duration = end - start

        # Download the data
        inv = lpy.download_data(start, duration=duration,
                                network=networksstring,
                                station=None, dtype='stations')

        # Write it to the script data directory
        inv.write(invfile, "STATIONXML")

    return inv


# Pre-setup
base_measure_dict = dict(epics=[], wins=[], dlnAs=[],
                         cc_shifts=[], max_ccs=[])
base_wtype_dict = dict(body=deepcopy(base_measure_dict),
                       surface=deepcopy(base_measure_dict),
                       mantle=deepcopy(base_measure_dict))


def create_measurement_pickle(databases: List[str], outputdir: str) -> dict:

    # Load a standard inventory
    inv = load_stations(outputdir)

    filelists = dict(body=[], surface=[], mantle=[])
    for _database in databases:
        for _wtype in filelists.keys():
            # Create Strting to glob the whole thing
            globstr = os.path.join(
                _database, "*", "window_data", f"{_wtype}.windows.json")

            # Glob all files
            files = glob(globstr)

            # Append to superfilelist
            filelists[_wtype].extend(files)

    # empty dict for measurements
    measurements = dict(R=deepcopy(base_wtype_dict),
                        T=deepcopy(base_wtype_dict),
                        Z=deepcopy(base_wtype_dict))

    # Populate the dictionary
    for _wtype, _filelist in filelists.items():
        print(_wtype, _filelist)
        # Load window file content
        for _file in _filelist:
            # Get CMT file in database
            print(_file)
            cmtdir = os.path.dirname(os.path.dirname(_file))
            cmtID = os.path.basename(cmtdir)
            cmtfile = os.path.join(cmtdir, cmtID + ".cmt")
            cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)
            mdict = load_winfile_json(_file, inv, cmtsource)

            # Add window file content to superdict
            for _channel, _measure_dict in mdict.items():
                for _measure, _measurelist in _measure_dict.items():
                    measurements[_channel][_wtype][_measure].extend(
                        _measurelist)

    # Saving the whole thinng to a pickle.
    timestr = strftime("%Y%m%dT%H%M", localtime())
    outfile = os.path.join(
        outputdir, f"window_measurements_{timestr}.pickle")

    with open(outfile, 'wb') as f:
        pickle.dump(measurements, f)


def plot_window_hist(filename: str, outputdir: str, deg_res: float = 0.5,
                     t_res: float = 15.0, minillum: int = 25,
                     illumdecay: int = 50, noplot: bool = False,
                     save: bool = False, fromhistfile: bool = True) -> None:

    if fromhistfile:
        with open(filename, 'rb') as f:
            hists = pickle.load(f)

        # Define bin edges
        xbins = hists['xbins']
        ybins = hists['ybins']
        shape = (xbins.size, ybins.size)

        base_hist_measure_dict = dict(counts=np.ndarray(shape),
                                      dlnAs=np.ndarray(shape),
                                      cc_shifts=np.ndarray(shape),
                                      max_ccs=np.ndarray(shape))
        base_hist_wtype_dict = dict(body=deepcopy(base_hist_measure_dict),
                                    surface=deepcopy(base_hist_measure_dict),
                                    mantle=deepcopy(base_hist_measure_dict))

    else:
        with open(filename, 'rb') as f:
            measurements = pickle.load(f)

        # Define bin edges
        xbins = np.arange(0, 181, deg_res)
        ybins = np.arange(0, 11800, t_res)
        shape = (xbins.size, ybins.size)

        # Empty dict for the histograms
        base_hist_measure_dict = dict(counts=np.ndarray(shape),
                                      dlnAs=np.ndarray(shape),
                                      cc_shifts=np.ndarray(shape),
                                      max_ccs=np.ndarray(shape))
        base_hist_wtype_dict = dict(body=deepcopy(base_hist_measure_dict),
                                    surface=deepcopy(base_hist_measure_dict),
                                    mantle=deepcopy(base_hist_measure_dict))

        # Empty hist
        hists = dict(R=deepcopy(base_hist_wtype_dict),
                     T=deepcopy(base_hist_wtype_dict),
                     Z=deepcopy(base_hist_wtype_dict),
                     xbins=xbins, ybins=ybins)

        for _channel in ["R", "T", "Z"]:
            for _wtype in ["body", "surface", "mantle"]:
                # Count histogram
                hists[_channel][_wtype]["counts"], _, _ = np.histogram2d(
                    measurements[_channel][_wtype]["epics"],
                    measurements[_channel][_wtype]["wins"],
                    bins=(xbins, ybins))

                # Data histograms
                dat_list = ["dlnAs", "cc_shifts", "max_ccs"]
                for _dat in dat_list:
                    hists[_channel][_wtype][_dat], _, _ = np.histogram2d(
                        measurements[_channel][_wtype]["epics"],
                        measurements[_channel][_wtype]["wins"],
                        bins=(xbins, ybins),
                        weights=measurements[_channel][_wtype][_dat])

        # Save histogram file if wanted.
        if save:

            timestr = strftime("%Y%m%dT%H%M", localtime())
            outfile = os.path.join(
                outputdir, f"window_hists_{timestr}.pickle")
            with open(outfile, 'wb') as f:
                pickle.dump(hists, f)

    if noplot:
        return None

    # Create Combined
    hists_wavetype = deepcopy(base_hist_wtype_dict)
    hists_channel = dict(R=deepcopy(base_hist_measure_dict),
                         T=deepcopy(base_hist_measure_dict),
                         Z=deepcopy(base_hist_measure_dict))
    hists_comb = deepcopy(base_hist_measure_dict)

    # Labels components etc.
    abc = 'abcdefghijklmnopqrstuvwxyz'
    wavetypes = ['body', 'surface', 'mantle']
    wave_labels = [wave.capitalize() for wave in wavetypes]
    channels = ['R', 'T', 'Z']
    channel_labels = ["Radial", "Transverse", "Vertical"]

    # Populate the combined histograms
    for _j, (_channel, _clabel) in enumerate(zip(channels, channel_labels)):
        for _i, (_wtype, _wlabel) in enumerate(zip(wavetypes, wave_labels)):
            for _k, _measure in enumerate(
                    ["counts", "dlnAs", "cc_shifts", "max_ccs"]):
                if _i == 0:
                    hists_channel[_channel][_measure] = \
                        hists[_channel][_wtype][_measure]
                else:
                    hists_channel[_channel][_measure] += \
                        hists[_channel][_wtype][_measure]

                if _j == 0:
                    hists_wavetype[_wtype][_measure] = \
                        hists[_channel][_wtype][_measure]
                else:
                    hists_wavetype[_wtype][_measure] += \
                        hists[_channel][_wtype][_measure]

                if _i == 0 and _j == 0:
                    hists_comb[_measure] = \
                        hists[_channel][_wtype][_measure]
                else:
                    hists_comb[_measure] += \
                        hists[_channel][_wtype][_measure]

    # # Ge boolean array about where we have data and where not
    # boolcounts = hists[_channel][_wtype]["counts"].astype(bool)
    # # Workaround for zero count values tto not get an error.
    # # Where counts == 0, zi = 0, else zi = zz/counts
    # zi = np.zeros_like(zz)
    # zi[boolcounts] = zz[boolcounts] / \
    #     hists[_channel][_wtype]["counts"][boolcounts]
    # hists[_channel][_wtype][_dat] = np.ma.masked_equal(zi, 0)

    # # Populate the combined histograms
    # # Data histograms
    # dat_list = ["dlnAs", "cc_shifts", "max_ccs"]

    # for _j, (_channel, _clabel) in enumerate(zip(channels, channel_labels)):

    #     # Ge boolean array about where we have data and where not
    #     boolcounts = hists_channel[_channel]["counts"].astype(bool)
    #     # Workaround for zero count values tto not get an error.
    #     # Where counts == 0, zi = 0, else zi = zz/counts
    #     for _dat in dat_list:
    #         zi = np.zeros(shape)
    #         zi[boolcounts] = hists_channel[_channel][_dat][boolcounts] / \
    #             hists_channel[_channel]["counts"][boolcounts]
    #         hists_channel[_channel][_dat] = np.ma.masked_equal(zi, 0)

    # for _i, (_wtype, _wlabel) in enumerate(zip(wavetypes, wave_labels)):

    #     # Ge boolean array about where we have data and where not
    #     boolcounts = hists_channel[_channel]["counts"].astype(bool)
    #     # Workaround for zero count values tto not get an error.
    #     # Where counts == 0, zi = 0, else zi = zz/counts
    #     for _dat in dat_list:
    #         zi = np.zeros(shape)
    #         zi[boolcounts] = hists_channel[_channel][_dat][boolcounts] / \
    #             hists_channel[_channel]["counts"][boolcounts]
    #         hists_wavetype[_channel][_dat] = np.ma.masked_equal(zi, 0)

    def get_illumination(mat: np.ndarray, minillum: int = 25,
                         illumdecay: int = 50, r: bool = False):
        # Get locations of the numbers
        below_minillum = np.where(mat < minillum)
        between = np.where((minillum < mat) & (mat < illumdecay))
        above_illumdecay = np.where(mat > illumdecay)

        # Create alpha array
        alphas = np.zeros_like(mat)
        alphas[above_illumdecay] = (illumdecay - minillum)
        alphas[below_minillum] = 0
        alphas[between] = mat[between] - (illumdecay - minillum)
        alphas = alphas/(illumdecay - minillum)

        if r:
            alphas[between] = 1.0 - alphas[between]
            alphas[above_illumdecay] = 0.0
            alphas[below_minillum] = 1.0
        return alphas

    def set_facealpha(p, alphaarray):
        print(np.min(alphaarray), np.max(alphaarray), np.mean(alphaarray),
              np.median(alphaarray))

    alphas = get_illumination(hists_comb["counts"].T, minillum, illumdecay)
    alphas_r = get_illumination(hists_comb["counts"].T, minillum, illumdecay,
                                r=True)
    xmin, xmax = np.min(hists["xbins"]), np.max(hists["xbins"])
    ymin, ymax = np.min(hists["ybins"]), np.max(hists["ybins"])
    extent = [xmin, xmax, ymin/60, ymax/60]

    fig = plt.figure()
    ax = plt.axes()
    # ax.set_facecolor((0.3, 0.3, 0.3))
    # pm = ax.pcolormesh(hists["xbins"], hists["ybins"] /
    #                    60, hists_comb["counts"].T, cmap='afmhot_r')
    alphamat = 0.8 * np.ones((shape[1] - 1, shape[0] - 1, 4))
    alphamat[:, :, 3] = alphas_r
    # im1 = ax.imshow(hists_comb["counts"].T, cmap='afmhot_r',
    #                 extent=extent, alpha=alphas)
    # pm = ax.pcolormesh(hists["xbins"], hists["ybins"] /
    #                    60, hists_comb["counts"].T, cmap='afmhot_r',
    #                    zorder=-1)
    im1 = ax.imshow(hists_comb["counts"].T[::-1, :], cmap='afmhot_r',
                    interpolation='none', extent=extent, aspect='auto',
                    alpha=alphas[::-1, :])
    im2 = ax.imshow(alphamat[::-1, :, :], interpolation='none',
                    extent=extent, zorder=0, aspect='auto')

    # pg = plt.pcolormesh(hists["xbins"], hists["ybins"] /
    #                     60, np.zeros_like(hists_comb["counts"].T),
    #                     color=(0.3, 0.3, 0.3))
    c = lpy.nice_colorbar(
        matplotlib.cm.ScalarMappable(cmap=im1.cmap, norm=im1.norm),
        orientation='horizontal', aspect=40)  # , ax=axes.ravel().tolist())
    plt.xlabel("$\\Delta$ [$^\\circ$]")
    lpy.plot_label(ax, "Radial", location=1, box=False, dist=0.01)
    ax.xaxis.set_label_position('top')
    ax.tick_params(labelbottom=False, labeltop=True)
    plt.ylabel("Traveltime [min]")
    # set_facealpha(pm, alphas)
    # set_facealpha(pm, alphas_r)
    # fig.canvas.draw()
    # fc = pm.cmap(pm.norm(pm.get_array()))
    # fc[:, 3] = alphas.flatten()
    # print(fc)
    # print(np.mean(fc, axis=0))
    # pm.set_facecolor(fc)
    # print(pm.get_facecolors())

    # plt.show(block=False)
    # # set_facealpha(pm, alphas)
    # # set_facealpha(pg, alphas_r)
    # fig.canvas.draw()
    plt.show(block=True)
    # nsp = 330
    # axes = []
    # # for _i, (_wtype, _wlabel) in enumerate(wavetypes, wave_labels):
    #     subaxes = []
    #     for _j, (_channel, _clabel) in enumerate(channels, channel_labels):
    #         ax = plt.subplots(nsp + _i * 3 + _j)
    #         subaxes.append(ax)
    #     axes.append(subaxes)

    # for _i, (_wtype, _wlabel) in enumerate(wavetypes, wave_labels):
    #     for _j, (_channel, _clabel) in enumerate(channels, channel_labels):

    #     for _j, (_channel, _clabel) in enumerate(channels, channel_labels):
    #         subaxes[_j][_j].pcolormesh(hists["xbins"], hists["ybins"] /
    #             60, hists["R"]["surface"]["counts"].T, cmap='afmhot_r'))
    #         c = lpy.nice_colorbar(pm, orientation = 'horizontal',
    #                               aspect = 40, ax = axes.ravel().tolist())
    #         plt.xlabel("$\\Delta$ [$^\\circ$]")
    #         lpy.plot_label(ax, "Radial", location = 1,
    #                        aspect = 0.5, box = False)
    #         ax.xaxis.set_label_position('top')
    #         ax.tick_params(labelbottom = False, labeltop = True)
    #         plt.ylabel("Traveltime [min]")
    #         plt.show()

    return None


def bin_create_measurement_pickle():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("databaselocs", nargs="+", type=str,
                        help="Input format: plot-window-hist path/to/database path/to/database")
    parser.add_argument("-o", dest="outputdir", default=".", type=str,
                        help="Directory to save the images to",
                        required=False)
    args = parser.parse_args()

    create_measurement_pickle(args.databaselocs, args.outputdir)


def bin():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str,
                        help="filename for either a measurement or histogram file")
    parser.add_argument("-o", dest="outputdir", default=".", type=str,
                        help="Directory to save the images to",
                        required=False)
    parser.add_argument("-s", "--save", dest="save", action="store_true",
                        help="Flag to save npzfile with histograms",
                        default=False, required=False)
    parser.add_argument("-d", "--deg_res", dest="deg_res", type=float,
                        default=0.5, required=False,
                        help="Define histogram epicentral resolution.")
    parser.add_argument("-t", "--t_res", dest="t_res", type=float,
                        default=15, required=False,
                        help="Define histogram time in seconds resolution.")
    parser.add_argument("-m", "--minillum", dest="minillum", type=int,
                        default=25, required=False,
                        help="Illum necessary for minimum illumination")
    parser.add_argument("-i", "--illumdecay", dest="illumdecay", type=int,
                        default=50, required=False,
                        help="Illum necessary for full illumination")
    parser.add_argument("-f", "--fromhistfile", dest="fromhistfile",
                        help="Flag to load npzfile with histograms",
                        action="store_true", default=False, required=False)
    parser.add_argument("-np", "--noplot", dest="noplot", action="store_true",
                        help="Flag to define whether to plot or not",
                        default=False, required=False)
    args = parser.parse_args()

    plot_window_hist(args.filename, outputdir=args.outputdir,
                     deg_res=args.deg_res, t_res=args.t_res,
                     minillum=args.minillum, illumdecay=args.illumdecay,
                     noplot=args.noplot, fromhistfile=args.fromhistfile,
                     save=args.save)
