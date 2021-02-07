from typing import List
import os
from copy import deepcopy
import numpy as np
import matplotlib
import lwsspy as lpy
from glob import glob
from time import localtime, strftime
from obspy import UTCDateTime
from gcmt3d.source import CMTSource
from ..window.load_winfile_json import load_winfile_json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import pandas as pd
lpy.updaterc()


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


def create_measurement_pickle(databases: List[str], outputdir: str,
                              simple: bool = True):

    # Load a standard inventory
    lpy.print_bar("Loading Obspy Inventory:")
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

    lpy.print_bar("Adding measurements:")
    # Populate the dictionary
    for _wtype, _filelist in filelists.items():
        lpy.print_section(f"Looping over {_wtype} files")
        # Load window file content
        for _file in _filelist:
            lpy.print_action(f"Adding {_file}")
            # Get CMT file in database
            cmtdir = os.path.dirname(os.path.dirname(_file))
            cmtID = os.path.basename(cmtdir)
            cmtfile = os.path.join(cmtdir, cmtID + ".cmt")
            cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)
            mdict = load_winfile_json(
                _file, inv=inv, event=cmtsource, simple=simple)

            # Add window file content to superdict
            for _channel, _measure_dict in mdict.items():
                for _measure, _measurelist in _measure_dict.items():
                    print(_channel, _wtype, _measure, len(_measurelist))
                    measurements[_channel][_wtype][_measure].extend(
                        _measurelist)

    lpy.print_section("Reading measurements done.")

    # Saving the whole thing to a pickle.
    timestr = strftime("%Y%m%dT%H%M", localtime())

    # Create HDF5 Storage
    if simple:
        post = "simple_"
    else:
        post = ""
    outfile = os.path.join(
        outputdir, f"window_measurements_{post}{timestr}.h5")
    lpy.print_bar("Opening HDF5 file...")
    store = pd.HDFStore(outfile, 'w')

    # Create pandas dataframes from the measurements
    for _comp, _wtype_dict in measurements.items():
        for _wtype in _wtype_dict.keys():

            # Create tag for hdf5
            tag = f"{_comp}/{_wtype}"

            # Print
            lpy.print_action(f"Saving {tag:9} to {outfile}")

            # Createt Dataframe for measurements
            df = pd.DataFrame.from_dict(measurements[_comp][_wtype])

            # Save the
            store.put(tag, df, format='fixed')

            del df

    store.close()


def plot_window_hist_tdist(filename: str, outputdir: str, deg_res: float = 0.1,
                           t_res: float = 15, minillum: int = 0,
                           illumdecay: int = 50, noplot: bool = False,
                           save: bool = False,
                           fromhistfile: bool = True) -> None:

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
        # Load HDF5 file with measurements
        store = pd.HDFStore(filename, 'r')

        # Define bin edges
        xbins = np.arange(0, 181, deg_res)
        ybins = np.arange(0, 11800, t_res)
        shape = (xbins.size - 1, ybins.size - 1)

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

        # Compute histograms
        for _channel in ["R", "T", "Z"]:
            for _wtype in ["body", "surface", "mantle"]:

                lpy.print_action(f"Binning {_channel}/{_wtype}")

                # Count histogram
                hists[_channel][_wtype]["counts"], _, _ = np.histogram2d(
                    store[f"{_channel}/{_wtype}"]["epics"].to_numpy(),
                    store[f"{_channel}/{_wtype}"]["wins"].to_numpy(),
                    bins=(xbins, ybins))

                # Data histograms
                dat_list = ["dlnAs", "cc_shifts", "max_ccs"]
                for _dat in dat_list:
                    hists[_channel][_wtype][_dat], _, _ = np.histogram2d(
                        store[f"{_channel}/{_wtype}"]["epics"].to_numpy(),
                        store[f"{_channel}/{_wtype}"]["wins"].to_numpy(),
                        bins=(xbins, ybins),
                        weights=store[f"{_channel}/{_wtype}"][_dat].to_numpy())
        # Close HDF5
        store.close()
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
    # abc = 'abcdefghijklmnopqrstuvwxyz'
    wavetypes = ['body', 'surface', 'mantle']
    wave_labels = [wave.capitalize() for wave in wavetypes]
    channels = ['R', 'T', 'Z']
    channel_labels = ["Radial", "Transverse", "Vertical"]

    # Populate the combined histograms
    for _j, (_channel, _clabel) in enumerate(zip(channels, channel_labels)):
        for _i, (_wtype, _wlabel) in enumerate(zip(wavetypes, wave_labels)):
            for _k, _measure in enumerate(
                    ["counts", "dlnAs", "cc_shifts", "max_ccs"]):
                # Create combined histogram
                if _i == 0 and _j == 0:
                    hists_comb[_measure] = \
                        deepcopy(hists[_channel][_wtype][_measure])
                else:
                    hists_comb[_measure] += \
                        hists[_channel][_wtype][_measure]

                # Create combined wavetypes histogram
                if _i == 0:
                    hists_channel[_channel][_measure] = \
                        deepcopy(hists[_channel][_wtype][_measure])
                else:
                    hists_channel[_channel][_measure] += \
                        hists[_channel][_wtype][_measure]

                # Create combined channels histogram
                if _j == 0:
                    hists_wavetype[_wtype][_measure] = \
                        deepcopy(hists[_channel][_wtype][_measure])
                else:
                    hists_wavetype[_wtype][_measure] += \
                        hists[_channel][_wtype][_measure]

                if _measure == "counts":
                    print(f"{_channel} Counts mean:",
                          hists_channel[_channel]["counts"].mean())
                    print(f"{_wtype} Counts mean:",
                          hists_wavetype[_wtype]["counts"].mean())

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
        alphas[between] = mat[between] - minillum
        alphas = alphas/(illumdecay - minillum)

        if r:
            alphas[between] = 1.0 - alphas[between]
            alphas[above_illumdecay] = 0.0
            alphas[below_minillum] = 1.0
        return alphas

    # Print combined figures
    # Get main info for plotting
    xmin, xmax = np.min(hists["xbins"]), np.max(hists["xbins"])
    ymin, ymax = np.min(hists["ybins"]), np.max(hists["ybins"])
    extent = [xmin, xmax, ymin/60, ymax/60]

    combo_plot = False
    if combo_plot:
        alphas = get_illumination(
            hists_comb["counts"].T[::-1, :], 100, 200)

        # Divide sum by all other measurements
        # Dataplotting dict
        dpd = dict(
            counts=dict(
                cmap="gray_r", norm=colors.LogNorm(
                    vmin=1000, vmax=hists_comb["counts"].max()),
                label="Counts"),
            dlnAs=dict(
                cmap="seismic",
                norm=lpy.MidpointNormalize(vmin=-0.5, vmax=0.5, midpoint=0.0),
                label="dlnA"),
            cc_shifts=dict(
                cmap="seismic",
                norm=lpy.MidpointNormalize(
                    vmin=-12.5, vmax=12.5, midpoint=0.0),
                label="CC-$\\Delta$ t"),
            max_ccs=dict(
                cmap="gray_r",
                norm=colors.Normalize(vmin=0.925, vmax=0.98),
                label="Max. CrossCorr"),
        )
        fig = plt.figure(figsize=(10, 6))
        spcount = 141
        fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.0)
        axes = []
        boolcounts = hists_comb["counts"].astype(bool)
        for _i, (_dat, _hist) in enumerate(hists_comb.items()):

            # Define Data
            if _dat == "counts":
                zz = _hist
            else:
                zz = np.zeros_like(hists_comb[_dat])
                zz[boolcounts] = hists_comb[_dat][boolcounts] / \
                    hists_comb["counts"][boolcounts]

            if _i == 0:
                axes.append(plt.subplot(spcount))
            else:
                axes.append(plt.subplot(
                    spcount + _i, sharey=axes[0], sharex=axes[0]))
                axes[_i].tick_params(labelleft=False)

            if _dat == "counts":
                im1 = axes[_i].imshow(zz.T[::-1, :], cmap=dpd[_dat]['cmap'],
                                      interpolation='none', extent=extent,
                                      norm=dpd[_dat]['norm'], aspect='auto')
                plt.ylabel("Traveltime [min]")
            else:
                im1 = axes[_i].imshow(
                    zz.T[::-1, :], cmap=dpd[_dat]['cmap'],
                    interpolation='none', norm=dpd[_dat]['norm'],
                    extent=extent, aspect='auto', alpha=alphas)

            lpy.plot_label(axes[_i],
                           f"$\\mu$ = {zz[boolcounts].mean():>4.2f}\n"
                           f"$\\sigma$ = {zz[boolcounts].std():>4.2f}\n",
                           location=2, box=False,
                           fontdict={"fontfamily": "monospace"},
                           )
            c = lpy.nice_colorbar(
                matplotlib.cm.ScalarMappable(cmap=im1.cmap, norm=im1.norm),
                pad=0.025, orientation='horizontal', aspect=40)

            # Set labels
            c.set_label(dpd[_dat]['label'])

            plt.xlabel("$\\Delta$ [$^\\circ$]")

            # Put xlabel on top
            axes[_i].xaxis.set_label_position('top')
            axes[_i].tick_params(labelbottom=False, labeltop=True)

    single_hists = True
    if single_hists:

        # Create plotting style dictionary
        dpd = dict(
            body=dict(
                counts=dict(
                    cmap="gray_r", norm=colors.LogNorm(
                        vmin=100,
                        vmax=hists_wavetype['body']["counts"].max()*0.75),
                    label="Counts"),
                dlnAs=dict(
                    cmap="seismic",
                    norm=lpy.MidpointNormalize(midpoint=0.0),
                    label="dlnA"),
                cc_shifts=dict(
                    cmap="seismic",
                    norm=lpy.MidpointNormalize(
                        vmin=-12.5, vmax=12.5, midpoint=0.0),
                    label="CC-$\\Delta$ t"),
                max_ccs=dict(
                    cmap="gray_r",
                    norm=colors.Normalize(vmin=0.9, vmax=0.98),
                    label="Max. CrossCorr")
            ),
            surface=dict(
                counts=dict(
                    cmap="gray_r", norm=colors.LogNorm(
                        vmin=100,
                        vmax=hists_wavetype['body']["counts"].max()*0.75),
                    label="Counts"),
                dlnAs=dict(
                    cmap="seismic",
                    norm=lpy.MidpointNormalize(midpoint=0.0),
                    label="dlnA"),
                cc_shifts=dict(
                    cmap="seismic",
                    norm=lpy.MidpointNormalize(
                        vmin=-12.5, vmax=12.5, midpoint=0.0),
                    label="CC-$\\Delta$ t"),
                max_ccs=dict(
                    cmap="gray_r",
                    norm=colors.Normalize(vmin=0.9, vmax=0.98),
                    label="Max. CrossCorr")
            ),
            mantle=dict(
                counts=dict(
                    cmap="gray_r", norm=colors.LogNorm(
                        vmin=100,
                        vmax=hists_wavetype['body']["counts"].max()*0.75),
                    label="Counts"),
                dlnAs=dict(
                    cmap="seismic",
                    norm=lpy.MidpointNormalize(midpoint=0.0),
                    label="dlnA"),
                cc_shifts=dict(
                    cmap="seismic",
                    norm=lpy.MidpointNormalize(
                        vmin=-12.5, vmax=12.5, midpoint=0.0),
                    label="CC-$\\Delta$ t"),
                max_ccs=dict(
                    cmap="gray_r",
                    norm=colors.Normalize(vmin=0.9, vmax=0.98),
                    label="Max. CrossCorr"))
        )

        for _dat in hists_wavetype["body"].keys():
            for _wtype in hists["R"].keys():

                fig = plt.figure(figsize=(9, 6))
                fig.subplots_adjust(
                    left=0.06, right=0.94, top=0.925, bottom=-0.05, wspace=0.2
                )
                spcount = 131
                axes = []
                for _i, (_channel) in enumerate(["Z", "R", "T"]):
                    boolcounts = hists[_channel][_wtype]["counts"].astype(bool)
                    alphas = get_illumination(
                        hists[_channel][_wtype]["counts"].T[::-1, :], 25, 75)

                    # Define Data
                    if _dat == "counts":
                        zz = hists[_channel][_wtype]["counts"]
                    else:
                        zz = np.zeros_like(hists[_channel][_wtype][_dat])
                        zz[boolcounts] = \
                            hists[_channel][_wtype][_dat][boolcounts] / \
                            hists[_channel][_wtype]["counts"][boolcounts]

                    if _i == 0:
                        axes.append(plt.subplot(spcount))
                        plt.ylabel("Traveltime [min]")
                    else:
                        axes.append(plt.subplot(
                            spcount + _i, sharey=axes[0], sharex=axes[0]))
                        axes[_i].tick_params(labelleft=False)

                    if _dat == "counts":
                        im1 = axes[_i].imshow(
                            zz.T[::-1, :], cmap=dpd[_wtype][_dat]['cmap'],
                            interpolation='none', extent=extent,
                            norm=dpd[_wtype][_dat]['norm'], aspect='auto',
                            zorder=-10)
                    else:
                        im1 = axes[_i].imshow(
                            zz.T[::-1, :], cmap=dpd[_wtype][_dat]['cmap'],
                            interpolation='none',
                            norm=dpd[_wtype][_dat]['norm'],
                            extent=extent, aspect='auto', alpha=alphas,
                            zorder=-10)

                    # Plot traveltimes and fix limits
                    if _wtype == "body":
                        if _dat in ["dlnAs", "cc_shifts"]:
                            cmap = 'Dark2'
                        else:
                            cmap = 'rainbow'
                        axes[_i].set_ylim(0.0, 60.0)
                        lpy.plot_traveltimes_ak135(
                            ax=axes[_i], cmap=cmap,
                            labelkwargs=dict(fontsize='small'))
                    elif _wtype == "surface":
                        axes[_i].set_ylim(0.0, 120.0)
                    else:
                        axes[_i].set_ylim(0.0, 180.0)

                    mu = zz[boolcounts].mean()
                    sig = zz[boolcounts].std()
                    labint, ndec = lpy.get_stats_label_length(mu, sig, ndec=2)
                    lpy.plot_label(axes[_i],
                                   f"$\\mu$ = {mu:>{labint}.{ndec}f}\n"
                                   f"$\\sigma$ = {sig:>{labint}.2f}",
                                   location=4, box=False,
                                   fontdict=dict(
                                       fontfamily="monospace",
                                       fontsize="small"))

                    # Plot Type label
                    if _wtype == "mantle":
                        location = 2
                    else:
                        location = 1
                    lpy.plot_label(
                        axes[_i],
                        f"{_wtype.capitalize()}\n{_channel.capitalize()}",
                        location=location, box=False,
                        fontdict=dict(fontsize="small"))

                    # Plot colorbar
                    c = lpy.nice_colorbar(
                        matplotlib.cm.ScalarMappable(
                            cmap=im1.cmap, norm=im1.norm),
                        pad=0.025, orientation='horizontal', aspect=40)

                    # Set labels
                    c.set_label(dpd[_wtype][_dat]['label'])
                    plt.xlabel("$\\Delta$ [$^\\circ$]")

                    # Put xlabel on top
                    axes[_i].xaxis.set_label_position('top')
                    axes[_i].tick_params(labelbottom=False, labeltop=True)

                    # Set rasterization zorder to rasteriz images for pdf
                    # output
                    axes[_i].set_rasterization_zorder(-5)

                outfile = os.path.join(
                    outputdir, f"{_wtype}_{_dat}.pdf")
                plt.savefig(outfile)

    # plt.show(block=True)

    return None


def plot_window_hist(filename: str, outputdir: str,
                     fromhistfile: bool = True, save: bool = False) -> None:

    if fromhistfile:
        with open(filename, 'rb') as f:
            hists = pickle.load(f)

    else:
        # Load HDF5 file with measurements
        store = pd.HDFStore(filename, 'r')

        # Empty dict for the histograms
        base_hist_measure_dict = dict(
            dlnAs=dict(hist=None, edges=None, mean=None, std=None),
            cc_shifts=dict(hist=None, edges=None, mean=None, std=None),
            max_ccs=dict(hist=None, edges=None, mean=None, std=None)
        )
        base_hist_wtype_dict = dict(body=deepcopy(base_hist_measure_dict),
                                    surface=deepcopy(base_hist_measure_dict),
                                    mantle=deepcopy(base_hist_measure_dict))

        # Empty hist
        hists = dict(R=deepcopy(base_hist_wtype_dict),
                     T=deepcopy(base_hist_wtype_dict),
                     Z=deepcopy(base_hist_wtype_dict))

        # Compute histograms
        for _i, _channel in enumerate(["R", "T", "Z"]):
            for _j, _wtype in enumerate(["body", "surface", "mantle"]):

                lpy.print_action(f"Binning {_channel}/{_wtype}")

                # Data histograms
                dat_list = ["dlnAs", "cc_shifts", "max_ccs"]
                for _dat in dat_list:
                    data = store[f"{_channel}/{_wtype}"][_dat].to_numpy()
                    hists[_channel][_wtype][_dat]["mean"] = data.mean()
                    hists[_channel][_wtype][_dat]["std"] = data.std()
                    if _dat == "cc_shifts":
                        nbins = 25
                        brange = [-25, 25]
                    elif _dat == "dlnAs":
                        nbins = 50
                        brange = None
                    else:
                        nbins = 50
                        brange = [0.85, 1.0]

                    hists[_channel][_wtype][_dat]["hist"], \
                        hists[_channel][_wtype][_dat]["edges"] = \
                        np.histogram(data, bins=nbins, range=brange)

        # Close HDF5
        store.close()
        # Save histogram file if wanted.
        if save:

            timestr = strftime("%Y%m%dT%H%M", localtime())
            outfile = os.path.join(
                outputdir, f"window_hists_comp_wtype{timestr}.pickle")
            with open(outfile, 'wb') as f:
                pickle.dump(hists, f)

    # Data histograms
    dat_list = ["dlnAs", "cc_shifts", "max_ccs"]
    for _dat in dat_list:
        plt.figure(figsize=(8, 7))
        plt.subplots_adjust(left=0.075, right=0.925,
                            bottom=0.075, top=0.925, wspace=0.375, hspace=0.15)

        counter = 1
        for _i, _channel in enumerate(["Z", "R", "T"]):
            for _j, _wtype in enumerate(["body", "surface", "mantle"]):

                # Create axes
                if counter == 1:
                    ax = plt.subplot(3, 3, counter)
                else:
                    ax = plt.subplot(3, 3, counter, sharex=ax)

                # Draw histogram using previously created histograms
                mean = hists[_channel][_wtype][_dat]["mean"]
                std = hists[_channel][_wtype][_dat]["std"]
                bins = hists[_channel][_wtype][_dat]["edges"]
                counts = hists[_channel][_wtype][_dat]["hist"]
                centroids = (bins[1:] + bins[:-1]) / 2
                counts_, bins_, _ = plt.hist(
                    centroids, bins=len(counts),
                    weights=counts, range=(np.min(bins), np.max(bins)),
                    facecolor=(0.8, 0.8, 0.8), edgecolor='k',
                    histtype='stepfilled')

                # Set limits
                ax.set_ylim(0, np.max(counts) * 1.3)
                ax.set_xlim(np.min(bins), np.max(bins))

                # Plot vertical 0 line
                ax.grid('on', linewidth=0.75)

                # Plot label
                if _i == 2:
                    if _dat == "cc_shifts":
                        ax.set_xlabel("CC-$\\Delta t$")
                    elif _dat == "dlnAs":
                        ax.set_xlabel("dlnA")
                    elif _dat == "max_ccs":
                        ax.set_xlabel("Max CC")
                else:
                    ax.tick_params(labelbottom=False)

                # Plot channel/wave tyoe label
                lpy.plot_label(ax, f"{_wtype.capitalize()}\n{_channel}",
                               box=dict(
                                   facecolor='w', edgecolor='none', pad=0.2,
                                   alpha=0.66),
                               fontdict=dict(fontsize="small"),
                               )

                # Plot stats label
                labint, ndec = lpy.get_stats_label_length(mean, std, ndec=2)
                lpy.plot_label(ax,
                               f"$\\mu$ = {mean:>{labint}.{ndec}f}\n"
                               f"$\\sigma$ = {std:>{labint}.2f}",
                               location=2,
                               fontdict=dict(
                                   fontfamily="monospace",
                                   fontsize="small"),
                               box=dict(
                                   facecolor='w', edgecolor='none', pad=0.2,
                                   alpha=0.66))

                counter += 1

        # Save plot
        filename = f"window_hist_{_dat}.pdf"
        outfile = os.path.join(outputdir, filename)
        plt.savefig(outfile)

    plt.show()


def bin_create_measurement_pickle():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("databaselocs", nargs="+", type=str,
                        help="Input format: plot-window-hist path/to/database "
                        "path/to/database")
    parser.add_argument("-o", dest="outputdir", default=".", type=str,
                        help="Directory to save the images to",
                        required=False)
    parser.add_argument("-s", "--simple", dest="simple", action="store_true",
                        help="Flag to create simple histogramfiles containing"
                        "only the window centers.",
                        default=False, required=False)
    args = parser.parse_args()

    create_measurement_pickle(args.databaselocs, args.outputdir, args.simple)


def bin():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str,
                        help="filename for either a measurement or "
                        "histogram file")
    parser.add_argument("-o", dest="outputdir", default=".", type=str,
                        help="Directory to save the images to",
                        required=False)
    parser.add_argument("-s", "--save", dest="save", action="store_true",
                        help="Flag to save npzfile with histograms",
                        default=False, required=False)
    parser.add_argument("-d", "--deg_res", dest="deg_res", type=float,
                        default=1.0, required=False,
                        help="Define histogram epicentral resolution.")
    parser.add_argument("-t", "--t_res", dest="t_res", type=float,
                        default=2.0, required=False,
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

    plot_window_hist_tdist(args.filename, outputdir=args.outputdir,
                           deg_res=args.deg_res, t_res=args.t_res,
                           minillum=args.minillum, illumdecay=args.illumdecay,
                           noplot=args.noplot, fromhistfile=args.fromhistfile,
                           save=args.save)


def bin_plot_hist():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str,
                        help="filename for either a measurement or "
                        "histogram file")
    parser.add_argument("-o", dest="outputdir", default=".", type=str,
                        help="Directory to save the images to",
                        required=False)
    parser.add_argument("-s", "--save", dest="save", action="store_true",
                        help="Flag to save npzfile with histograms",
                        default=False, required=False)
    parser.add_argument("-f", "--fromhistfile", dest="fromhistfile",
                        help="Flag to load npzfile with histograms",
                        action="store_true", default=False, required=False)
    args = parser.parse_args()

    plot_window_hist(args.filename, outputdir=args.outputdir,
                     fromhistfile=args.fromhistfile,
                     save=args.save)
