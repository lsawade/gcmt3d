from copy import deepcopy
from typing import Union
import json
from ..source import CMTSource
from obspy import Inventory
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
import numpy as np


def load_winfile_json(flexwin_file: str, inv: Union[Inventory, None] = None,
                      event: Union[CMTSource, None] = None,
                      simple: bool = False, v: bool = False):
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

    # Make measurement dicts
    base_measure_dict = dict(epics=[], wins=[], dlnAs=[], cc_shifts=[],
                             max_ccs=[])
    measurement_dict = dict(R=deepcopy(base_measure_dict),
                            T=deepcopy(base_measure_dict),
                            Z=deepcopy(base_measure_dict))
    try:
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
                        dlnA[_idx] = _win["dlnA"]
                        cc_shift[_idx] = _win["cc_shift_in_seconds"]
                        max_cc[_idx] = _win["max_cc_value"]

                    if (inv is not None) and (event is not None):
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

                        except Exception as e:
                            if v:
                                print(channel_id, e)
                            epi = np.nan
                    else:
                        epi = np.nan
                        if 'BHR' in channel_id:
                            channel = 'R'
                        elif 'BHT' in channel_id:
                            channel = 'T'
                        else:
                            channel = 'Z'

                    # Summarize time values and distances
                    for _idx, (_dt, _dlnA, _cc_shift, _max_cc) in \
                            enumerate(zip(dt, dlnA, cc_shift, max_cc)):
                        if simple:
                            times = np.mean(win_time[_idx, :])
                            measurement_dict[channel]["wins"].append(times)
                            measurement_dict[channel]["epics"].append(epi)
                            measurement_dict[channel]["dlnAs"].append(
                                _dlnA)
                            measurement_dict[channel]["cc_shifts"].append(
                                _cc_shift)
                            measurement_dict[channel]["max_ccs"].append(
                                _max_cc)
                        else:
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
        if v:
            print(e)

    return measurement_dict
