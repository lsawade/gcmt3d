import os
import inspect
import pytest
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from obspy import read, read_inventory, read_events, UTCDateTime
from pyflex import WindowSelector
from pyflex.window import Window
import gcmt3d.window.window as win
import gcmt3d.window.io as wio
import warnings


def assertUTCDateTimeEqual(UTC1, UTC2):
    """Testing whether UTC Date and time are equal."""
    assert UTC1 == UTC2


def is_numeric(obj):
    attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


def assertDictAlmostEqual(dict1, dict2):
    """Test whether two dictionaries are almost equal"""

    if len(dict2) is not len(dict1):
        raise AssertionError("Dictionaries have different sizes.")

    # Go through all keys in the dictionary
    for key in dict1:

        # Get Value from dictionary
        val1 = dict1[key]

        try:
            val2 = dict2[key]
        except KeyError:
            warnings.warn("One of the dictionaries did not contain a keyword.")
            val2 = None

        if val2 is None:
            continue

        elif type(val1) is dict and type(val2) is dict:
            assertDictAlmostEqual(val1, val2)

        # Check if almost equal of both values
        elif is_numeric(val1) and is_numeric(val2):
            np.testing.assert_almost_equal(val1, val2)

        elif type(val1) is UTCDateTime and type(val2) is UTCDateTime:
            assertUTCDateTimeEqual(val1, val2)

        elif type(val1) is str and type(val2) is str:
            assert val1 == val2

        elif type(val1) is not type(val2):
            if is_numeric(val1) and is_numeric(val2):
                np.testing.assert_almost_equal(val1, val2)
            else:
                raise AssertionError("Too fields are not equal.")

        elif val1 is val2 is None:
            pass

        else:
            raise AssertionError("Dictionary: Dont know whats going on.")


def assertListAlmostEqual(list1, list2):
    """Test whether two lists are almost equal"""

    if len(list2) is not len(list1):
        raise AssertionError("lists have different sizes.")

    # Go through all things in list

    for thing1, thing2 in zip(list1, list2):

        # The below things are basically just to satisfy the possible
        # list entries
        if type(thing1) is dict and type(thing2) is dict:
            assertDictAlmostEqual(thing1, thing2)

        # Check if almost equal of both values
        elif type(thing1) in [int, float, complex, np.float, np.float32] \
                and type(thing2) in \
                [int, float, complex, np.float, np.float32]:
            np.testing.assert_almost_equal(thing1, thing2)

        elif type(thing1) is list and type(thing2) is list:
            assertListAlmostEqual(thing1, thing2)

        elif type(thing1) is str and type(thing2) is str:
            assert thing1 == thing2

        elif type(thing1) is UTCDateTime and type(thing2) is UTCDateTime:
            assertUTCDateTimeEqual(thing1, thing2)

        elif type(thing1) is not type(thing2):
            if is_numeric(thing1) and is_numeric(thing2):
                np.testing.assert_almost_equal(thing1, thing2)
            else:
                raise AssertionError("Too fields are not equal.")

        elif thing1 is thing2 is None:
            pass

        else:
            raise AssertionError("List: Dont know whats going on")


def assertWinAlmostEqual(win1, win2, parameters):
    """Test whether two windows are almost equal"""

    # Go through all keys in the Window
    for parameter in parameters:

        # Get Value from Window 1
        try:
            val1 = getattr(win1, parameter)
        except KeyError:
            raise AssertionError('Missing field')

        # Get Value from Window 2
        try:
            val2 = getattr(win2, parameter)
        except KeyError:
            raise AssertionError('Missing field')

        # The below things are basically just to satisfy the possible
        # window

        if type(val1) is dict and type(val2) is dict:
            assertDictAlmostEqual(val1, val2)

        elif type(val1) is list and type(val2) is list:
            assertListAlmostEqual(val1, val2)

        # Check if almost equal of both values
        elif is_numeric(val1) and is_numeric(val2):
            np.testing.assert_almost_equal(val1, val2, decimal=5)

        elif type(val1) is UTCDateTime and type(val2) is UTCDateTime:
            assertUTCDateTimeEqual(val1, val2)

        elif type(val1) is str and type(val2) is str:
            assert val1 == val2

        elif type(val1) is not type(val2):
            if is_numeric(val1) and is_numeric(val2):
                np.testing.assert_almost_equal(val1, val2, decimal=5)
            else:
                raise AssertionError("Too fields are not equal.")

        elif val1 is val2 is None:
            pass

        else:
            raise AssertionError("Window Dont know whats going on")


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


def reset_matplotlib():
    """
    Reset matplotlib to a common default.
    """
    # Set all default values.
    mpl.rcdefaults()
    # Force agg backend.
    plt.switch_backend('agg')
    # These settings must be hardcoded for running the comparision tests and
    # are not necessarily the default values.
    mpl.rcParams['font.family'] = 'Bitstream Vera Sans'
    mpl.rcParams['text.hinting'] = False
    # Not available for all matplotlib versions.
    try:
        mpl.rcParams['text.hinting_factor'] = 8
    except KeyError:
        pass
    import locale
    locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))


# Most generic way to get the data folder path.
TESTBASE_DIR = _upper_level(
    os.path.abspath(inspect.getfile(inspect.currentframe())), 4)
DATA_DIR = os.path.join(TESTBASE_DIR, "tests", "data")

obsfile = os.path.join(DATA_DIR, "proc", "IU.KBL.obs.proc.mseed")
synfile = os.path.join(DATA_DIR, "proc", "IU.KBL.syn.proc.mseed")
staxml = os.path.join(DATA_DIR, "stationxml", "IU.KBL.xml")
quakeml = os.path.join(DATA_DIR, "quakeml", "C201009031635A.xml")


def test_update_user_levels():
    obs_tr = read(obsfile)[0]
    syn_tr = read(synfile)[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    user_module = "gcmt3d.window.tests.user_module_example"
    config = win.update_user_levels(user_module, config, inv, cat,
                                    obs_tr, syn_tr)

    npts = obs_tr.stats.npts
    assert isinstance(config.stalta_waterlevel, np.ndarray)
    assert len(config.stalta_waterlevel) == npts
    assert isinstance(config.tshift_acceptance_level, np.ndarray)
    assert len(config.tshift_acceptance_level) == npts
    assert isinstance(config.dlna_acceptance_level, np.ndarray)
    assert len(config.dlna_acceptance_level) == npts
    assert isinstance(config.cc_acceptance_level, np.ndarray)
    assert len(config.cc_acceptance_level) == npts
    assert isinstance(config.s2n_limit, np.ndarray)
    assert len(config.s2n_limit) == npts


def test_update_user_levels_raise():
    user_module = "gcmt3d.window.tests.which_does_not_make_sense"
    with pytest.raises(Exception) as errmsg:
        win.update_user_levels(user_module, None, None, None,
                               None, None)

    assert "Could not import the user_function module" in str(errmsg)

    user_module = "gcmt3d.window.io"
    with pytest.raises(Exception) as errmsg:
        win.update_user_levels(user_module, None, None, None,
                               None, None)
    assert "Given user module does not have a generate_user_levels method" \
        in str(errmsg)


def test_window_on_trace():
    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, _verbose=False,
                                  figure_mode=False)

    assert len(windows) == 5

    winfile_bm = os.path.join(DATA_DIR, "window",
                              "IU.KBL..BHR.window.json")

    parameters = ["left", "right", "center", "time_of_first_sample",
                  "max_cc_value", "cc_shift", "dlnA", "dt", "min_period",
                  "channel_id", "phase_arrivals", "weight_function"]

    with open(winfile_bm) as fh:
        windows_json = json.load(fh)
        for _win, _win_json_bm in zip(windows, windows_json):
            _win_bm = Window._load_from_json_content(_win_json_bm)

        assertWinAlmostEqual(_win, _win_bm, parameters)


def test_window_on_trace_user_levels():
    obs_tr = read(obsfile)[0]
    syn_tr = read(synfile)[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)
    user_module = "gcmt3d.window.tests.user_module_example"

    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, user_module=user_module,
                                  _verbose=False,
                                  figure_mode=False)
    assert len(windows) == 4


def test_window_on_trace_with_none_user_levels():
    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    windows = win.window_on_trace(obs_tr, syn_tr, config, station=inv,
                                  event=cat, user_module="None",
                                  _verbose=False, figure_mode=False)

    winfile_bm = os.path.join(DATA_DIR, "window",
                              "IU.KBL..BHR.window.json")

    parameters = ["left", "right", "center", "time_of_first_sample",
                  "max_cc_value", "cc_shift", "dlnA", "dt", "min_period",
                  "channel_id", "phase_arrivals", "weight_function"]

    with open(winfile_bm) as fh:
        windows_json = json.load(fh)
    for _win, _win_json_bm in zip(windows, windows_json):
        _win_bm = Window._load_from_json_content(_win_json_bm)

        assertWinAlmostEqual(_win, _win_bm, parameters)


def test_window_on_stream():
    obs_tr = read(obsfile)
    syn_tr = read(synfile)

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)
    config_dict = {"Z": config, "R": config, "T": config}

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    windows = win.window_on_stream(obs_tr, syn_tr, config_dict, station=inv,
                                   event=cat, _verbose=False,
                                   figure_mode=False)

    assert len(windows) == 3
    nwins = dict((_w, len(windows[_w])) for _w in windows)
    assert nwins == {"IU.KBL..BHR": 5, "IU.KBL..BHZ": 2, "IU.KBL..BHT": 4}


def test_window_on_stream_user_levels():
    obs_tr = read(obsfile)
    syn_tr = read(synfile)

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)
    config_dict = {"Z": config, "R": config, "T": config}

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    _mod = "gcmt3d.window.tests.user_module_example"
    user_modules = {"BHZ": _mod, "BHR": _mod, "BHT": _mod}

    windows = win.window_on_stream(obs_tr, syn_tr, config_dict, station=inv,
                                   event=cat, user_modules=user_modules,
                                   _verbose=False,
                                   figure_mode=False)

    assert len(windows) == 3
    nwins = dict((_w, len(windows[_w])) for _w in windows)
    assert nwins == {"IU.KBL..BHR": 5, "IU.KBL..BHZ": 2, "IU.KBL..BHT": 4}


def test_plot_window_figure(tmpdir):
    reset_matplotlib()

    obs_tr = read(obsfile).select(channel="*R")[0]
    syn_tr = read(synfile).select(channel="*R")[0]

    config_file = os.path.join(DATA_DIR, "window", "27_60.BHZ.config.yaml")
    config = wio.load_window_config_yaml(config_file)

    cat = read_events(quakeml)
    inv = read_inventory(staxml)

    ws = WindowSelector(obs_tr, syn_tr, config, event=cat, station=inv)
    windows = ws.select_windows()

    assert len(windows) > 0

    win.plot_window_figure(str(tmpdir), obs_tr.id, ws, True,
                           figure_format="png")
