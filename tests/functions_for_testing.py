"""

A few functions that come in handy when comparing dictionaries

:copyright:
    Lucas Sawade (lsawade@princeton.edu) (2019)

:license:
    :license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
"""

from obspy import UTCDateTime
import numpy as np
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
        print(key, dict1[key])
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
            raise AssertionError("Window Dont know whats going on")
