"""
This contains a class that controls how the waveforms are processed and how
they are weighted in the CMT inversion

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.orcopyleft/gpl.html)

Last Update: June 2019

"""

import numpy as np


def filter_scaling(startcorners, startmag, endcorners, endmag, newmag):
    """ This function is taking a set of corner frequencies or periods that
    should be in a sorted list or array and create taper scaling for the
    Global CMT workflow.

    Args:
        startcorners: starting filters
        startmag: magnitude of starting filter
        endcorners: new end boundaries
        endmag: magnitude of ending filter
        newmag: magnitude to query in between startmag and end mag

    Returns:
        list of corners

    .. rubric:: Explanation

    The Global CMT workflow changes the tapers for mantle waves
    linearly depending on the magnitude from magnitude 7.0 to 8.0. This is
    a Python implementation for exactly that function. It's a simple scaling,
    much scaling of an integral in elementary Calculus. It first finds the new
    bounds that depend on the change in starting and end magnitude and then the
    corresponding corner frequencies that bound the flat part, which is found
    by scaling the bounds.


    Examples:

        >>> startcorners = [125, 150, 300, 350]
        >>> endcorners = [200, 400]
        >>> startmag = 7.0
        >>> endmag = 8.0
        >>> newmag = 7.5
        >>> newcorners = filter_scaling(startcorners, startmag, endcorners,
        >>>                             endmag, newmag)
        >>> print(newcorners)
        [162.5         186.11111111  327.77777778  375.]

    """

    lower = np.min(startcorners) \
        + (np.min(endcorners) - np.min(startcorners)) \
        / (endmag - startmag) * (newmag - startmag)
    upper = np.max(startcorners) \
        + (np.max(endcorners) - np.max(startcorners)) \
        / (endmag - startmag) * (newmag - startmag)

    scaled_filter = (startcorners - np.min(startcorners)) \
        * (upper - lower) \
        / (np.max(startcorners) - np.min(startcorners)) + lower

    return scaled_filter.tolist()


class ProcessParams(object):

    def __init__(self, mw: float, depth: float):
        """Given a cmtsource this class determines processing scheme for the
        CMT inversion based on the 2012 Ekström article on the updated CMT
        inversion workflow.

        This class could have easily been a function, but the structure of the
        class just makes it so much simpler to modify and understand.

        Args:
            mw: Moment Magnitude
            depth: Depth


        .. rubric:: Example output dictionary

        .. code::

            {'body': {'filter': [150.0, 100.0, 60.0, 50.0],
                      'weight': 1.0},
             'mantle': {"filter": [350.0, 300.0, 150.0, 125.0],
                        'weight': 1.0},
             'surface': {'filter': [150.0, 100.0, 60.0, 50.0],
                         'weight': 1.0}

        """
        # Magnitude
        self.mw = mw
        self.depth = depth

        # Filter
        self.bodywave_filter = None
        self.surfacewave_filter = None
        self.mantlewave_filter = None

        # Weights
        self.bodywave_weight = 1.0
        self.surfacewave_weight = 1.0
        self.mantlewave_weight = 1.0

        # Record length
        self.body_relative_endtime = 1.0 * 3600.0
        self.surface_relative_endtime = 2.0 * 3600.0
        self.mantle_relative_endtime = 4.5 * 3600.0

        # Whether to use velocity as a measurement.
        self.velocity = bool(mw < 5.5)

    def determine_all(self):
        """Main class method. Would have called it a __call__
        but wanted to make it more clear."""

        # Determine filters
        self.determine_bodywave_filter()
        self.determine_surfacewave_filter()
        self.determine_mantlewave_filter()

        # Determine weights
        self.determine_bodywave_weight()
        self.determine_surfacewave_weight()
        self.determine_mantlewave_weight()

        # Create dictionary that contains only the necessary entries
        # to clarify which ones are actual necessary.
        outdict = dict()

        if self.bodywave_weight is not None \
                and self.bodywave_weight != 0.0:
            outdict["body"] = {"weight": float(self.bodywave_weight),
                               "filter": self.bodywave_filter,
                               "relative_endtime":
                                   self.body_relative_endtime,
                               "velocity": self.velocity}

        if self.surfacewave_weight is not None \
                and self.surfacewave_weight != 0.0:
            outdict["surface"] = {"weight": float(self.surfacewave_weight),
                                  "filter": self.surfacewave_filter,
                                  "relative_endtime":
                                      self.surface_relative_endtime,
                                  "velocity": self.velocity}

        if self.mantlewave_weight is not None \
                and self.mantlewave_weight != 0.0:
            outdict["mantle"] = {"weight": float(self.mantlewave_weight),
                                 "filter": self.mantlewave_filter,
                                 "relative_endtime":
                                     self.mantle_relative_endtime,
                                 "velocity": False}

        return outdict

    def determine_bodywave_filter(self):
        """Here I didn't implement the velocity filter,
        I'm not quite sure what it means in the paper."""
        if self.mw < 6.5:
            self.bodywave_filter = [150.0, 100.0, 50.0, 40.0]
        elif 6.5 <= self.mw <= 7.5:
            self.bodywave_filter = [150.0, 100.0, 60.0, 50.0]
        else:
            self.bodywave_filter = None

    def determine_surfacewave_filter(self):
        """Less that 5.5 should be filter in the velocity spectrum. Same
        as the bodywave filter."""
        if self.mw <= 7.5:
            self.surfacewave_filter = [150.0, 100.0, 60.0, 50.0]
        else:
            self.surfacewave_filter = None

    def determine_mantlewave_filter(self):
        """Uses the scaled corners from magnitude 7.0 to 8.0"""
        if self.mw < 5.5:
            self.mantlewave_filter = None
        elif 5.5 <= self.mw <= 7.0:
            self.mantlewave_filter = [350.0, 300.0, 150.0, 125.0]
        # For events larger than magnitude 7.0s
        elif 7.0 <= self.mw and self.mw <= 8.0:

            # Config values
            startcorners = [125.0, 150.0, 300.0, 350.0]
            endcorners = [200.0, 400.0]
            startmag = 7.0
            endmag = 8.0

            # Compute new corners
            newcorners = filter_scaling(startcorners, startmag,
                                        endcorners, endmag, self.mw)

            # Since the scaling works with a sorted list of periods.
            # We reverse the order so that largest period (lowest f) is the
            # first element of the corners filter
            self.mantlewave_filter = newcorners[::-1]

        else:
            # Config values
            startcorners = [125.0, 150.0, 300.0, 350.0]
            endcorners = [200.0, 400.0]
            startmag = 7.0
            endmag = 8.0

            # Compute new corners
            newcorners = filter_scaling(startcorners, startmag,
                                        endcorners, endmag, 8.0)

            self.mantlewave_filter = newcorners[::-1]

    def determine_bodywave_weight(self):
        if self.mw > 7.5:
            self.bodywave_weight = None
        elif self.mw < 6.5:
            self.bodywave_weight = 1.0
        else:
            self.bodywave_weight = (7.5 - self.mw) / (7.5 - 6.5)

    def determine_surfacewave_weight(self):
        """No weight for events deeper that 300km."""
        if self.mw > 7.5:
            self.surfacewave_weight = None
        elif self.mw <= 6.5:
            self.surfacewave_weight = 1.0
        else:
            self.surfacewave_weight = (7.5 - self.mw) / (7.5 - 6.5)

        if self.depth > 300000.0:
            self.surfacewave_weight = None

    def determine_mantlewave_weight(self):
        if self.mw > 6.5:
            self.mantlewave_weight = 1.0
        elif self.mw < 5.5:
            self.mantlewave_weight = None
        else:
            self.mantlewave_weight = (self.mw - 5.5)/(6.5 - 5.5)
