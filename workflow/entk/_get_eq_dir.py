#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple function to get the earthquake directory from the database location
and the CMTSolution file
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from gcmt3d.source import CMTSource


def get_eq_entry_path(database_dir, cmtfilename):
    """ Gets database entry directory for a specific earthquake.

    :param database_dir:
    :param cmtfilename:
    :return: cmtdir
    """

    # Get CMTSource
    cmt = CMTSource.from_CMTSOLUTION_file(cmtfilename)

    # Get earthquake id
    eq_id = cmt.eventname

    # Earthquake directory
    eq_dir = os.path.join(database_dir, "eq_" + eq_id)

    return str(eq_dir), eq_id
