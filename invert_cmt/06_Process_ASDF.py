#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

This script writes specfem sources into the respective simulation directories.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)

"""


from gcmt3d.asdf.utils import smart_read_yaml, is_mpi_env
from gcmt3d.asdf.convert import ConvertASDF
import os
import argparse

def main(cmt_filename):

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "params")
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    DB_params = smart_read_yaml(databaseparam_path,
                                mpi_mode=is_mpi_env())

    # File and directory
    cmt_dir = os.path.dirname(cmt_filename)
    sim_dir = os.path.join(cmt_dir, "CMT_SIMs")

    attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
            "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

    for _i, at in enumerate(attr[:DB_params["npar"]+1]):

        # Path file
        path_file = os.path.join(sim_dir, at, at + ".yml")

        converter = ConvertASDF(path_file, verbose=DB_params["verbose"],
                                status_bar=DB_params["verbose"])
        converter.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    main(args.filename)
