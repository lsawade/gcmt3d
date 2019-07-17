Create a Database Entry
-----------------------

The first step of the workflow for any earthquake is the creation of a
database entry. This means that an initial directory for the earthquake is
created in the database directory defined in the ``DatabaseParameters.yml``
parameter file. The structure of the Database looks as follows:

.. code-block::

    database
    ├──eq_<eq_ID_1>
    │   ├──CMT_SIMs/
    │   │   ├──CMT/
    │   │   │   ├──DATA/
    │   │   │   ├──DATABASES_MPI/
    │   │   │   ├──OUTPUTFILES/
    │   │   │   │   ├──CMT.yml # ASDF conversion path file
    │   │   │   │   :
    │   │   │   └──bin/  # symbolic link --> SPECFEM_DIR/bin/
    │   │   ├──CMT_rr/
    │   │   │   ├──DATA/
    │   │   │   :
    │   │   ├──CMT_pp/
    │   │   ├──CMT_rp/
    │   │   ├──CMT_rt/
    │   │   ├──CMT_tp/
    │   │   ├──CMT_tt/
    │   │   ├──CMT_depth/
    │   │   ├──CMT_lat/
    │   │   └──CMT_lon/
    │   ├──eq_9703873.cmt  # CMTSOLUTION
    │   ├──eq_9703873.xml  # QuakeML
    │   ├──station_data/
    │   │   ├──STATIONS
    │   │   ├──station.xml
    │   │   :
    │   ├──window_data/
    │   │   ├──window_paths/
    │   │       ├──windows.<lT1>_<hT1>f#<wave_type>.yml  # lT and hT are low and
    │   │       │                                        # high period,
    │   │       │                                        # respectively
    │   │       ├──windows.<lT2>_<hT2>f#<wave_type>.yml  # wave_type refers to
    │   │       │                                        # body or surface wave
    │   │       :
    │   │
    │   ├──inversion/
    │   │   ├──inversion_dicts/
    │   │   │   ├──inversion_file_dict.<lT1>_<hT1>.yml
    │   │   └──inversion_output/
    │   │       ├──inv_summary.pdf
    │   │
    │   └──seismograms/
    │       ├──obs/
    │       │   ├──observed.mseed
    │       │   └──observed.h5
    │       ├──syn/
    │       │   ├──CMT.h5
    │       │   ├──CMT_rr.h5
    │       │   :
    │       │   └──CMT_lon.h5
    │       ├──process_paths/
    │       │   ├──process_observed.<lT1>_<hT1>.yml
    │       │   ├──process_synthetic_CMT.<lT1>_<hT1>.yml
    │       │   ├──process_synthetic_CMT_rr.<lT1>_<hfT>.yml
    │       │   :
    │       │   ├──process_synthetic_CMT_lon.<lT1>_<hT1>.yml
    │       │   │
    │       │   ├──process_observed.<lT2>_<hT2>.yml
    │       │   ├──process_synthetic_CMT.<lT2>_<hT2>.yml
    │       │   ├──process_synthetic_CMT_rr.<lT2>_<hT2>.yml
    │       │   :
    │       │   ├──process_synthetic_CMT_lon.<lT2>_<hT2>.yml
    │       │   :
    │       │
    │       └──processed_seismograms/
    │           ├──processed_observed.<lT1>_<hT1>.h5
    │           ├──processed_synthetic_CMT.<lT1>_<hT1>.h5
    │           ├──processed_synthetic_CMT_rr.<lT1>_<hT1>.h5
    │           :
    │           ├──processed_synthetic_CMT_lon.<lT1>_<hT1>.h5
    │           │
    │           ├──processed_observed.<lT2>_<hT2>.h5
    │           ├──processed_synthetic_CMT.<lT2>_<hT2>.h5
    │           ├──processed_synthetic_CMT_rr.<lT2>_<hT2>.h5
    │           :
    │           ├──processed_synthetic_CMT_lon.<lT2>_<hT2>.h5
    │           :
    │
    ├──eq_<eq_ID_2>/
    │   ├──CMT_SIMs/
    │   :
    │
    ├──eq_<eq_ID_3>/
    :



As seen in the directory tree, each earthquake has its own entry and
directory. Into each directory corresponding observed data are downloaded,
simulation data are saved, processed and saved, windowed, and inverted. Note
that the tree above is the result of all steps in the workflow (except the
cleaning step since the cleaning step will clean out the simulation directory).

The ``DatabaseSkeleton`` class will, as the name suggests, simply create a
skeleton for the above database tree, and the subsequent steps of the
workflow will fill it.




