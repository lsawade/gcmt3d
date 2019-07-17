Create a Database Entry
-----------------------

The first step of the workflow for any earthquake is the creation of a
database entry. This means that an initial directory for the earthquake is
created in the database directory defined in the ``DatabaseParameters.yml``
parameter file. The structure of the Database looks as follows:

.. code-block::

    database
    ├── eq_<eq_ID_1>
    │   ├── CMT_SIMs/
    │   │   ├── CMT/
    │   │   │    ├── DATA/
    │   │   │    ├── DATABASES_MPI/
    │   │   │    ├── OUTPUTFILES/
    │   │   │    └── bin/        # symbolic link --> SPECFEM_DIR/bin/
    │   │   ├── CMT_rr/
    │   │   │    ├── DATA/
    │   │   │    :
    │   │   ├── CMT_pp/
    │   │   ├── CMT_rp/
    │   │   ├── CMT_rt/
    │   │   ├── CMT_tp/
    │   │   ├── CMT_tt/
    │   │   ├── CMT_depth/
    │   │   ├── CMT_lat/
    │   │   └── CMT_lon/
    │   ├── eq_9703873.cmt
    │   ├── station_data/
    │   │   ├──STATIONS
    │   │   ├──station.xml
    │   │   :
    │   ├── window_data/
    │   │   ├──
    │   │   :
    │   ├── inversion_results/
    │   │   ├──new_cmt.cmt
    │   │   ├──inv_summary.pdf-->
    │   │   :
    │   └── seismograms/
    │       ├──obs/
    │       │   ├──observed.mseed
    │       │   ├──observed.h5
    │       │   ├──processed_observed.<lf>_<hf>.h5
    │       │   └──observed.h5
    │       └──syn/
    │
    │
    ├── eq_<eq_ID_2>/
    │   ├── CMT_SIMs/
    │   :
    │
    ├── eq_<eq_ID_3>/
    :



As seen in the directory tree, the each earthquake has its own entry and
directory into which the corresponding observed data are downloaded,
simulation data are saved, window data