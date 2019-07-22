Request Data
------------

The inversion for a moment tensor requires, as so many problems in
seismology, observed data. In a routine workflow it is important that the
download of the observed data is also automated. Here, we use the
``DataRequest`` class of the GCMT3D package to do that. The request itself
require some parameters that are given in the ``RequestParams.yml`` parameter
file. A sample of such a file is shown below

.. code-block:: yaml

    # seconds, 1h = 3600s
    duration: 7400

    # needs to be a list
    channels: ['BHN','BHE','BHZ']

    # location 00 is the primary location for most seismometers
    locations: ['00']

    # -60 seconds is 1 min before the earthquakes inverted cmt solution
    starttime_offset: -60

    # response format
    resp_format: "resp"

    # OPTIONAL
    stationnlist: None







