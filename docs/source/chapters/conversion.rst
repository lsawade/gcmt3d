Conversion of the Synthetic and Observed Data to ASDF
-----------------------------------------------------

For further processing and inversion, the ``ASDF`` data format is used to
simplify those steps. To convert the ``SAC`` or ``mseed`` data to ``ASDF``
format, the ``ConvertASDF`` ``class``, which is included in ``GCMT3D``, is used.

The ConvertASDF ``ConvertASDF`` ``class`` takes in one main argument, which
is the path to a ``Path`` file. The ``Path`` file contains the paths to either
the ``SAC`` files to be converted or the directory that contains the
``SAC`` files, the path to the earthquake ``XML`` file, as well as the path
to the ``StationXML`` file. Both path files for the synthetic and the
observed seismic data are created automatically when the database entry is
created. The usual content of the observed path file looks as follows:

.. code-block:: yaml

    waveform_dir: path/to/waveform_dir
    filetype: SAC
    quakeml_file: path/to/quake.xml
    staxml_files: path/to/station.xml
    tag: obsd
    output_file: path/to/raw_observed.h5

And the following snippet is a standard content of the path file for the
synthetic data.

.. code-block:: yaml

    waveform_dir: path/to/waveform_dir
    filetype: SAC
    quakeml_file: path/to/quake.xml
    tag: synt
    output_file: path/to/raw_synt.h5

Note that synthetic data does not require a ``StationXML`` since the SAC
header contains all necessary information and no response info is needed.


processing path files
inversion dictionaries


