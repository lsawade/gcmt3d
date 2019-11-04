GCMT3D API
==========

This contains the function set that is included in GCMT3D. Some of the
modules originate in other packages such as ``pytomo3d`` and ``pypaw``.
Those  modules have been updated and integrated into GCMT3D.

ASDF (pypaw)
------------

The functions in this section originally belonged to pypaw which partly
relied on pytomo. To make this package as simple as possible I extracted the
only functions necessary for the source inversion.

Conversion
++++++++++

.. automodule:: gcmt3d.asdf.convert
    :members:
    :undoc-members:
    :show-inheritance:

Processing Abstract Base Class
++++++++++++++++++++++++++++++

.. automodule:: gcmt3d.asdf.procbase
    :members:

Seismic Trace Processing Class
++++++++++++++++++++++++++++++

.. automodule:: gcmt3d.asdf.process
    :members:
    :undoc-members:
    :show-inheritance:

Stations Info Utilities
+++++++++++++++++++++++

.. automodule:: gcmt3d.asdf.stations
    :members:
    :undoc-members:
    :show-inheritance:

General ASDF utilities
++++++++++++++++++++++

.. automodule:: gcmt3d.asdf.utils
    :members:
    :undoc-members:
    :show-inheritance:

ASDF Windowing
++++++++++++++

.. automodule:: gcmt3d.asdf.window
    :members:
    :undoc-members:
    :show-inheritance:


Data
----

Download
++++++++


Request seismic data
********************

.. automodule:: gcmt3d.data.download.requestBuilder
    :members:
    :undoc-members:
    :show-inheritance:

Management
++++++++++

Create database entry/database skeleton
***************************************

.. automodule:: gcmt3d.data.management.skeleton
    :members:
    :undoc-members:
    :show-inheritance:

Edit Specfem sources
********************

.. automodule:: gcmt3d.data.management.specfem_sources
    :members:
    :undoc-members:
    :show-inheritance:

Create Path files to process asdf files
***************************************

.. automodule:: gcmt3d.data.management.create_process_paths
    :members:
    :undoc-members:
    :show-inheritance:

Create inversion dictionaries
*****************************

.. automodule:: gcmt3d.data.management.inversion_dicts
    :members:
    :undoc-members:
    :show-inheritance:


Run Specfem
-----------

.. automodule:: gcmt3d.runSF3D.runSF3D
    :members:
    :undoc-members:
    :show-inheritance:


Signal Processing
-----------------

This part is also largely taken from pytomo3d, but it has been updated to
Python 3 such that there is support in the future.

Compare traces
++++++++++++++

.. automodule:: gcmt3d.signal.compare_trace
    :members:
    :undoc-members:
    :show-inheritance:

Function to process streams
+++++++++++++++++++++++++++

.. automodule:: gcmt3d.signal.process
    :members:
    :undoc-members:
    :show-inheritance:

Rotate seismograms
++++++++++++++++++

.. automodule:: gcmt3d.signal.rotate
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: gcmt3d.signal.rotate_utils
    :members:
    :undoc-members:
    :show-inheritance:


Utilities
---------

.. automodule:: gcmt3d.utils.download
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: gcmt3d.utils.io
    :members:
    :undoc-members:
    :show-inheritance:

Windowing of traces
-------------------

The following modules are all concerned with the processing of both synthetic
and observed data and their similarities.

Windowing of the data
+++++++++++++++++++++

.. automodule:: gcmt3d.window.window
    :members:
    :undoc-members:
    :show-inheritance:

Weighting the windows
+++++++++++++++++++++

.. automodule:: gcmt3d.window.window_weights
    :members:
    :undoc-members:
    :show-inheritance:

Filtering the windows
+++++++++++++++++++++

.. automodule:: gcmt3d.window.filter_windows
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
+++++++++

.. automodule:: gcmt3d.window.utils
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: gcmt3d.window.io
    :members:
    :undoc-members:
    :show-inheritance:

Source ``class``
----------------

.. automodule:: gcmt3d.source
    :members:
    :undoc-members:
    :show-inheritance:


stage del
stage 1500dbar (50dbar) 1000mn (1000mn)
stage 1500dbar (50dbar) 5857mn (6857mn)
stage store