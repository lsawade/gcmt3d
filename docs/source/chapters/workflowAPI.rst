EnTK Workflow API
=================

This contains the single steps in the workflow of GCMT3D as given in the
workflow module.



Standard Workflow components
============================

Earthquake ID reader
--------------------

.. automodule:: gcmt3d.workflow._get_eq_dir
    :members:

Database Entry Creation
-----------------------

.. automodule:: gcmt3d.workflow.create_database_entry
    :members:

Requesting Seismic Data
-----------------------

.. automodule:: gcmt3d.workflow.data_request
    :members:

.. automodule:: gcmt3d.workflow.data_request2
    :members:

.. automodule:: gcmt3d.workflow.data_request3
    :members:

Writing the Sources
-------------------

.. automodule:: gcmt3d.workflow.generate_sources
    :members:

Create Path Files
-----------------

.. automodule:: gcmt3d.workflow.prepare_path_files
    :members:

Create Inversion Dictionaries
-----------------------------

.. automodule:: gcmt3d.workflow.create_inversion_dictionaries
    :members:

Inversion
---------

.. automodule:: gcmt3d.workflow.tensor_inversion
    :members:

Grid Search
-----------

.. automodule:: gcmt3d.workflow.tensor_rise
    :members:
