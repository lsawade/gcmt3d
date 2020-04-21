EnTK Workflow API
=================

This contains the single steps in the workflow of GCMT3D as given in the
workflow module.



Standard Workflow components
============================

Specfem Fixer
-------------

.. automodule:: workflow.invert_cmt.00_Fix_Specfem_And_Recompile
    :members:

Database Entry Creation
-----------------------

.. automodule:: workflow.invert_cmt.01_Create_Database_Entry
    :members:

Requesting Seismic Data
-----------------------

.. automodule:: workflow.invert_cmt.02_Request_Data
    :members:

Writing the Sources
-------------------

.. automodule:: workflow.invert_cmt.03_Write_Sources
    :members:

Specfem
-------

Run Simulations
+++++++++++++++

.. automodule:: workflow.invert_cmt.04a_Run_Specfem_Run
    :members:

Clean Up Data Base
++++++++++++++++++

.. automodule:: workflow.invert_cmt.04b_Run_Specfem_Database_Clean_Up
    :members:

Conversion to ASDF
------------------

.. automodule:: workflow.invert_cmt.05_Convert_Observed_to_ASDF
    :members:

Create Path Files
-----------------

.. automodule:: workflow.invert_cmt.06_Create_Path_Files
    :members:

Process Data
------------

.. automodule:: workflow.invert_cmt.07_Process_Traces
    :members:

Window Traces
-------------

.. automodule:: workflow.invert_cmt.08_Window_Traces
    :members:

Create Inversion Dictionaries
-----------------------------

.. automodule:: workflow.invert_cmt.09_Create_Inversion_Dictionary
    :members:

Inversion
---------

.. automodule:: workflow.invert_cmt.10_Inversion
    :members:


Specfem Extras
==============

Run Mesher
----------

.. automodule:: workflow.invert_cmt.21_Run_Mesher
    :members:

Run Solver
----------

.. automodule:: workflow.invert_cmt.22_Run_Solver
    :members:





























