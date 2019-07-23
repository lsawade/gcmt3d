Creating Synthetics
-------------------

The standard procedure here for the creation of synthetic
data is running ``specfem3d_globe`` with the perturbed sources.
``specfem3d_globe`` is run using
either the ``RunSimulation`` ``class`` in
``GCMT3D`` or a simple setup through an EnTK pipeline using the directories
inside the database. Since ``specfem`` had to
be configured and compiled in the very beginning before even creating the
entries in the earthquake directories, that part is skipped here.


Cleaning the simulation directories
-----------------------------------

Each simulation directory takes quite the chunk of space on the disk. To
ensure that the database does not take too much space on a cluster or
hard disk, the ``clean_up`` function of the ``Run_simulation`` ``class`` will
clean up the simulation directories and only save the necessary parts.

