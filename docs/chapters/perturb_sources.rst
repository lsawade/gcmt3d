Perturbation of Specfem Source
------------------------------

Thye next step in the workflow is to create the sources for the inversion
workflow. Meaning, for each parameter to invert for we need to perturb the
source and simulate an earthquake using a perturbed source to compute the
Fréchet derivative afterwards. The creation of the specfem_sources is done
using the ``GCMT3D``'s ``SpecfemSources`` class. The class will create both
the ``CMTSOLUTION`` needed for the specfem simulations as well as the
``QuakeML`` needed for the inversion/ASDF file creation later on.

The perturbation parameters can be found in the ``InversionParams.yml``
parameter file.

.. code-block:: yaml

    # Weight configuration for pycmt3d
    weight_config:
      normalize_by_energy: False
      normalize_by_category: False
      comp_weight:
        Z: 1.0
        R: 1.0
        T: 1.0
      love_dist_weight: 1.0
      pnl_dist_weight: 1.0
      rayleigh_dist_weight: 1.0
      azi_exp_idx: 0.5

    #  configuration for pycmt3d
    config:
      # Perturbation of the location (deg)
      dlocation: 0.5

      # Perturbation of the depth (km)
      ddepth: 0.5

      # Perturbation of the depth (Nm)
      dmoment: 1.0e22

      # ???
      zero_trace: True
      # ??? Weight setup?
      weight_data: True

      # ???
      station_correction: True

      # Bootstrap Parameters
      bootstrap: True
      bootstrap_repeat: 20
      bootstrap_subset_ratio: 0.4








