Quickstart
==========

This page shows how to load a model, build a system, and run a simple simulation.

.. code-block:: python

    from alpsqutip.alpsmodels import model_from_alps_xml
    system = model_from_alps_xml("my_model.xml")

    # Build quantum operators
    H = system.build_operator("hamiltonian")
    Sz = system.build_operator("Sz", site=0)

    # Simulate time evolution
    from alpsqutip.evolution import evolve
    result = evolve(H, initial_state=...)

    # Compute Gibbs state (mean-field)
    from alpsqutip.operators.states.meanfield import compute_gibbs_state
    gibbs = compute_gibbs_state(H, beta=1.0)
