Examples
========

.. code-block:: python

   # Load model and build system
   from alpsqutip.alpsmodels import model_from_alps_xml
   system = model_from_alps_xml("example.xml")

   # Build Hamiltonian and evolve
   H = system.build_operator("hamiltonian")
   from alpsqutip.evolution import evolve
   result = evolve(H, initial_state=...)

   # Compute mean-field Gibbs state
   from alpsqutip.operators.states.meanfield import compute_gibbs_state
   gibbs = compute_gibbs_state(H, beta=1.0)

# Add more end-to-end or advanced examples as needed.
