Loading and Building Models
===========================

The main entry point is :func:`alpsqutip.alpsmodels.model_from_alps_xml`, which reads an ALPS XML file and returns a `SystemDescriptor` object.

.. autofunction:: alpsqutip.alpsmodels.model_from_alps_xml

Once loaded, you can build system objects via:

.. autofunction:: alpsqutip.model.build_system

.. autoclass:: alpsqutip.model.SystemDescriptor
   :members:
   :inherited-members:
