Quickstart
==========

This quickstart guide shows how to get started with alps2qutipplus: building a quantum system, exploring its structure, manipulating operators, and simulating quantum evolution.

Prerequisites
-------------
You should have installed `alps2qutipplus`, `matplotlib`, `numpy`, and `qutip`.

Importing Required Libraries
----------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from alpsqutip import build_system, model_from_alps_xml
    # Optional: list_models_in_alps_xml, list_geometries_in_alps_xml, graph_from_alps_xml

Building a Simple Quantum System
-------------------------------

To create a default system (a spin-1/2 chain with four sites), use:

.. code-block:: python

    system = build_system()

The system is described by a ``SystemDescriptor`` object, which contains information about the model, its geometry, and parameters.

Visualizing the Lattice
-----------------------

You can access the underlying graph and draw it:

.. code-block:: python

    system.spec["graph"].draw(plt)
    plt.show()

Exploring Sites and Local Properties
------------------------------------

Each site is described in the ``system.sites`` dictionary:

.. code-block:: python

    print(system.sites.keys())  # Lists all site names

    # Explore the first site's properties:
    site = system.sites['1[0]']
    print("Dimension:", site["dimension"])
    print("Quantum numbers:", site["qn"])
    print("Operators:", tuple(site["operators"]))

Working with Operators
----------------------

Global operators (like the Hamiltonian and magnetization) are readily available:

.. code-block:: python

    H = system.global_operator("Hamiltonian")
    Sz = system.global_operator("Sz")
    print(H)
    print(Sz)

You can view the list of predefined global operators:

.. code-block:: python

    print(tuple(system.operators["global_operators"]))

Site-specific operators are also accessible:

.. code-block:: python

    sx0 = system.site_operator("Sx@1[0]")
    print(sx0)

Operators can be combined algebraically:

.. code-block:: python

    Hzeeman = -2 * Sz
    Htotal = (Hzeeman + H).simplify()
    print(Htotal)

Analyzing Operators
-------------------

You can compute eigenvalues, exponentiate, or take the trace of operators:

.. code-block:: python

    print(Htotal.eigenenergies())      # Spectrum
    print(Htotal.expm())               # Exponential
    print("Partition function:", (-Htotal).expm().tr())

Visualizing Operator Support
----------------------------

To see which sites an operator acts on, use:

.. code-block:: python

    from alpsqutip.utils import draw_operator
    fig, ax = plt.subplots()
    draw_operator(Htotal, ax)
    Htotal.system.spec["graph"].draw(ax)
    plt.show()

Qutip Integration and Time Evolution
------------------------------------

Operators can be converted to qutip objects and used in qutip solvers:

.. code-block:: python

    import qutip

    sx01 = system.site_operator("Sx@1[0]") + system.site_operator("Sx@1[1]")
    rho0 = sx01.expm()
    rho0 = rho0 / rho0.tr()
    ts = np.linspace(0, 10, 100)

    result = qutip.mesolve(
        H=Hzeeman.to_qutip(),
        rho0=rho0.to_qutip(),
        tlist=ts,
        e_ops=(sx01.to_qutip(),)
    )
    plt.plot(ts, result.expect[0], label="$H_{Zeeman}$")

    result = qutip.mesolve(
        H=H.to_qutip(),
        rho0=rho0.to_qutip(),
        tlist=ts,
        e_ops=(sx01.to_qutip(),)
    )
    plt.plot(ts, result.expect[0], label="$H_{exc}$")

    result = qutip.mesolve(
        H=Htotal.to_qutip(),
        rho0=rho0.to_qutip(),
        tlist=ts,
        e_ops=(sx01.to_qutip(),)
    )
    plt.plot(ts, result.expect[0], label="$H_{total}$")

    plt.legend()
    plt.xlabel("t")
    plt.ylabel(r"$\langle sx_1+sx_2\rangle$")
    plt.show()

Larger Systems
--------------

As long as you avoid explicit diagonalization for very large systems, you can define and manipulate larger quantum systems using the same methods.

.. code-block:: python

    large_system = build_system()  # Adjust parameters for larger systems as needed
    # ... proceed as above
