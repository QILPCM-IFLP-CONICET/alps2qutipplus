"""
Basic unit test.
"""

from alpsqutip.model import build_spin_chain


def test_system_operations():
    system = build_spin_chain(10)
    sites = tuple(system.sites)
    system_a = system.subsystem(frozenset(sites[:4]))
    system_b = system.subsystem(frozenset(sites[2:6]))
    system_c = system.subsystem(frozenset(sites[7:]))

    print(system_a.name, "with", system_a.sites.keys())
    print(system_b.name, "with", system_b.sites.keys())
    print(system_c.name, "with", system_c.sites.keys())
    cases = {
        "system_ab": system_a * system_b,
        "system_ac": system_a * system_c,
        "system_bc": system_b * system_c,
    }
    for name, case in cases.items():
        print("subsystem", name, "is", case.name, "with", case.sites.keys())

    assert (
        len(cases["system_ab"].sites)
        == len(set(system_a.sites).union(set(system_b.sites)))
        < 8
    )
    assert (
        len(cases["system_ac"].sites)
        == len(set(system_a.sites).union(set(system_c.sites)))
        < 8
    )
    assert (
        len(cases["system_bc"].sites)
        == len(set(system_b.sites).union(set(system_c.sites)))
        < 8
    )
