"""
Graphs and conversions from ALPS
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import rand

from alpsqutip.settings import LATTICE_LIB_FILE
from alpsqutip.utils import eval_expr, find_ref, next_name


def list_geometries_in_alps_xml(filename=LATTICE_LIB_FILE) -> Tuple[str, ...]:
    """
    List all the graph names in a lattice.xml ALPS file
    """
    result = []
    xmltree = ET.parse(filename)
    lattices = xmltree.getroot()

    for graph in lattices.findall("./GRAPH"):
        name = graph.attrib.get("name", "")
        if name:
            result.append(name)

    # Otherwise, try with a lattice
    for lat in lattices.findall("./LATTICEGRAPH"):
        name = lat.attrib.get("name", "")
        if name:
            result.append(name)
    return tuple(result)


def graph_from_alps_xml(
    filename=LATTICE_LIB_FILE, name="rectangular lattice", parms=None
):
    """
    Load from `filename` xml library a Graph or LatticeGraph of name
    `name`, using `parms` as parameters.
    """
    xmltree = ET.parse(filename)
    lattices = xmltree.getroot()
    if parms is None:
        parms = {}

    def build_lattice_positions(extents: List):
        """Build the indices for each lattice position."""
        cells = []
        dim = len(extents)
        curr_coords = (dim + 1) * [0]
        while True:
            cells.append(tuple(curr_coords[:-1]))
            d_int = 0
            curr_coords[d_int] += 1
            while d_int < dim and curr_coords[d_int] == extents[d_int]:
                curr_coords[d_int] = 0
                d_int += 1
                curr_coords[d_int] += 1

            if curr_coords[-1]:
                break
        return cells

    def complete_periodic_or_none(coords, extents, bcs) -> Optional[List[int]]:
        """
        If coords are out of the extents limits,
        check if the site has a periodic image.
        If it has, return it. Otherwise return None.
        """
        new_coords = [c for c in coords]
        for d_int, extent in enumerate(extents):
            if 0 <= coords[d_int] < extent:
                continue
            if bcs[d_int] == "periodic":
                new_coords[d_int] = new_coords[d_int] % extent
            else:
                return None
        return new_coords

    def process_coordinates(text: Optional[str] = None) -> Optional[list]:
        if text:
            coords_expr = [c for c in text.split(" ") if c != ""]
            try:
                return [eval_expr(c.strip(), parms) for c in coords_expr]
            except SyntaxError:
                logging.error(coords_expr, "are not valid coordinates.")
                raise
        return None

    def process_graph(node, parms):
        """Process a <GRAPH> node"""
        g_items = node.attrib
        parms = process_parms(node, parms)
        num_vertices = int(g_items["vertices"])
        vertices = {}
        edges = {}
        loops = {}
        default_vertex = {"type": "0", "coords": None}

        for vert in node.findall("./VERTEX"):
            v_items = process_vertex(vert, parms)
            v_name = v_items.pop("id", None) or next_name(vertices)
            vertices[v_name] = v_items

        while len(vertices) < num_vertices:
            vertices[next_name(vertices)] = default_vertex.copy()

        for edge in node.findall("./EDGE"):
            edge_items = edge.attrib
            edge_type = edge_items.pop("type", "0")
            edges.setdefault(edge_type, []).append(
                (
                    edge_items["source"],
                    edge_items["target"],
                )
            )

        for loop_descr in node.findall("./LOOP"):
            loop_dict = process_loop_graph(loop_descr, parms)
            loop_type = loop_dict.pop("type", "0")
            loops.setdefault(loop_type, []).append(loop_dict)

        return GraphDescriptor(
            name=name, nodes=vertices, edges=edges, loops=loops, parms=parms
        )

    def process_lattice(node, parms):
        """Process a <LATTICE> node"""
        parms = process_parms(node, parms)
        lattice = {"basis": [], "reciprocal_basis": []}
        lattice["dimension"] = int(node.attrib.get("dimension", 0))
        lattice["name"] = node.attrib.get("name", "")
        basis = lattice["basis"]
        reciprocal_basis = lattice["reciprocal_basis"]

        for b_desc in node.findall("./BASIS"):
            for vert in b_desc.findall("./VECTOR"):
                coords = process_coordinates(vert.text)
                basis.append(coords)

        for b_desc in node.findall("./RECIPROCALBASIS"):
            for vec in b_desc.findall("./VECTOR"):
                coords = [
                    eval_expr(coord.strip(), parms) for coord in vec.text.split(" ")
                ]
                reciprocal_basis.append(coords)

        return lattice

    def process_latticegraph(node, parms):
        """Process a <LATTICEGRAPH> node"""
        parms = process_parms(node, parms)
        unitcell = []
        vertices = {}
        edges = {}
        loops = {}

        for uc_desc in node.findall("./UNITCELL"):
            assert len(unitcell) == 0, "UNITCELL already defined."
            uc_desc = find_ref(uc_desc, lattices)
            unitcell = process_unitcell(uc_desc, parms)

        dim = unitcell["dimension"]

        for fl_desc in node.findall("./FINITELATTICE"):
            parms = process_parms(fl_desc, parms)
            extents = [1]
            bcs = ["open"]
            if dim:
                bcs = dim * bcs
                for bc_desc in fl_desc.findall("./BOUNDARY"):
                    bc_items = bc_desc.attrib
                    if "dimension" in bc_items:
                        c_dim = int(bc_items.get("dimension", 1)) - 1
                        bcs[c_dim] = bc_items.get("type", bcs[c_dim])
                    else:
                        bcs = dim * [bc_items.get("type", bcs[0])]

                extents = dim * [1]
                for ext in fl_desc.findall("./EXTENT"):
                    ext_items = ext.attrib
                    c_dim = int(ext_items.get("dimension", 1)) - 1
                    size = ext_items.get("size", extents[c_dim])
                    while size in parms:
                        size = parms[size]
                    extents[c_dim] = int(size)

            for lat_entry in fl_desc.findall("./LATTICE"):
                lattice = process_lattice(find_ref(lat_entry, lattices), parms)

        # Build a list of cells
        assert isinstance(
            bcs, (tuple, list)
        ), f"type of bcs is {type(bcs)}, not a list."

        cells = build_lattice_positions(extents)

        # Add the vertices associated to each cell
        cell_vertices = unitcell["vertices"]
        lattice_basis = [np.array(v) for v in lattice["basis"]]
        for cell in cells:
            for vertex, v_attr in cell_vertices.items():
                coords = v_attr.get("coords", None)
                if coords is not None:
                    v_attr = v_attr.copy()
                    v_attr["coords"] = (
                        sum(c * b for c, b in zip(cell, lattice_basis)) + coords
                    )

                vertices[f"{vertex}{list(cell)}"] = v_attr

        # Add inhomogeneous vertices
        for inhomogeneous in node.findall("./INHOMOGENEOUS"):
            for v_desc in inhomogeneous.findall("VERTEX"):
                v_items = process_vertex(v_desc, parms)
                v_name = v_items.get("name", None) or next_name(vertices, 1, "defect_")
                vertices[v_name] = v_items

        # Build loops
        cell_loops = unitcell["loops"]
        for loop_type, loop_desc_list in cell_loops.items():
            loop_list = []
            for cell in cells:
                # Loop over nodes in a loop
                for loop_descr in loop_desc_list:
                    loop_sites = []
                    skip = False
                    for node_descr in loop_descr["nodes"]:
                        node_vertex = node_descr["vertex"]
                        node_offset = node_descr["offset"]
                        src = list(u + d for u, d in zip(cell, node_offset))
                        src = complete_periodic_or_none(src, extents, bcs)
                        if src is None:
                            skip = True
                            break
                        loop_sites.append(f"{node_vertex}{src}")
                    # If the loop goes beyond the lattice,
                    # don't add it.
                if skip:
                    continue
                loop_list.append(loop_sites)
            loops[loop_type] = loop_list

        # Build edges
        cell_edges = unitcell["edges"]
        for e_type, bond_desc_list in cell_edges.items():
            bonds = []
            for cell in cells:
                for bnd_desc in bond_desc_list:
                    src_name = bnd_desc["src"]
                    src = list(u + d for u, d in zip(cell, bnd_desc["offset_src"]))
                    src = complete_periodic_or_none(src, extents, bcs)
                    tgt_name = bnd_desc["tgt"]
                    tgt = list(u + d for u, d in zip(cell, bnd_desc["offset_tgt"]))
                    tgt = complete_periodic_or_none(tgt, extents, bcs)

                    if src is None or tgt is None:
                        continue
                    new_bond = (
                        f"{src_name}{src}",
                        f"{tgt_name}{tgt}",
                    )
                    bonds.append(new_bond)
            edges[e_type] = bonds

        return GraphDescriptor(
            name=name,
            nodes=vertices,
            edges=edges,
            loops=loops,
            lattice=lattice,
            parms=parms,
        )

    def process_parms(node, parms):
        """Process the <PARAMETER>s nodes"""
        default_parms = {}
        for parameter in node.findall("./PARAMETER"):
            key_vals = parameter.attrib
            default_parms[key_vals["name"]] = key_vals["default"]
        default_parms.update(parms)
        return default_parms

    def process_edge_unitcell(e_desc, parms, dimension):
        e_items = e_desc.attrib
        e_type = e_items.get("type", "0")
        e_src = e_items.get("source", "0")
        e_tgt = e_items.get("target", "0")
        e_offset_src = ""
        e_offset_tgt = ""
        for src in e_desc.findall("./SOURCE"):
            src_items = src.attrib
            e_src = src_items.get("vertex", e_src)
            e_offset_src = src_items.get("offset", e_offset_src)
        for tgt in e_desc.findall("./TARGET"):
            tgt_items = tgt.attrib
            e_tgt = tgt_items.get("vertex", e_tgt)
            e_offset_tgt = tgt_items.get("offset", e_offset_tgt)

        e_offset_src = e_offset_src.split(" ") if e_offset_src else dimension * ["0"]
        e_offset_tgt = e_offset_tgt.split(" ") if e_offset_tgt else dimension * ["0"]
        e_offset_src = [int(c) for c in e_offset_src]
        e_offset_tgt = [int(c) for c in e_offset_tgt]

        result = {
            "type": e_type,
            "src": e_src,
            "tgt": e_tgt,
            "offset_src": e_offset_src,
            "offset_tgt": e_offset_tgt,
        }
        return result

    def process_loop_graph(l_desc, parms):
        l_items = l_desc.attrib
        l_type = l_items.get("type", "0")
        nodes = []
        for node in l_desc.findall("./NODE"):
            node_items = node.attrib
            l_src = node_items.get("vertex", "")
            nodes.append({"vertex": l_src})

        return {"type": l_type, "nodes": nodes}

    def process_loop_unitcell(l_desc, parms, dimension):
        l_items = l_desc.attrib
        l_type = l_items.get("type", "0")
        nodes = []
        for node in l_desc.findall("./NODE"):
            node_items = node.attrib
            l_src = node_items.get("vertex", "")
            l_offset = [
                int(c) for c in node_items.get("offset", "").split(" ") if c != ""
            ]
            l_offset = l_offset + (dimension - len(l_offset)) * [0]
            nodes.append({"vertex": l_src, "offset": l_offset})

        return {"type": l_type, "nodes": nodes}

    def process_unitcell(node, parms):
        parms = process_parms(node, parms)
        vertices = {}
        edges = {}
        loops = {}
        unitcell = {"vertices": vertices, "edges": edges, "loops": loops}
        dimension = int(node.attrib.get("dimension", 0))
        unitcell["dimension"] = dimension
        unitcell["name"] = node.attrib.get("name", "")

        for v_desc in node.findall("./VERTEX"):
            v_items = process_vertex(v_desc, parms)
            name = v_items.pop("name", None) or next_name(vertices)
            vertices[name] = v_items

        for e_desc in node.findall("./EDGE"):
            edge_dict = process_edge_unitcell(e_desc, parms, dimension)
            e_type = edge_dict.pop("type", None) or next_name(edges)
            edge_type_list = edges.get(e_type, [])
            edge_type_list.append(edge_dict)
            edges[e_type] = edge_type_list

        for loop_desc in node.findall("./LOOP"):
            loop_dict = process_loop_unitcell(loop_desc, parms, dimension)
            l_type = loop_dict.pop("type", None) or next_name(loops)
            loop_type_list = loops.get(l_type, [])
            loop_type_list.append(loop_dict)
            loops[l_type] = loop_type_list
        return unitcell

    def process_vertex(node, parms):
        """Process a <VERTEX> node"""
        v_attributes = node.attrib
        v_attributes["type"] = v_attributes.get("type", "0")
        v_attributes["coords"] = None
        for coord_desc in node.findall("./COORDINATE"):
            v_attributes["coords"] = process_coordinates(coord_desc.text)
        return v_attributes

    # Try to find a graph
    for graph_desc in lattices.findall("./GRAPH"):
        if ("name", name) in graph_desc.items():
            return process_graph(graph_desc, parms)

    # Otherwise, try with a lattice
    for lat_desc in lattices.findall("./LATTICEGRAPH"):
        if ("name", name) in lat_desc.items():
            return process_latticegraph(lat_desc, parms)

    return None


class GraphDescriptor:
    """
    A description of a Graph
    """

    name: str
    nodes: dict
    edges: dict
    parms: dict
    subgraphs: dict

    def __init__(
        self,
        name: str,
        nodes: dict,
        edges: Optional[dict],
        loops: Optional[dict],
        lattice: Optional[dict] = None,
        parms: Optional[dict] = None,
    ):
        self.name = name
        self.nodes = nodes
        self.edges = edges or {}
        self.loops = loops or {}
        self.lattice = lattice or None
        self.parms = parms or {}
        self.complete_coordiantes()
        self.subgraphs = {}

    def complete_coordiantes(self):
        """Add coordinates to nodes without specified coordinates"""
        nodes = self.nodes
        lattice = self.lattice
        # TODO: it would be great to use a better algorithm to
        # build the missing coordinates.
        if lattice is not None:
            basis = [np.array(b) for b in lattice["basis"]]
            dim = len(basis[0])
            local_coords = {}
            for name, n_attr in nodes.items():
                if n_attr.get("coords", None) is not None:
                    continue
                n_attr = n_attr.copy()
                nodes[name] = n_attr
                if "[" in name:
                    v_name, cell_str = name.split("[")
                    cell = [int(c) for c in cell_str[:-1].strip().split(",")]
                    l_coords = local_coords.get(v_name, rand(dim))
                    local_coords[v_name] = l_coords
                    n_attr["coords"] = (
                        sum(c * b for c, b in zip(cell, basis)) + l_coords
                    )
                else:
                    n_attr["coords"] = rand(dim)
        else:
            for name, n_attr in nodes.items():
                if n_attr.get("coords", None) is not None:
                    continue
                n_attr = n_attr.copy()
                nodes[name] = n_attr
                n_attr["coords"] = rand(2)

    def __repr__(self):
        result = (
            f"Graph {self.name}. Vertices: \n  "
            + "\n  ".join([f" {i} of type {t}" for i, t in self.nodes.items()])
            + "\nEdges"
        )

        for b_type, bnds in self.edges.items():
            result += f"\n type {b_type}:\n    " + "\n    ".join(
                f"{b[0]}-{b[1]}" for b in bnds
            )
        return result

    def draw(self, ax_mpl, node_spec=None, edge_spec=None):
        """Draw the graph over a matplotlib axis"""
        coords = {}

        if node_spec is None:
            node_spec = {}
        if edge_spec is None:
            edge_spec = {}

        default_node_spec = {
            "c": "blue",
        }
        default_edge_spec = {
            "c": "blue",
        }

        nodes = self.nodes
        edges = self.edges

        if self.lattice and self.lattice["dimension"] > 2:
            for name in self.nodes:
                coords[name] = nodes[name]["coords"][:3]

        elif self.lattice and self.lattice["dimension"] == 1:
            for name in self.nodes:
                if len(nodes[name]["coords"]) == 1:
                    coords[name] = np.array([nodes[name]["coords"][0], 0.0])
                else:
                    coords[name] = np.array(nodes[name]["coords"][:2])
        else:
            for name in self.nodes:
                coords[name] = nodes[name]["coords"][:2]

        for name, x_coord in coords.items():
            spec = node_spec.get(nodes[name]["type"], default_node_spec)
            ax_mpl.scatter(*[[u] for u in x_coord], **spec)

        for bond_type, b_list in edges.items():
            spec = edge_spec.get(bond_type, default_edge_spec)
            for src_name, tgt_name in b_list:
                src, tgt = coords[src_name], coords[tgt_name]
                # TODO: color by type t
                ax_mpl.plot(*[[u, v] for u, v in zip(src, tgt)], **spec)

    def subgraph(self, node_tuple: frozenset, name: str = ""):
        """A subgraph containing the specified nodes"""
        assert isinstance(node_tuple, frozenset)
        subgraph = self.subgraphs.get(node_tuple, None)
        if subgraph is not None:
            return subgraph

        nodes = self.nodes
        nodes = {n: nodes[n] for n in node_tuple}
        edges = {
            t: [
                (
                    src,
                    dst,
                )
                for src, dst in e
                if (src in nodes) and (dst in nodes)
            ]
            for t, e in self.edges.items()
        }

        loops = {
            l_t: loop
            for l_t, loop in self.loops.items()
            if all(all(v in nodes for v in v_lst) for v_lst in loop)
        }

        subgraph = GraphDescriptor(name, nodes, edges, loops)
        self.subgraphs[node_tuple] = subgraph
        return subgraph

    def __add__(self, other):
        return self.union(other)

    def union(self, other):
        """
        Join two graphics
        """
        if self is other or other is None:
            return self
        other_nodes = other.nodes
        nodes = self.nodes.copy()
        intersection = [site for site in nodes if site in other.nodes]
        assert all(
            nodes[site] == other_nodes[site] for site in intersection
        ), "nodes in the intersection should be equal"
        nodes.update(other_nodes)
        edges = self.edges.copy()
        edges.update(other.edges)
        loops = self.loops.copy()
        loops.update(other.loops)
        result = GraphDescriptor(f"{self.name}+{other.name}", nodes, edges, loops)
        result.subgraphs[frozenset(self.nodes)] = self
        result.subgraphs[frozenset(other.nodes)] = other
        return result
