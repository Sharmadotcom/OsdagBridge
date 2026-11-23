"""
results_data.py
---------------
Extracts analysis results from an ospgrillage OspGrillage model directly into
the flat dict format consumed by plot_generator functions.

Accepts the OspGrillage model object and its xarray Dataset directly, with no
bridge-wrapper dependency.
"""

from __future__ import annotations
import json
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

from .results_data_post_processing import post_process, FORCE_KEEP, DISP_KEEP


class LazyLoadcaseResults(Mapping):
    """
    Per-loadcase results table backed by the xarray dataset.

    Behaves like ``{loadcase: {id: {component: value}}}`` but materializes a
    load case's nested dict only on access and holds at most ``_MAX_CACHED``
    of them resident. Consumers (the governing-force scans, dumps, dialogs)
    walk load cases sequentially, so the small LRU yields one materialization
    per case instead of keeping every case's tree of Python floats alive —
    previously ~n_loadcases full nested dicts duplicating the dataset.

    The FORCE_KEEP/DISP_KEEP component whitelist (post_process's cleaning
    step) is applied during materialization, so entries look exactly like the
    cleaned eager dicts.
    """
    _MAX_CACHED = 8

    def __init__(self, ds_all, var: str, keep=None):
        self._ds = ds_all
        self._var = var
        self._keep = keep
        self._keys = [str(lc) for lc in ds_all.coords["Loadcase"].values]
        self._key_set = set(self._keys)
        self._cache: OrderedDict = OrderedDict()

    def _materialize(self, lc: str) -> dict:
        da = self._ds[self._var].sel(Loadcase=lc)
        cids = [str(c) for c in da.coords["Component"].values]
        col_idx = (
            list(range(len(cids))) if self._keep is None
            else [i for i, c in enumerate(cids) if c in self._keep]
        )
        kept = [cids[i] for i in col_idx]
        dim = "Element" if "Element" in da.coords else "Node"
        ids = da.coords[dim].values
        return {
            str(int(i)): {c: row[j] for c, j in zip(kept, col_idx)}
            for i, row in zip(ids, da.values.tolist())
        }

    def __getitem__(self, lc):
        lc = str(lc)
        if lc in self._cache:
            self._cache.move_to_end(lc)
            return self._cache[lc]
        if lc not in self._key_set:
            raise KeyError(lc)
        table = self._materialize(lc)
        self._cache[lc] = table
        if len(self._cache) > self._MAX_CACHED:
            self._cache.popitem(last=False)
        return table

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

_TOOLS_DIR = Path(__file__).resolve().parents[5] / "tools"


def _build_nodes_members() -> tuple[dict, dict]:
    """Read node coords and element connectivity from the live openseespy model."""
    nodes = {
        int(n): list(map(float, ops.nodeCoord(n)))
        for n in ops.getNodeTags()
    }
    members = {
        int(e): list(map(int, ops.eleNodes(e)))
        for e in ops.getEleTags()
    }
    return nodes, members


def classify_grillage_elements(
    nodes: dict,
    members: dict,
    model=None,
    groups: dict | None = None,
) -> dict:
    """
    Classify grillage elements into longitudinal and transverse members,
    distinguishing end diaphragms and intermediate cross bracings.

    Parameters
    ----------
    nodes   : dict — {node_id: [x, y, z]}
    members : dict — {elem_id: [n1, n2]}
    model   : OspGrillage, optional
    groups  : dict, optional

    Returns
    -------
    dict with keys:
        - "longitudinal_members": list of int element tags
        - "transverse_members": list of int element tags
        - "end_diaphragm_members": list of int element tags (start_edge + end_edge)
        - "cross_bracing_members": list of int element tags (transverse_slab)
        - "start_edge": list of int element tags
        - "end_edge": list of int element tags
        - "transverse_slab": list of int element tags
        - "groups": dict of group_name -> list of int tags
    """
    nodes = {int(k): list(map(float, v)) for k, v in nodes.items()}
    members = {int(k): list(map(int, v)) for k, v in members.items()}

    grp = dict(groups) if groups else {}
    standard_groups = [
        "edge_beam", "exterior_main_beam_1", "exterior_main_beam_2",
        "interior_main_beam", "start_edge", "end_edge", "transverse_slab",
        "shell_slab"
    ]
    if model is not None:
        for g_name in standard_groups:
            if g_name not in grp or not grp[g_name]:
                try:
                    grp[g_name] = [int(e) for e in model.get_element(member=g_name, options="elements")]
                except Exception:
                    grp.setdefault(g_name, [])

    start_edge = [int(e) for e in grp.get("start_edge", []) if int(e) in members]
    end_edge = [int(e) for e in grp.get("end_edge", []) if int(e) in members]
    trans_slab = [int(e) for e in grp.get("transverse_slab", []) if int(e) in members]

    # Geometric classification for all elements
    longitudinal_members = []
    transverse_members = []
    for tag, (n1, n2) in members.items():
        if n1 not in nodes or n2 not in nodes:
            continue
        c1 = nodes[n1]
        c2 = nodes[n2]
        dx = abs(c1[0] - c2[0])
        dz = abs(c1[2] - c2[2])
        if dz > dx:
            transverse_members.append(tag)
        else:
            longitudinal_members.append(tag)

    # Fallback if model group queries returned empty
    if not start_edge and not end_edge and not trans_slab and transverse_members:
        trans_node_xs = [nodes[n][0] for e in transverse_members for n in members[e] if n in nodes]
        if trans_node_xs:
            min_x = min(trans_node_xs)
            max_x = max(trans_node_xs)
            tol = 0.05  # 50 mm
            for e in transverse_members:
                n1, n2 = members[e]
                mx = (nodes[n1][0] + nodes[n2][0]) / 2.0
                if abs(mx - min_x) <= tol:
                    start_edge.append(e)
                elif abs(mx - max_x) <= tol:
                    end_edge.append(e)
                else:
                    trans_slab.append(e)

    end_diaphragm_members = sorted(set(start_edge + end_edge))
    cross_bracing_members = sorted(set(trans_slab))
    all_trans = sorted(set(transverse_members + end_diaphragm_members + cross_bracing_members))
    all_long = sorted(set(e for e in members if e not in all_trans))

    grp["start_edge"] = sorted(set(start_edge))
    grp["end_edge"] = sorted(set(end_edge))
    grp["transverse_slab"] = sorted(set(trans_slab))

    return {
        "longitudinal_members": all_long,
        "transverse_members": all_trans,
        "end_diaphragm_members": end_diaphragm_members,
        "cross_bracing_members": cross_bracing_members,
        "start_edge": grp["start_edge"],
        "end_edge": grp["end_edge"],
        "transverse_slab": grp["transverse_slab"],
        "groups": grp,
    }


def extract_transverse_chains(
    nodes: dict,
    members: dict,
    transverse_tags: list[int] | None = None,
    model=None,
    groups: dict | None = None,
) -> list[dict]:
    """
    Extract ordered transverse member chains (lines across bridge width).
    Each chain represents an end diaphragm line or intermediate cross bracing line.

    Returns a list of dicts:
        [
            {
                "name": str,
                "short_name": str,
                "category": "end_diaphragm" | "cross_bracing",
                "station_x": float,
                "elements": list[int],
                "nodes": list[int],
                "xs": np.ndarray,
                "ys": np.ndarray,
                "zs": np.ndarray,
            },
            ...
        ]
    """
    nodes = {int(k): list(map(float, v)) for k, v in nodes.items()}
    members = {int(k): list(map(int, v)) for k, v in members.items()}

    classification = classify_grillage_elements(nodes, members, model=model, groups=groups)
    if transverse_tags is None:
        transverse_tags = classification["transverse_members"]
    transverse_set = set(int(t) for t in transverse_tags if int(t) in members)
    if not transverse_set:
        return []

    end_diaphragm_set = set(classification["end_diaphragm_members"])

    # Cluster transverse elements into stations (by midpoint X)
    elem_info = []
    for e in transverse_set:
        n1, n2 = members[e]
        if n1 not in nodes or n2 not in nodes:
            continue
        c1 = nodes[n1]
        c2 = nodes[n2]
        mid_x = (c1[0] + c2[0]) / 2.0
        mid_z = (c1[2] + c2[2]) / 2.0
        elem_info.append((e, mid_x, mid_z, n1, n2))

    # Sort primarily by mid_x
    elem_info.sort(key=lambda item: item[1])

    # Group into station clusters within 0.08m tolerance
    clusters: list[list[tuple]] = []
    for item in elem_info:
        if not clusters:
            clusters.append([item])
        else:
            prev_station = np.mean([it[1] for it in clusters[-1]])
            if abs(item[1] - prev_station) <= 0.08:
                clusters[-1].append(item)
            else:
                clusters.append([item])

    chains = []
    station_xs = [float(np.mean([it[1] for it in cl])) for cl in clusters]
    min_sx = min(station_xs) if station_xs else 0.0
    max_sx = max(station_xs) if station_xs else 0.0

    cb_idx = 1
    for cl in clusters:
        # Sort elements within cluster across bridge width by mid_z
        cl.sort(key=lambda it: it[2])
        station_x = round(float(np.mean([it[1] for it in cl])), 3)

        ordered_elements = [it[0] for it in cl]

        # Trace / order nodes along the transverse line
        ordered_nodes = []
        for e in ordered_elements:
            n1, n2 = members[e]
            if nodes[n1][2] > nodes[n2][2]:
                n1, n2 = n2, n1
            if not ordered_nodes:
                ordered_nodes.extend([n1, n2])
            else:
                if n1 == ordered_nodes[-1]:
                    ordered_nodes.append(n2)
                elif n2 == ordered_nodes[-1]:
                    ordered_nodes.append(n1)
                else:
                    if n1 not in ordered_nodes:
                        ordered_nodes.append(n1)
                    if n2 not in ordered_nodes:
                        ordered_nodes.append(n2)

        xs = np.array([nodes[n][0] for n in ordered_nodes])
        ys = np.array([nodes[n][1] for n in ordered_nodes])
        zs = np.array([nodes[n][2] for n in ordered_nodes])

        is_end_diaphragm = (
            any(e in end_diaphragm_set for e in ordered_elements)
            or abs(station_x - min_sx) < 0.08
            or abs(station_x - max_sx) < 0.08
        )

        if is_end_diaphragm:
            category = "end_diaphragm"
            if abs(station_x - min_sx) <= abs(station_x - max_sx):
                name = f"End Diaphragm (Start - {station_x:.2f}m)"
                short_name = "End Diaphragm 1"
            else:
                name = f"End Diaphragm (End - {station_x:.2f}m)"
                short_name = "End Diaphragm 2"
        else:
            category = "cross_bracing"
            name = f"Cross Bracing @ {station_x:.2f}m"
            short_name = f"Cross Bracing {cb_idx} ({station_x:.2f}m)"
            cb_idx += 1

        chains.append({
            "name": name,
            "short_name": short_name,
            "category": category,
            "station_x": station_x,
            "elements": ordered_elements,
            "nodes": ordered_nodes,
            "xs": xs,
            "ys": ys,
            "zs": zs,
        })

    return chains


def extract_member_diagram_data(
    ds,
    member_selection: str,
    component_key: str,
    nodes: dict,
    members: dict,
    model=None,
    edge_dist: float = 0.0,
) -> tuple[list[dict], dict]:
    """
    Unified extraction router for longitudinal and transverse member diagram data.
    Extracts all 9 internal forces and displacements:
      - Forces:       Fx (axial), Fy (shear y), Fz (shear z), Mx (torsion), My (moment y), Mz (moment z)
      - Displacements: Dx (dx), Dy (dy), Dz (dz)

    Parameters
    ----------
    ds               : xarray.Dataset (must have 'forces' or 'displacements')
    member_selection : str
        "All", "G1", "G2", "Transverse Members", "End Diaphragms",
        "Intermediate Cross Bracings", or specific chain name.
    component_key    : str
        "Fx", "Fy", "Fz", "Mx", "My", "Mz", "Dx", "Dy", "Dz", or aliases.
    nodes            : dict — {node_id: [x, y, z]}
    members          : dict — {elem_id: [n1, n2]}
    model            : OspGrillage, optional
    edge_dist        : float

    Returns
    -------
    polylines    : list of polyline dicts
    summary_data : dict of {name: {"max": float, "min": float}}
    """
    nodes = {int(k): list(map(float, v)) for k, v in nodes.items()}
    members = {int(k): list(map(int, v)) for k, v in members.items()}

    disp_aliases = {
        "dx": "x", "dy": "y", "dz": "z",
        "Dx": "x", "Dy": "y", "Dz": "z",
        "x": "x", "y": "y", "z": "z",
    }
    force_aliases = {
        "fx": ("Vx_i", "Vx_j"), "Fx": ("Vx_i", "Vx_j"),
        "fy": ("Vy_i", "Vy_j"), "Fy": ("Vy_i", "Vy_j"), "vy": ("Vy_i", "Vy_j"), "Vy": ("Vy_i", "Vy_j"),
        "fz": ("Vz_i", "Vz_j"), "Fz": ("Vz_i", "Vz_j"), "vz": ("Vz_i", "Vz_j"), "Vz": ("Vz_i", "Vz_j"),
        "mx": ("Mx_i", "Mx_j"), "Mx": ("Mx_i", "Mx_j"), "tx": ("Mx_i", "Mx_j"), "Tx": ("Mx_i", "Mx_j"),
        "my": ("My_i", "My_j"), "My": ("My_i", "My_j"),
        "mz": ("Mz_i", "Mz_j"), "Mz": ("Mz_i", "Mz_j"),
    }

    is_displacement = component_key in disp_aliases
    is_transverse_sel = (
        member_selection in ("Transverse Members", "End Diaphragms", "Intermediate Cross Bracings")
        or "Diaphragm" in member_selection
        or "Cross Bracing" in member_selection
        or member_selection.startswith("ED")
        or member_selection.startswith("CB")
    )

    # 1. Resolve dataset components
    actual_disp_comp = None
    comp_i_name, comp_j_name = None, None

    if is_displacement:
        target_name = disp_aliases.get(component_key, component_key).lower()
        if "displacements" in ds:
            for c in ds["displacements"].coords["Component"].values:
                if str(c).lower() == target_name:
                    actual_disp_comp = c
                    break
    else:
        fi, fj = force_aliases.get(component_key, ("Vy_i", "Vy_j"))
        if "forces" in ds:
            comp_vals = [str(c) for c in ds["forces"].coords["Component"].values]
            for c in comp_vals:
                if c.lower() == fi.lower():
                    comp_i_name = c
                if c.lower() == fj.lower():
                    comp_j_name = c

    # Helper: read displacement at node
    def _get_node_disp(nid: int) -> float:
        if actual_disp_comp is None or "displacements" not in ds:
            return 0.0
        try:
            val = float(ds["displacements"].sel(Node=nid, Component=actual_disp_comp).values)
            if str(actual_disp_comp).lower() in ("y", "z"):
                val = -val  # engineering sign flip for sag
            return val * 1000.0  # m -> mm
        except Exception:
            return 0.0

    # Helper: read force at element
    def _get_elem_forces(eid: int) -> tuple[float, float]:
        if "forces" not in ds or comp_i_name is None or comp_j_name is None:
            return 0.0, 0.0
        try:
            vi = float(ds["forces"].sel(Element=eid, Component=comp_i_name).values) / 1000.0
            vj = float(ds["forces"].sel(Element=eid, Component=comp_j_name).values) / 1000.0
            return vi, vj
        except Exception:
            return 0.0, 0.0

    polylines = []
    summary_data = {}

    if is_transverse_sel:
        all_chains = extract_transverse_chains(nodes, members, model=model)
        for chain in all_chains:
            if member_selection == "End Diaphragms" and chain["category"] != "end_diaphragm":
                continue
            if member_selection == "Intermediate Cross Bracings" and chain["category"] != "cross_bracing":
                continue
            if member_selection not in ("Transverse Members", "End Diaphragms", "Intermediate Cross Bracings"):
                if member_selection != chain["name"] and member_selection != chain["short_name"]:
                    continue

            ordered_elems = chain["elements"]
            ordered_nodes = chain["nodes"]

            if is_displacement:
                vals = [_get_node_disp(n) for n in ordered_nodes]
            else:
                vals = []
                for idx, e in enumerate(ordered_elems):
                    vi, vj = _get_elem_forces(e)
                    vals.append(vi)
                    if idx == len(ordered_elems) - 1:
                        vals.append(-vj if component_key in ("Fy", "Fz", "Fx", "Vy", "Vz") else vj)

            vals_arr = np.array(vals, dtype=float)
            polyline_entry = {
                "name": chain["short_name"],
                "full_name": chain["name"],
                "category": "transverse",
                "sub_category": chain["category"],
                "is_transverse": True,
                "station_x": chain["station_x"],
                "elements": ordered_elems,
                "node_ids": ordered_nodes,
                "xs": chain["xs"],
                "ys": chain["ys"],
                "zs": chain["zs"],
                "vals": vals_arr,
                "is_edge": False,
            }
            polylines.append(polyline_entry)
            if vals_arr.size:
                summary_data[chain["short_name"]] = {
                    "max": float(np.max(vals_arr)),
                    "min": float(np.min(vals_arr)),
                }

    else:
        # Longitudinal girders
        node_z = {n: round(coord[2], 3) for n, coord in nodes.items()}
        girders_map = defaultdict(list)
        for ele, (n1, n2) in members.items():
            if n1 in node_z and n2 in node_z and node_z[n1] == node_z[n2]:
                girders_map[node_z[n1]].append(ele)
        for z_val in girders_map:
            girders_map[z_val].sort(key=lambda e: nodes[members[e][0]][0])

        girder_items = sorted(girders_map.items())
        n_girders = len(girder_items)

        for i, (z_val, elems) in enumerate(girder_items):
            is_edge_beam = edge_dist > 0 and (i == 0 or i == n_girders - 1)
            girder_name = f"G{i}" if edge_dist > 0 else f"G{i + 1}"

            if member_selection != "All" and girder_name != member_selection:
                continue

            node_list = [members[e][0] for e in elems] + [members[elems[-1]][1]]
            xs = np.array([nodes[n][0] for n in node_list])
            ys = np.array([nodes[n][1] for n in node_list])
            zs = np.array([nodes[n][2] for n in node_list])

            if is_displacement:
                vals = [_get_node_disp(n) for n in node_list]
            else:
                vals = []
                for idx, e in enumerate(elems):
                    vi, vj = _get_elem_forces(e)
                    vals.append(vi)
                    if idx == len(elems) - 1:
                        vals.append(-vj if component_key in ("Fy", "Fz", "Fx", "Vy", "Vz") else vj)

            vals_arr = np.array(vals, dtype=float)
            polyline_entry = {
                "name": girder_name,
                "full_name": girder_name,
                "category": "longitudinal",
                "sub_category": "edge_beam" if is_edge_beam else "girder",
                "is_transverse": False,
                "station_x": None,
                "elements": elems,
                "node_ids": node_list,
                "xs": xs,
                "ys": ys,
                "zs": zs,
                "vals": vals_arr,
                "is_edge": is_edge_beam,
            }
            polylines.append(polyline_entry)
            if (not is_edge_beam) and vals_arr.size:
                summary_data[girder_name] = {
                    "max": float(np.max(vals_arr)),
                    "min": float(np.min(vals_arr)),
                }

    return polylines, summary_data


def dump_full_data(
    model,
    edge_dist: float = 0.0,
    out_path: str | Path | None = None,
    dataset=None,
) -> dict:
    """
    Dump the *entire* extracted result data to a JSON file.

    Unlike ``restructure_data(model, dev=True)``, this writes the complete,
    un-filtered data dict (including ``groups``, ``reactions``,
    ``forces_shell``, ``stresses_shell`` and ``edge_dist`` — keys that
    ``post_process`` would otherwise drop) and performs **no** other side
    effects (no crossbracing dump, no plotly HTML generation).

    Parameters
    ----------
    model : OspGrillage
        Fully analysed ospgrillage model (analyze() already called).
    edge_dist : float
        Deck overhang distance in metres (0.0 when no overhang).
    out_path : str | Path, optional
        Destination JSON path.  Defaults to ``tools/bridge_full_data.json``.

    Returns
    -------
    dict — the full data dict that was written.
    """
    data = _build_full_data(model, edge_dist, dataset=dataset)

    if out_path is None:
        _TOOLS_DIR.mkdir(exist_ok=True)
        out_path = _TOOLS_DIR / "bridge_full_data.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print("=" * 60)
    print(f"[full data] saved → {out_path.resolve()}")
    print("=" * 60)

    return data


def restructure_data(
    model,
    edge_dist: float = 0.0,
    dev: bool = False,
    dataset=None,
    lazy: bool = False,
) -> dict:
    """
    Restructure xarray analysis results into a flat dict for plot_generator and
    any other consumer.  Optionally dumps to tools/bridge_plot_data.json.

    Uses all available ospgrillage APIs to extract metadata (nodes, groups) 
    and results (forces, displacements, reactions, shell results).

    Parameters
    ----------
    model : OspGrillage
        Fully analysed ospgrillage model (analyze() already called).
    dataset : xarray.Dataset, optional
        Pre-computed results dataset.  When omitted, ``model.get_results()``
        is called to obtain it.
    edge_dist : float
        Deck overhang distance in metres (0.0 when no overhang).
    dev : bool
        If True, also write the result to tools/bridge_plot_data.json.

    Returns
    -------
    dict — structure::

    {
        "nodes": {
            "1": [0.0, 0.0, 0.0],
            "2": [0.0, 0.0, 0.75],
            "...": "..."
        },

        "members": {
            "1": [1, 2],
            "2": [2, 3],
            "...": "..."
        },

        "groups": {
            "edge_beam": [
                11,
                22,
                33,
                "..."
            ],

            "exterior_main_beam_1": [
                12,
                23,
                34,
                "..."
            ],

            "exterior_main_beam_2": [
                22,
                33,
                44,
                "..."
            ],

            "interior_main_beam": [
                13,
                24,
                35,
                "..."
            ],

            "start_edge": [
                1,
                2,
                3,
                "..."
            ],
            "end_edge": [
                138,
                139,
                140,
                141,
                ...
            ],
            "transverse_slab": [
                6,
                7,
                8,
                9,
                ...,

            "shell_slab": []
        },

        "longitudinal_members": [
            11,
            12,
            13,
            "..."
        ],

        "transverse_members": [
            1,
            2,
            3,
            "..."
        ],

        "edge_dist": 0.75,

        "loadcases": [
            "girder self weight",
            "superimposed dead load",
            "live load",
            "..."
        ],

        "force_components": [
            "Mx_i",
            "Mx_j",
            "My_i",
            "My_j",
            "Mz_i",
            "Mz_j",
            "Vx_i",
            "Vx_j",
            "Vy_i",
            "Vy_j",
            "Vz_i",
            "Vz_j"
        ],

        "disp_components": [
            "theta_x",
            "theta_y",
            "theta_z",
            "x",
            "y",
            "z"
        ],

        "forces": {
            "girder self weight": {
            "1": {
                "Mx_i": -0.7137482121047469,
                "Mx_j": -4478.21603280773,
                "My_i": 0.0,
                "My_j": 0.0,
                "...": "..."
            },

            "2": {
                "...": "..."
            },

            "...": "..."
            },

            "live load": {
            "...": "..."
            }
        },

        "displacements": {
            "girder self weight": {
            "1": {
                "theta_x": 2.1983232320953836e-05,
                "theta_y": 0.0,
                "theta_z": 0.00038309565871718596,
                "x": 0.0,
                "y": 1.3989811021922501e-05,
                "z": 0.0
            },

            "2": {
                "...": "..."
            },

            "...": "..."
            }
        },

        "reactions": {
            "girder self weight": {
             "...": "..."
            }
        },

        "forces_shell": {
            "...": "..."
        },

        "stresses_shell": {
            "...": "..."
        }
    }
    """
    data = _build_full_data(model, edge_dist, dataset=dataset, lazy=lazy)

    if dev:
        # dev dumps require eager (JSON-serializable) tables.
        _dump(data, "bridge_plot_data_raw.json")

    data = post_process(data)

    if dev:
        _dump(data)
        _dump_crossbracing(data)
        _plot_dev(model, model.get_results())

    return data


def _build_full_data(model, edge_dist: float = 0.0, dataset=None, lazy: bool = False) -> dict:
    """
    Build the complete, un-filtered result data dict from the live model and
    its xarray results.  This is the shared extraction step used by both
    ``restructure_data`` (before post-processing) and ``dump_full_data``.

    When ``dataset`` is provided it is used in place of ``model.get_results()``.
    This is how the envelope pseudo load cases (``Envelope ULS`` /
    ``Envelope SLS``) reach the dump: they live only on the cached, augmented
    dataset, never on the model's freshly-rebuilt results.
    """
    ds_all = model.get_results() if dataset is None else dataset
    # Deduplicate: ospgrillage appends results on every analyze() call, so
    # a second analyze() (e.g. after adding the governing LL case) produces
    # duplicate Loadcase entries.  Keep first occurrence of each name and
    # drop the duplicates from ds_all so that .sel(Loadcase=lc) always
    # returns a 0-d slice (not a multi-row array that breaks float()).
    _seen_lcs: set = set()
    _unique_idx: list = []
    for _i, _lc in enumerate(ds_all.coords["Loadcase"].values):
        _s = str(_lc)
        if _s not in _seen_lcs:
            _seen_lcs.add(_s)
            _unique_idx.append(_i)
    if len(_unique_idx) < len(ds_all.coords["Loadcase"].values):
        ds_all = ds_all.isel(Loadcase=_unique_idx)
    loadcases = [str(lc) for lc in ds_all.coords["Loadcase"].values]

    # ── 1. Metadata from live model / ospgrillage APIs ────────────────
    nodes, members = _build_nodes_members()

    # Extract standard member groups via ospgrillage get_element API
    standard_groups = [
        "edge_beam", "exterior_main_beam_1", "exterior_main_beam_2",
        "interior_main_beam", "start_edge", "end_edge", "transverse_slab",
        "shell_slab"
    ]
    groups = {}
    for group_name in standard_groups:
        try:
            # get_element returns list of tags
            groups[group_name] = [int(e) for e in model.get_element(
                member=group_name, options="elements")]
        except Exception:
            groups[group_name] = []

    # Coordinate components
    ds0 = ds_all.isel(Loadcase=0)
    force_components = [str(
        c) for c in ds0["forces"].coords["Component"].values] if "forces" in ds0.data_vars else []
    disp_components = [str(
        c) for c in ds0["displacements"].coords["Component"].values] if "displacements" in ds0.data_vars else []

    # Classify elements into longitudinal, transverse, end diaphragms, and cross bracings
    classification = classify_grillage_elements(nodes, members, model=model, groups=groups)
    transverse_chains = extract_transverse_chains(nodes, members, model=model, groups=classification["groups"])

    data: dict = {
        "nodes": {str(k): list(v) for k, v in nodes.items()},
        "members": {str(k): list(v) for k, v in members.items()},
        "groups": classification["groups"],
        "longitudinal_members": classification["longitudinal_members"],
        "transverse_members": classification["transverse_members"],
        "end_diaphragm_members": classification["end_diaphragm_members"],
        "cross_bracing_members": classification["cross_bracing_members"],
        "start_edge": classification["start_edge"],
        "end_edge": classification["end_edge"],
        "transverse_slab": classification["transverse_slab"],
        "transverse_chains": transverse_chains,
        "edge_dist": float(edge_dist),
        "loadcases": list(loadcases),
        "force_components": force_components,
        "disp_components": disp_components,
        "forces": {},
        "displacements": {},
        "reactions": {},
        "forces_shell": {},
        "stresses_shell": {},
    }

    # Lazy mode: the two big per-loadcase tables are served straight from the
    # dataset on demand (LRU of a few load cases) instead of being duplicated
    # up front as nested dicts of Python floats. The minor tables (reactions /
    # shell results) stay eager below.
    if lazy:
        if "forces" in ds_all.data_vars:
            data["forces"] = LazyLoadcaseResults(ds_all, "forces", FORCE_KEEP)
        if "displacements" in ds_all.data_vars:
            data["displacements"] = LazyLoadcaseResults(ds_all, "displacements", DISP_KEEP)

    # ── 2. Results from xarray Dataset ────────────────────────────────
    # Extract full numpy arrays once per load case instead of one .sel()
    # call per (element/node × component) — avoids tens of thousands of
    # individual xarray label lookups.
    for lc in loadcases:
        ds = ds_all.sel(Loadcase=lc)
        str_lc = str(lc)

        # Displacements — one .values call → (n_nodes, n_comps) numpy array
        if not lazy and "displacements" in ds.data_vars:
            arr  = ds["displacements"].values
            nids = ds["displacements"].coords["Node"].values
            cids = [str(c) for c in ds["displacements"].coords["Component"].values]
            data["displacements"][str_lc] = {
                str(int(n)): dict(zip(cids, row.tolist()))
                for n, row in zip(nids, arr)
            }

        # Forces — one .values call → (n_elems, n_comps) numpy array
        if not lazy and "forces" in ds.data_vars:
            arr  = ds["forces"].values
            eids = ds["forces"].coords["Element"].values
            cids = [str(c) for c in ds["forces"].coords["Component"].values]
            data["forces"][str_lc] = {
                str(int(e)): dict(zip(cids, row.tolist()))
                for e, row in zip(eids, arr)
            }

        # Reactions — one .values call → (n_nodes, n_comps) numpy array
        if "reactions" in ds.data_vars:
            arr  = ds["reactions"].values
            nids = ds["reactions"].coords["Node"].values
            cids = [str(c) for c in ds["reactions"].coords["Component"].values]
            data["reactions"][str_lc] = {
                str(int(n)): dict(zip(cids, row.tolist()))
                for n, row in zip(nids, arr)
            }

        # Shell Forces — one .values call → (n_elems, n_comps) numpy array
        if "forces_shell" in ds.data_vars:
            arr  = ds["forces_shell"].values
            eids = ds["forces_shell"].coords["Element"].values
            cids = [str(c) for c in ds["forces_shell"].coords["Component"].values]
            data["forces_shell"][str_lc] = {
                str(int(e)): dict(zip(cids, row.tolist()))
                for e, row in zip(eids, arr)
            }

        # Shell Stresses — one .values call → (n_elems, n_stress_comps) numpy array
        if "stresses_shell" in ds.data_vars:
            arr  = ds["stresses_shell"].values
            eids = ds["stresses_shell"].coords["Element"].values
            cids = [str(c) for c in ds["stresses_shell"].coords["Stress"].values]
            data["stresses_shell"][str_lc] = {
                str(int(e)): dict(zip(cids, row.tolist()))
                for e, row in zip(eids, arr)
            }

    return data


def _dump_crossbracing(data: dict) -> None:
    """Write raw crossbracing Vz forces to tools/crossbracing_results.json."""
    import math
    import warnings as _warnings

    EQ_TOL = 1e-3  # kN

    cb_chains  = data.get("crossbracings", [])
    forces     = data.get("forces", {})
    load_cases = data.get("loadcases", list(forces.keys()))

    results = {}

    for cb_num, chain in enumerate(cb_chains, start=1):
        mems = chain.get("members", [])
        if not mems:
            continue

        m_id = str(mems[0])

        start = chain.get("start")
        end   = chain.get("end")
        length_m = None
        if start and end:
            sc = start["coords"]
            ec = end["coords"]
            length_m = round(math.sqrt(sum((ec[i] - sc[i]) ** 2 for i in range(3))), 4)

        lc_results = []

        for lc in load_cases:
            lc_str = str(lc)
            elem_f = forces.get(lc_str, {}).get(m_id, {})

            vz_i = elem_f.get("Vz_i")
            vz_j = elem_f.get("Vz_j")

            if vz_i is None or vz_j is None:
                continue

            vz_i_kn = vz_i / 1e3
            vz_j_kn = vz_j / 1e3

            eq_ok   = abs(vz_i_kn + vz_j_kn) <= EQ_TOL
            warning = None

            if not eq_ok:
                warning = (
                    f"Vz_i={vz_i_kn:.4f} kN, Vz_j={vz_j_kn:.4f} kN — "
                    f"expected Vz_i = -Vz_j (diff={vz_i_kn + vz_j_kn:.4f} kN)"
                )
                _warnings.warn(
                    f"[CB {cb_num}] LC '{lc_str}': equilibrium violated — {warning}",
                    stacklevel=2,
                )

            lc_results.append({
                "load_case":      lc_str,
                "Vz_i_kN":        round(vz_i_kn, 4),
                "Vz_j_kN":        round(vz_j_kn, 4),
                "type":           "tension" if vz_i_kn > 0 else "compression",
                "equilibrium_ok": eq_ok,
                "warning":        warning,
            })

        # Forces are in global axis. Cross-bracings run along global Z, so Vz is their axial force.
        # Vz_i > 0 → tension, Vz_i < 0 → compression.
        tension_rows     = [r for r in lc_results if r["Vz_i_kN"] >  0]
        compression_rows = [r for r in lc_results if r["Vz_i_kN"] < 0]

        critical_tension = None
        if tension_rows:
            row = max(tension_rows, key=lambda r: r["Vz_i_kN"])  # largest positive = worst tension
            critical_tension = {
                "load_case": row["load_case"],
                "Vz_i_kN":  row["Vz_i_kN"],
                "Vz_j_kN":  row["Vz_j_kN"],
            }

        critical_compression = None
        if compression_rows:
            row = min(compression_rows, key=lambda r: r["Vz_i_kN"])  # most negative = worst compression
            critical_compression = {
                "load_case": row["load_case"],
                "Vz_i_kN":  row["Vz_i_kN"],
                "Vz_j_kN":  row["Vz_j_kN"],
            }

        left_girder  = chain.get("left_girder", "")
        right_girder = chain.get("right_girder", "")
        girder_pair  = f"{left_girder}-{right_girder}" if left_girder and right_girder else None

        results[str(cb_num)] = {
            "cb_number":            cb_num,
            "member_id":            m_id,
            "girder_pair":          girder_pair,
            "length_m":             length_m,
            "critical_tension":     critical_tension,
            "critical_compression": critical_compression,
            "results":              lc_results,
        }

    _TOOLS_DIR.mkdir(exist_ok=True)
    out_path = _TOOLS_DIR / "crossbracing_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 60)
    print(f"[crossbracing] saved → {out_path.resolve()}")
    print("=" * 60)


def _extract_osdag_summary(result: dict) -> dict:
    if not result:
        return {}
    def _first(*keys):
        for k in keys:
            v = result.get(k)
            if v is not None:
                return v
        return None

    return {
        "section":     _first("section_size.designation", "Optimum.Designation"),
        "capacity_kN": _first("Member.tension_capacity",  "Design.Strength"),
        "efficiency":  _first("Member.efficiency",        "Optimum.UR"),
        "slenderness": result.get("Member.Slenderness"),
        "connection":  "Welded" if "Weld.Type" in result else "Bolted",
    }


def enrich_crossbracing_dump(pair_designs: dict) -> None:
    """
    Update tools/crossbracing_results.json with Osdag design results.

    Parameters
    ----------
    pair_designs : dict
        {
            "G1-G2": {
                "diagonal": {"tension": result_or_None, "compression": result_or_None},
                "chord":    {"tension": result_or_None, "compression": result_or_None},
            }, ...
        }
    """
    out_path = _TOOLS_DIR / "crossbracing_results.json"
    if not out_path.exists():
        return

    with open(out_path) as f:
        data = json.load(f)

    for entry in data.values():
        pair    = entry.get("girder_pair")
        designs = pair_designs.get(pair, {})

        diag_designs  = designs.get("diagonal", {})
        chord_designs = designs.get("chord",    {})

        if entry.get("critical_tension"):
            entry["critical_tension"]["osdag_diagonal"] = _extract_osdag_summary(
                diag_designs.get("tension") or {}
            )
            entry["critical_tension"]["osdag_chord"] = _extract_osdag_summary(
                chord_designs.get("tension") or {}
            )

        if entry.get("critical_compression"):
            entry["critical_compression"]["osdag_diagonal"] = _extract_osdag_summary(
                diag_designs.get("compression") or {}
            )
            entry["critical_compression"]["osdag_chord"] = _extract_osdag_summary(
                chord_designs.get("compression") or {}
            )

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print("=" * 60)
    print(f"[crossbracing] enriched with Osdag designs → {out_path.resolve()}")
    print("=" * 60)


def build_load_effects_cache(result_handler) -> dict:
    """
    Pre-compute per-girder, per-load-case max/min Mz and Vy for all non-vehicle
    load cases.  Called once at the end of design() and stored on the bridge as
    ``_load_effects_cache``.

    Structure returned::

        {
            "G1": {
                "SW":      {"Mz_max": 123.4, "Mz_min": -45.6, "Vy_max": 78.9, "Vy_min": -12.3},
                "BASIC_1": {"Mz_max": ..., ...},
                ...
            },
            "G2": { ... },
        }

    Values are in kNm (Mz) and kN (Vy).
    Vehicle static / moving load cases are excluded so the table only shows
    dead loads and combination load cases.  Any load case whose name contains
    "moving" is also excluded.
    """
    ds = result_handler.ds
    if ds is None:
        return {}

    g_map, _ = result_handler.build_girders(verbose=False)
    classified = result_handler.classify_loadcases()

    exclude = set(
        classified.get("vehicle_static", []) +
        classified.get("vehicle_moving", [])
    )
    lcs = [
        lc for lc in classified.get("all", [])
        if lc not in exclude and "moving" not in str(lc).lower()
    ]

    cache: dict = {}
    gi = 1
    for girder, gdata in g_map.items():
        if girder.startswith("EB"):
            continue
        girder_label = f"G{gi}"
        gi += 1
        elements = gdata["elements"]
        cache[girder_label] = {}
        for lc in lcs:
            mz_vals: list[float] = []
            vy_vals: list[float] = []
            for comp in ("Mz_i", "Mz_j"):
                for eid in elements:
                    try:
                        val = float(ds.sel(Loadcase=lc, Element=eid, Component=comp)["forces"]) / 1000.0
                        mz_vals.append(val)
                    except Exception:
                        pass
            for comp in ("Vy_i", "Vy_j"):
                for eid in elements:
                    try:
                        val = float(ds.sel(Loadcase=lc, Element=eid, Component=comp)["forces"]) / 1000.0
                        vy_vals.append(val)
                    except Exception:
                        pass
            if mz_vals or vy_vals:
                cache[girder_label][str(lc)] = {
                    "Mz_max": round(max(mz_vals), 3) if mz_vals else None,
                    "Mz_min": round(min(mz_vals), 3) if mz_vals else None,
                    "Vy_max": round(max(vy_vals), 3) if vy_vals else None,
                    "Vy_min": round(min(vy_vals), 3) if vy_vals else None,
                }
    return cache


def build_deflections_cache(result_handler) -> dict:
    """
    Pre-compute maximum vertical (y) deflection per girder for:
      - live load only  (load case "1.0 LL")
      - total load      (load case "1.0 DL + 1.0 LL")

    Returns::

        {
            "G1": {"live_mm": 12.3, "total_mm": 25.6},
            "G2": {...},
        }

    Values are in mm (converted from metres, which is OpenSees' native unit).
    Edge-beam girders (EB1/EB2) are skipped; remaining girders are numbered G1…Gn.
    """
    ds = result_handler.ds
    if ds is None:
        return {}
    disp_da = ds.get("displacements")
    if disp_da is None:
        return {}

    all_lcs = [str(lc) for lc in ds.coords["Loadcase"].values]

    # Governing LL case created by create_governing_ll_load_case(partial_safety_factor=1.0)
    ll_case = next(
        (lc for lc in all_lcs if lc.strip() == "1.0 LL"),
        None,
    )
    # DL + LL combination created by create_dl_ll_combination(dl_factor=1.0, ll_factor=1.0)
    dl_ll_case = next(
        (lc for lc in all_lcs if " DL + " in lc and lc.strip().endswith("LL")
         and not lc.startswith(("BASIC_", "SLS_", "SEISMIC_", "ACCIDENTAL_"))),
        None,
    )

    g_map, _ = result_handler.build_girders(verbose=False)
    node_set = set(disp_da.coords["Node"].values)

    def _max_defl_mm(lc_name: str | None, nodes: list) -> float | None:
        if lc_name is None or not nodes:
            return None
        try:
            vals = []
            for node in nodes:
                if node not in node_set:
                    continue
                v = float(disp_da.sel(Loadcase=lc_name, Node=node, Component="y")) * 1000.0
                vals.append(v)
            return round(max(abs(v) for v in vals), 3) if vals else None
        except Exception:
            return None

    cache: dict = {}
    gi = 1
    for girder, gdata in g_map.items():
        if girder.startswith("EB"):
            continue
        girder_label = f"G{gi}"
        gi += 1
        nodes = gdata.get("path", [])
        cache[girder_label] = {
            "live_mm":  _max_defl_mm(ll_case,    nodes),
            "total_mm": _max_defl_mm(dl_ll_case, nodes),
        }

    return cache

def build_forces_summary(result_handler, load_effects_cache: dict) -> dict:
    """
    Build Chapter 4 Table 4.1 data by reusing load_effects_cache for Mz/Vy
    values and fetching only x-locations and reactions from the dataset.

    Called once from compute_load_effects_cache() immediately after
    build_load_effects_cache() so the cache is already available.

    Parameters
    ----------
    result_handler   : PlateGirderAnalysisResults
    load_effects_cache : dict — output of build_load_effects_cache()
        Structure: {girder_label: {lc_name: {Mz_max, Mz_min, Vy_max, Vy_min}}}

    Returns
    -------
    dict with two keys:
        "load_cases" : {lc_name: {max_bm, bm_girder, bm_location,
                                   max_sf, sf_girder, sf_location}}
        "reactions"  : {lc_name: {left_kN, right_kN}}
    """
    ds = result_handler.ds
    if ds is None or not load_effects_cache:
        return {"load_cases": {}, "reactions": {}}

    # Build girder map and node coords once — reused for both location and reactions
    g_map, _ = result_handler.build_girders(verbose=False)
    nodes_coords, _, _ = result_handler.build_grillage_connectivity()

    # Map cache labels (G1, G2, …) → g_map keys (which may be EB1/G2/G3/EB2 when edge_dist > 0)
    # Mirrors the exact skip-and-recount logic in build_load_effects_cache
    cache_label_to_gmap_key = {}
    gi = 1
    for girder_name in g_map:
        if girder_name.startswith("EB"):
            continue
        cache_label_to_gmap_key[f"G{gi}"] = girder_name
        gi += 1

    # Collect the same filtered load case list the cache was built with
    all_lcs = set()
    for girder_data in load_effects_cache.values():
        all_lcs.update(girder_data.keys())

    lc_summary = {}
    rxn_summary = {}

    for lc in all_lcs:
        # ── 1. Derive max_bm / bm_girder and max_sf / sf_girder from cache ──
        global_bm, bm_girder = 0.0, "—"
        global_sf, sf_girder = 0.0, "—"

        for girder_label, lc_data in load_effects_cache.items():
            entry = lc_data.get(lc)
            if not entry:
                continue

            # correct — check both max and min, take whichever has larger magnitude
            mz_max = entry.get("Mz_max")
            mz_min = entry.get("Mz_min")
            mz = mz_max if (mz_max is not None and abs(mz_max) >= abs(mz_min or 0.0)) else mz_min

            vy_max = entry.get("Vy_max")
            vy_min = entry.get("Vy_min")
            vy = vy_max if (vy_max is not None and abs(vy_max) >= abs(vy_min or 0.0)) else vy_min

            if mz is not None and abs(mz) > abs(global_bm):
                global_bm = mz
                bm_girder = girder_label
            if vy is not None and abs(vy) > abs(global_sf):
                global_sf = vy
                sf_girder = girder_label

        # ── 2. Fetch x-location for the winning girder's max Mz and max Vy ──
        bm_location = None
        sf_location = None

        bm_gmap_key = cache_label_to_gmap_key.get(bm_girder)
        sf_gmap_key = cache_label_to_gmap_key.get(sf_girder)

        if bm_gmap_key:
            best_abs = 0.0
            for eid, n1, n2 in g_map[bm_gmap_key].get("element_map", []):
                for comp, node in (("Mz_i", n1), ("Mz_j", n2)):
                    try:
                        val = float(ds.sel(
                            Loadcase=lc, Element=eid, Component=comp
                        )["forces"]) / 1000.0
                        if abs(val) > best_abs:
                            best_abs = abs(val)
                            bm_location = round(nodes_coords[node][0], 3)
                    except Exception:
                        pass

        if sf_gmap_key:
            best_abs = 0.0
            for eid, n1, n2 in g_map[sf_gmap_key].get("element_map", []):
                for comp, node in (("Vy_i", n1), ("Vy_j", n2)):
                    try:
                        val = float(ds.sel(
                            Loadcase=lc, Element=eid, Component=comp
                        )["forces"]) / 1000.0
                        if abs(val) > best_abs:
                            best_abs = abs(val)
                            sf_location = round(nodes_coords[node][0], 3)
                    except Exception:
                        pass

        lc_summary[lc] = {
            "max_bm":      round(global_bm, 3),
            "bm_girder":   bm_girder,
            "bm_location": bm_location,
            "max_sf":      round(global_sf, 3),
            "sf_girder":   sf_girder,
            "sf_location": sf_location,
        }

        # ── 3. Reactions — sum Ra and Rb across all interior girders ──
        left_total  = 0.0
        right_total = 0.0

        for gmap_key in cache_label_to_gmap_key.values():
            element_map = g_map[gmap_key].get("element_map", [])
            if not element_map:
                continue
            eid_start, n1_start, _ = element_map[0]
            eid_end,   _,  n2_end  = element_map[-1]
            try:
                ra = float(ds.sel(
                    Loadcase=lc, Element=eid_start, Component="Vy_i"
                )["forces"]) / 1000.0
                rb = -float(ds.sel(
                    Loadcase=lc, Element=eid_end, Component="Vy_j"
                )["forces"]) / 1000.0
                left_total  += ra
                right_total += rb
            except Exception:
                pass

        rxn_summary[lc] = {
            "left_kN":  round(left_total,  3),
            "right_kN": round(right_total, 3),
        }

    return {"load_cases": lc_summary, "reactions": rxn_summary}


def _dump(data: dict, filename: str = "bridge_plot_data.json") -> None:
    """Write data to tools/<filename>."""
    _TOOLS_DIR.mkdir(exist_ok=True)
    out_path = _TOOLS_DIR / filename
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("=" * 60)
    print(f"[plot data] saved → {out_path.resolve()}")
    print("=" * 60)


def _plot_dev(model, ds_all) -> None:
    """
    Generate interactive plotly HTML files in tools/plots/ using the
    ospgrillage plotting API.  Called only when dev=True.

    Files produced
    --------------
    model.html  — 3-D mesh with node/element labels
    Mz.html     — Bending moment diagram      (plot_bmd,   all members)
    Fy.html     — Shear force diagram         (plot_sfd,   all members)
    Mx.html     — Torsion moment diagram      (plot_tmd,   all members)
    y.html      — Vertical deflection         (plot_def,   all members)
    Fx.html     — Axial force                 (plot_force, all members)
    Fz.html     — Transverse shear            (plot_force, all members)
    My.html     — Minor-axis bending moment   (plot_force, all members)
    """
    import ospgrillage as og

    plots_dir = _TOOLS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Model visualisation ────────────────────────────────────────────
    try:
        fig = og.plot_model(
            model,
            backend="plotly",
            show_node_labels=True,
            show_element_labels=True,
            show=False,
        )
        out = plots_dir / "model.html"
        fig.write_html(str(out))
        print(f"[dev plot] model.html → {out}")
    except Exception as exc:
        print(f"[dev plot] model failed: {exc}")

    # ── 2. Convenience wrappers (Mz, Fy, Mx, y) ──────────────────────────
    convenience = [
        ("Mz.html", og.plot_bmd),
        ("Fy.html", og.plot_sfd),
        ("Mx.html", og.plot_tmd),
        ("y.html",  og.plot_def),
    ]
    for fname, fn in convenience:
        try:
            fig = fn(model, ds_all, backend="plotly", show=False)
            out = plots_dir / fname
            fig.write_html(str(out))
            print(f"[dev plot] {fname} → {out}")
        except Exception as exc:
            print(f"[dev plot] {fname} failed: {exc}")

    # ── 3. Generic plot_force for remaining components ────────────────────
    generic = ["Vx", "Vz", "My"]
    for comp in generic:
        fname = f"{comp}.html"
        try:
            fig = og.plot_force(
                model, ds_all,
                member=og.Members.ALL,
                component=comp,
                show=False,
            )
            out = plots_dir / fname
            fig.write_html(str(out))
            print(f"[dev plot] {fname} → {out}")
        except Exception as exc:
            print(f"[dev plot] {fname} failed: {exc}")
