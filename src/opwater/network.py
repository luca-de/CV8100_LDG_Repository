"""Load water distribution network data from EPANET files and plot networks."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import wntr


@dataclass
class WDN:
    """Network data used by the hydraulic solvers."""

    A12: sp.csr_matrix  # link-junction incidence matrix (np x nn)
    A10: sp.csr_matrix  # link-reservoir incidence matrix (np x n0)
    net_info: dict  # network dimensions and junction/reservoir names
    link_df: pd.DataFrame  # one row per link
    node_df: pd.DataFrame  # one row per node
    demand_df: pd.DataFrame  # junction demands [m^3/s], one column per time step
    h0_df: pd.DataFrame  # reservoir heads [m], one column per time step


def run_epanet(wn):
    """Run EPANET via WNTR, writing its temporary files outside the working directory."""
    with tempfile.TemporaryDirectory() as tmp:
        return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(tmp) / "temp"))


def load_network_data(inp_file):
    """Read an EPANET .inp file (Hazen-Williams, no tanks or pumps) and return a `WDN`."""
    wn = wntr.network.WaterNetworkModel(str(inp_file))
    results = run_epanet(wn)
    nt = max(int(wn.options.time.duration / wn.options.time.report_timestep), 1)

    net_info = {
        "np": wn.num_links,
        "nn": wn.num_junctions,
        "n0": wn.num_reservoirs,
        "nt": nt,
        "junction_names": wn.junction_name_list,
        "reservoir_names": wn.reservoir_name_list,
    }

    # 'C' is the H-W coefficient for pipes and the loss coefficient for valves; an active
    # throttle control valve's (TCV) setting is its loss coefficient, as in EPANET
    links = []
    for name, link in wn.links():
        if isinstance(link, wntr.network.Pipe):
            link_type, length, n_exp, C = "pipe", link.length, 1.852, link.roughness
        else:
            is_tcv = link.valve_type == "TCV" and link.initial_status != wntr.network.LinkStatus.Closed
            link_type, length, n_exp = "valve", 0.0, 2.0
            C = link.initial_setting if is_tcv else link.minor_loss
        links.append(
            {
                "link_ID": name,
                "link_type": link_type,
                "diameter": link.diameter,
                "length": length,
                "n_exp": n_exp,
                "C": C,
                "node_out": link.start_node_name,
                "node_in": link.end_node_name,
            }
        )
    link_df = pd.DataFrame(links)

    node_df = pd.DataFrame(
        [
            {
                "node_ID": name,
                "elev": getattr(node, "elevation", 0.0),  # reservoirs have no elevation
                "xcoord": node.coordinates[0],
                "ycoord": node.coordinates[1],
            }
            for name, node in wn.nodes()
        ]
    )

    # incidence matrices: -1 at each link's start node, +1 at its end node
    node_idx = {name: i for i, name in enumerate(node_df["node_ID"])}
    A = np.zeros((net_info["np"], len(node_df)))
    for k, row in link_df.iterrows():
        A[k, node_idx[row["node_out"]]] = -1
        A[k, node_idx[row["node_in"]]] = 1
    A12 = sp.csr_matrix(A[:, [node_idx[n] for n in net_info["junction_names"]]])
    A10 = sp.csr_matrix(A[:, [node_idx[n] for n in net_info["reservoir_names"]]])

    # EPANET reports nt + 1 time steps (start and end of the period); keep the first nt
    demand_df = results.node["demand"][net_info["junction_names"]].iloc[:nt].T
    demand_df.columns = [f"demands_{t}" for t in range(1, nt + 1)]
    h0_df = results.node["head"][net_info["reservoir_names"]].iloc[:nt].T
    h0_df.columns = [f"h0_{t}" for t in range(1, nt + 1)]

    return WDN(
        A12=A12,
        A10=A10,
        net_info=net_info,
        link_df=link_df,
        node_df=node_df,
        demand_df=demand_df.rename_axis("node_ID").reset_index(),
        h0_df=h0_df.rename_axis("node_ID").reset_index(),
    )


def plot_network(wdn, plot_type="layout", vals=None, t=None, sensor_nodes=None):
    """Plot the network layout, or node heads / link flows at time step t.

    plot_type: 'layout', 'hydraulic head', 'pressure head' or 'flow'
    vals: head (node_ID, h_1, ...) or flow (link_ID, q_1, ...) results from a solver
    t: time step to plot (1-based, matching the result column names)
    sensor_nodes: junction indices (0-based) to mark and number
    """
    G = nx.from_pandas_edgelist(wdn.link_df, source="node_out", target="node_in", edge_attr="link_ID")
    pos = dict(zip(wdn.node_df["node_ID"], zip(wdn.node_df["xcoord"], wdn.node_df["ycoord"])))
    junctions, reservoirs = wdn.net_info["junction_names"], wdn.net_info["reservoir_names"]

    fig, ax = plt.subplots(figsize=(10, 8))

    if plot_type == "layout":
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=junctions, node_size=10, node_color="black", ax=ax)
    elif plot_type in ("hydraulic head", "pressure head"):
        h = vals.set_index("node_ID").loc[junctions, f"h_{t}"].to_numpy()
        if plot_type == "pressure head":
            h = h - wdn.node_df.set_index("node_ID").loc[junctions, "elev"].to_numpy()
        nx.draw_networkx_edges(G, pos, edge_color="lightgray", ax=ax)
        nodes = nx.draw_networkx_nodes(
            G, pos, nodelist=junctions, node_size=15, node_color=h, cmap="RdYlBu", ax=ax
        )
        fig.colorbar(nodes, ax=ax, label=f"{plot_type.capitalize()} [m]")
    elif plot_type == "flow":
        flow = vals.set_index("link_ID")[f"q_{t}"].abs() * 1000  # [L/s]
        edge_flows = [flow[link_ID] for link_ID in nx.get_edge_attributes(G, "link_ID").values()]
        edges = nx.draw_networkx_edges(G, pos, edge_color=edge_flows, edge_cmap=plt.cm.RdYlBu, width=2, ax=ax)
        fig.colorbar(edges, ax=ax, label="Flow [L/s]")

    nx.draw_networkx_nodes(
        G, pos, nodelist=reservoirs, node_size=80, node_shape="s", node_color="black", ax=ax
    )
    nx.draw_networkx_labels(G, _offset(pos, reservoirs), {n: "Reservoir" for n in reservoirs}, ax=ax)

    if sensor_nodes is not None:
        sensors = [junctions[i] for i in sensor_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=sensors, node_size=60, node_color="red", ax=ax)
        nx.draw_networkx_labels(
            G, _offset(pos, sensors), {n: str(i + 1) for i, n in enumerate(sensors)}, ax=ax
        )

    ax.set_aspect("equal")
    ax.axis("off")


def _offset(pos, nodes):
    """Label positions just above the given nodes."""
    ys = [y for _, y in pos.values()]
    dy = 0.035 * (max(ys) - min(ys))
    return {n: (pos[n][0], pos[n][1] + dy) for n in nodes}
