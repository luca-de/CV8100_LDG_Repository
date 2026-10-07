"""Hydraulic solvers for water distribution networks."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import wntr

from opwater.network import run_epanet


def loss_coefficients(link_df):
    """Resistance coefficient K and head loss exponent n for every link, as (np x 1) arrays."""
    # Hazen-Williams for pipes, minor loss for valves (length and diameter in m)
    pipe_K = 10.67 * link_df["length"] * link_df["C"] ** -link_df["n_exp"] * link_df["diameter"] ** -4.8704
    valve_K = 8 / (np.pi**2 * 9.81) * link_df["diameter"] ** -4 * link_df["C"]
    K = np.where(link_df["link_type"] == "pipe", pipe_K, valve_K)
    return K.reshape(-1, 1), link_df["n_exp"].to_numpy().reshape(-1, 1)


def hydraulic_solver(wdn, method="nr_schur", tol=1e-5, kmax=50, tol_A11=1e-5):
    """Solve for link flows q [m^3/s] and junction heads h [m] at every time step.

    method: 'nr' solves the full Newton-Raphson system; 'nr_schur' solves for h
        using the Schur complement, then computes q by substitution
    tol: convergence tolerance on the energy and mass balance errors
    kmax: maximum number of iterations per time step
    tol_A11: lower bound on the diagonal of A11 (see Todini and Pilati, 1988)
    """
    A12, A10 = wdn.A12, wdn.A10
    n_links, n_nodes, nt = wdn.net_info["np"], wdn.net_info["nn"], wdn.net_info["nt"]
    K, n_exp = loss_coefficients(wdn.link_df)

    N = sp.diags(n_exp.ravel())
    inv_N = sp.diags(1 / n_exp.ravel())
    I = sp.eye(n_links)
    Z = sp.csr_matrix((n_nodes, n_nodes))

    q = np.zeros((n_links, nt))
    h = np.zeros((n_nodes, nt))

    for t in range(nt):
        d = wdn.demand_df.iloc[:, t + 1].to_numpy().reshape(-1, 1)
        h0 = wdn.h0_df.iloc[:, t + 1].to_numpy().reshape(-1, 1)
        qk = 0.03 * np.ones((n_links, 1))
        A11 = sp.diags(np.maximum(K * np.abs(qk) ** (n_exp - 1), tol_A11).ravel())

        for k in range(kmax):
            if method == "nr":
                J = sp.bmat([[N @ A11, A12], [A12.T, Z]], format="csc")
                b = np.vstack([(N - I) @ A11 @ qk - A10 @ h0, d])
                x = spla.spsolve(J, b).reshape(-1, 1)
                qk, hk = x[:n_links], x[n_links:]
            else:
                inv_A11 = sp.diags(1 / A11.diagonal())
                DD = inv_N @ inv_A11
                S = (A12.T @ DD @ A12).tocsc()
                b = -A12.T @ inv_N @ (qk + inv_A11 @ A10 @ h0) + A12.T @ qk - d
                hk = spla.spsolve(S, b).reshape(-1, 1)
                qk = (I - inv_N) @ qk - DD @ (A12 @ hk + A10 @ h0)

            A11 = sp.diags(np.maximum(K * np.abs(qk) ** (n_exp - 1), tol_A11).ravel())

            energy_err = A11 @ qk + A12 @ hk + A10 @ h0
            mass_err = A12.T @ qk - d
            if max(np.abs(energy_err).max(), np.abs(mass_err).max()) < tol:
                break
        else:
            print(f"Warning: time step t={t + 1} did not converge in {kmax} iterations.")

        q[:, t] = qk.ravel()
        h[:, t] = hk.ravel()

    return results_df(q, wdn.link_df["link_ID"], "link_ID", "q"), results_df(
        h, wdn.net_info["junction_names"], "node_ID", "h"
    )


def epanet_solver(inp_file):
    """Simulate hydraulics with EPANET (via WNTR), returning results in the same format as
    `hydraulic_solver`."""
    wn = wntr.network.WaterNetworkModel(str(inp_file))
    results = run_epanet(wn)
    nt = max(int(wn.options.time.duration / wn.options.time.report_timestep), 1)

    q = results.link["flowrate"][wn.link_name_list].iloc[:nt].T.to_numpy()
    h = results.node["head"][wn.junction_name_list].iloc[:nt].T.to_numpy()
    return results_df(q, wn.link_name_list, "link_ID", "q"), results_df(
        h, wn.junction_name_list, "node_ID", "h"
    )


def simulate_field_conditions(inp_file, d_data, h0_data):
    """Simulate hydraulics with EPANET using measured hourly demands (nn x nt) and
    reservoir heads (n0 x nt). Returns junction heads as an (nn x nt) array."""
    wn = wntr.network.WaterNetworkModel(str(inp_file))
    nt = h0_data.shape[1]
    wn.options.time.duration = (nt - 1) * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.pattern_timestep = 3600
    wn.options.time.report_timestep = 3600

    for i, name in enumerate(wn.reservoir_name_list):
        wn.add_pattern(f"h0_{name}", h0_data[i])
        reservoir = wn.get_node(name)
        reservoir.head_timeseries.base_value = 1.0
        reservoir.head_timeseries.pattern_name = f"h0_{name}"

    for i, name in enumerate(wn.junction_name_list):
        wn.add_pattern(f"d_{name}", d_data[i])
        node = wn.get_node(name)
        node.demand_timeseries_list.clear()
        node.add_demand(base=1.0, pattern_name=f"d_{name}")

    results = run_epanet(wn)
    return results.node["head"][wn.junction_name_list].T.to_numpy()


def results_df(values, ids, id_name, prefix):
    """Results table with an ID column followed by one column per time step."""
    df = pd.DataFrame(values, columns=[f"{prefix}_{t + 1}" for t in range(values.shape[1])])
    df.insert(0, id_name, list(ids))
    return df
