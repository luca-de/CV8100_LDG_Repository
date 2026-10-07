"""opwater: helper code for the Modelling and Optimization of Water Distribution Systems course."""

from opwater.data import data_dir, network_file
from opwater.hydraulics import (
    epanet_solver,
    hydraulic_solver,
    simulate_field_conditions,
)
from opwater.network import WDN, load_network_data, plot_network

__version__ = "0.1.0"

__all__ = [
    "WDN",
    "data_dir",
    "epanet_solver",
    "hydraulic_solver",
    "load_network_data",
    "network_file",
    "plot_network",
    "simulate_field_conditions",
]
