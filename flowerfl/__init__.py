"""
FlowerFL - Federated Learning Experiments for Byzantine Robustness Analysis.

This package implements FL simulations for IoT intrusion detection using the
modern Flower framework (flwr run). Supports dual datasets (Edge-IIoT, CIC-IoT2023),
Byzantine-robust strategies (FedAvg, Krum, FedMedian), attack simulations,
and a pluggable Byzantine defense hook architecture.

Usage:
    flwr run .                              # Run default configuration
    flwr run . --run-config "dataset=edge_full strategy=PluginKrum"

Strategies:
    FedAvg, Krum, FedMedian, FedTrimmedAvg  # Built-in Flower strategies
    PluginKrum, PluginTrust, PluginEnsemble  # Custom defense plugins
"""

__version__ = "0.3.0"
