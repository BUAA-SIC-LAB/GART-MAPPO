# GRAT-MAPPO Project

This project implements a Multi-Agent Reinforcement Learning (MARL) solution for drone formation control and gate traversal tasks. It is built upon the [Omnidrones](https://github.com/btx0424/omnidrones) framework, leveraging NVIDIA Isaac Sim for high-fidelity physics simulation.

The core contribution is the **GRAT-MAPPO** (Graph Recurrent Attention Network - Multi-Agent Proximal Policy Optimization) algorithm, designed to enable robust formation control and coordinated maneuver through complex environments.

## Overview

- **Framework**: Omnidrones (PyTorch + Isaac Sim)
- **Algorithm**: `mappo_graph_attention` (MAPPO with Graph Attention and Recurrent units)
- **Task**: `FormationGateTraversal` (Drones navigating gates while maintaining formation)

## Prerequisites

This project requires:
- **NVIDIA Isaac Sim**: Compatible version as required by Omnidrones.
- **Python**: 3.8+
- **Omnidrones**: The base framework code is included in this repository.

## Usage

To start training the drone formation policy, run the following command from the project root:

```bash
python -u train.py task=FormationGateTraversal algo=mappo_graph_attention
```

### Common Arguments
- `headless=true`: Run simulation without the GUI (useful for remote servers).
- `wandb.mode=disabled`: Disable WandB logging if not needed.
- `sim.num_envs=...`: Set the number of parallel environments.

Example:
```bash
python -u train.py task=FormationGateTraversal algo=mappo_graph_attention headless=true
```

## Task Description: FormationGateTraversal

The `FormationGateTraversal` task challenges a team of drones to fly through a series of gates while maintaining a specific geometric formation.

- **Objective**: Navigate through gates without collision while keeping formation.
- **Formations**: Defined in `omni_drones/envs/formation_gate_traversal.py`, supports various shapes like Tight, Wide, V-formation, Wedge, etc.
- **Reward**: Based on progress, formation maintenance, alignment, and avoiding collisions.

## Algorithm: GRAT-MAPPO

**GRAT-MAPPO** extends the standard MAPPO algorithm by incorporating:

1.  **Graph Neural Networks (GNN)**: To model the interaction between agents (drones). Each drone is a node, and edges represent communication or proximity.
2.  **Attention Mechanism**: To dynamically weight the importance of neighboring drones' information.
3.  **Recurrent Units (GRU)**: To handle partial observability and remember past states/trajectories.

Key configuration parameters for the algorithm can be found in `cfg/algo/mappo_graph_attention.yaml`, such as:
- `gnn_layers`: Number of graph propagation layers.
- `gnn_heads`: Number of attention heads.
- `seq_len`: Sequence length for RNN training.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
