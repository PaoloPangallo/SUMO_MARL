# Multi-Agent Reinforcement Learning for Traffic Signal Control

**Traffic-signal optimization with SUMO, TraCI and Multi-Agent Reinforcement Learning**

This project evaluates reinforcement-learning approaches for traffic-signal control on urban road networks simulated with **SUMO**. The focus is on how coordination strategies behave as the number of controlled intersections and the degree of interaction between them increase.

## What is evaluated

The experiments compare several controller families:

- **Fixed-time control** as a non-learning baseline;
- **IDQN** and **IPPO** as independent-agent approaches;
- **QMIX** for value decomposition;
- **MAPPO** for centralized training with decentralized execution;
- **MAPPO with attention**;
- **MAPPO with graph attention**.

The project uses scenarios derived from the **RESCO** benchmark for Cologne and Ingolstadt.

## Why MARL?

Traffic intersections are locally controlled but globally coupled. A decision at one signal can change queue length, downstream congestion and spillback at neighbouring intersections.

The project therefore studies not only single-agent performance, but also how learning changes under increasing coordination complexity.

## Experimental setup

Experiments are run in **SUMO** and controlled through **TraCI**. Evaluation considers traffic-level metrics such as:

- mean waiting time;
- mean speed;
- mean queue length.

Scenarios range from single-intersection environments to multi-intersection networks with up to 21 controlled agents.

## Selected findings

The experiments show that performance strongly depends on the traffic regime.

- In some low- or medium-congestion settings, policy-gradient and CTDE approaches substantially improve traffic flow.
- In multi-intersection scenarios, attention- and graph-based critics can exploit interactions between neighbouring signals.
- Under severe congestion, reinforcement learning does not automatically outperform fixed-time control; scalability and training stability remain important limitations.

These failure cases are kept in the analysis because they are useful for understanding when coordination mechanisms help and when they do not.

## Tech stack

- Python
- SUMO
- TraCI
- Ray / RLlib
- PyTorch
- Multi-Agent Reinforcement Learning
- Graph Attention Networks

## Repository structure

```text
SUMO_MARL/
├── experiments/
└── README.md
```

The `experiments/` directory contains the experimental code and scenario-specific runs.

## Reproducibility notes

Parallel SUMO rollouts require separate TraCI ports and careful synchronization between simulator instances. The repository keeps these operational constraints explicit because they materially affect stable MARL experimentation.

## Author

**Paolo Pangallo**  
M.Sc. candidate in Computer Engineering — Artificial Intelligence  
University of Calabria
