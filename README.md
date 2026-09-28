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
- In multi-intersection scenarios, attention-based critics can exploit interactions between neighbouring signals.
- Under severe congestion, reinforcement learning does not automatically outperform fixed-time control; scalability and training stability remain important limitations.

## Tech stack

- Python
- SUMO / TraCI
- Ray / RLlib
- PyTorch
- PettingZoo / SuperSuit
- Multi-Agent Reinforcement Learning

## Setup

Install SUMO separately and make sure it is available in the environment. Then create an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repository structure

```text
SUMO_MARL/
├── experiments/
│   ├── cologne1/
│   ├── cologne3/
│   ├── resco_ingolstadt1/
│   ├── resco_ingolstadt7/
│   └── resco_ingolstadt21/
├── requirements.txt
└── README.md
```

Generated rollouts and CSV outputs are excluded from version control. Existing result figures are retained as experiment documentation.

## Reproducibility notes

Parallel SUMO rollouts require separate TraCI ports and careful synchronization between simulator instances. Scripts use relative project paths where possible; `SUMO_OUT_DIR` can be set to override the evaluation-output location.

## Author

**Paolo Pangallo**  
M.Sc. Computer Engineering — Artificial Intelligence  
University of Calabria
