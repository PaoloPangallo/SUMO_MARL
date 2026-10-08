# Multi-Agent Reinforcement Learning for Traffic Signal Control

**Traffic-signal optimization with SUMO, TraCI and Multi-Agent Reinforcement Learning**

This project evaluates reinforcement-learning approaches for traffic-signal control on urban road networks simulated with **SUMO**. The focus is on how coordination strategies behave as the number of controlled intersections and the degree of interaction between them increase.

**Motivation:** A traffic signal makes local decisions, but its actions alter downstream queues and congestion. The project explores whether and how adaptive learning and coordination help as the traffic network becomes more interconnected, **without assuming RL is always superior to a traditional controller**.

### Architecture and project walkthrough

- [**System design** — simulation/agent architecture, two editable Mermaid diagrams, algorithm families, trade-offs and experimental limits](docs/SYSTEM_DESIGN.md)
- [**Guida italiana** — motivazione, spiegazione chiara, presentazione da colloquio e domande tecniche](docs/PROJECT_WALKTHROUGH_IT.md)

The experiment scripts remain organized by traffic scenario. The documentation distinguishes **conceptual CTDE/attention designs** from the properties currently verified by the code.

## What is evaluated

The experiments compare several controller families:

- **Fixed-time control** as a non-learning baseline;
- **IDQN-labelled** and **IPPO-labelled** DQN/PPO experiments with shared RLlib policy mapping across agents;
- **QMIX** for value decomposition;
- **MAPPO-oriented** PPO configurations with custom critic classes (end-to-end centralized critic wiring still needs verification);
- **PPO with attention-oriented critic models**;
- **GAT-style critic experiments** using attention over agent vectors (no explicit road-graph message passing demonstrated).

The project references Cologne and Ingolstadt scenarios from the **RESCO** suite. The required traffic-network and route XML files under `nets/RESCO/` are **not tracked in this repository**.

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

Historical figures suggest that performance depends on the traffic regime. These plots are exploratory and do not constitute a frozen-checkpoint, held-out multi-seed benchmark.

- Some plots illustrate promising behavior for PPO-based controllers under certain scenario configurations; the CTDE property itself requires a verified critic data flow.
- Attention-based critic variants explore whether combining signals' observations could capture interactions across intersections, but the inspected runners do not establish that these critics are trained on global observations.
- Under severe congestion, reinforcement learning does not automatically outperform fixed-time control; scalability and training stability remain important limitations.

## Tech stack

- Python
- SUMO / TraCI
- Ray / RLlib
- PyTorch
- PettingZoo / SuperSuit
- Multi-Agent Reinforcement Learning

## Setup

Install a compatible SUMO version and supply the RESCO network/route XML files under `nets/RESCO/` separately. These inputs are **not included** in the repository. Pin compatible versions of SUMO, SUMO-RL, Python and legacy RLlib before attempting reproduction. Then create an isolated Python environment:

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

Parallel SUMO rollouts require separate TraCI ports and careful synchronization between simulator instances. Scripts use relative project paths where possible; `SUMO_OUT_DIR` is supported by `experiments/valutazione_csv.py` to select **its plot input directory**, not as a universal override in every training runner.

**Evaluation caveat:** several comparison scripts select the episode with the lowest mean waiting time from training artifacts. That is an exploratory diagnostic, **not** held-out evaluation. Fixed-time and RL scripts also require a harmonized metric definition before numerical comparisons are treated as definitive.

**Architecture caveat:** many DQN/PPO runners map all traffic signals to a single `shared` policy, and defining a custom critic does not by itself guarantee that a centralized value model receives global observations or contributes to training. See the [system design](docs/SYSTEM_DESIGN.md) for the implementation boundaries.

## Author

**Paolo Pangallo**  
M.Sc. Computer Engineering — Artificial Intelligence  
University of Calabria
