MATE‑v1: Marker‑Affordance Test Environment
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20404687.svg)](https://doi.org/10.5281/zenodo.20404687)



MATE (Marker‑Affordance Test Environment) is a minimal reinforcement‑learning benchmark designed to test whether adding a single affordance‑enabling substrate feature — persistent agent‑placed markers — produces measurable increases in meaning density, emergent affordances, and new behaviour classes.

MATE compares two otherwise‑identical gridworld conditions:

C0 — No markers: The agent navigates a 10×10 grid with walls and a goal tile.

C1 — Persistent markers: The agent has one additional action, MARK, which places a persistent marker on the current tile for the remainder of the episode.

This environment is the minimal RL instantiation of the Meaning Density Threshold Experiment (MDTE) within the broader Meaning Substrate Theory (MST) research program.

Features
Lightweight 12×12 gridworld with 5×5 egocentric observations

PyTorch PPO agent (configurable)

Deterministic training/evaluation scripts

CSV/JSONL logging for episodes, rollouts, and marker usage

Jupyter notebooks for analysis (learning curves, marker roles, MI estimates)

Clean separation of raw logs (runs/raw/) and processed logs (runs/processed/)

Installation
bash
pip install -r requirements.txt
pip install -e .
Training
Train C0 (no markers):

bash
python -m mate.training.train \
    --env-config configs/env_c0.yaml \
    --agent-config configs/agent_ppo.yaml \
    --condition C0
Train C1 (with markers):

bash
python -m mate.training.train \
    --env-config configs/env_c1.yaml \
    --agent-config configs/agent_ppo.yaml \
    --condition C1
Checkpoints are saved under:

Code
runs/raw/<run_id>/
Processed episode‑level metrics are written to:

Code
runs/processed/
Evaluation
bash
python -m mate.training.eval \
    --env-config configs/env_c1.yaml \
    --agent-config configs/agent_ppo.yaml \
    --eval-config configs/eval.yaml \
    --condition C1
Evaluation outputs:

runs/processed/rollouts.csv

runs/processed/markers.csv

Notebooks
Launch Jupyter from the project root:

bash
jupyter notebook notebooks
The notebooks generate:

learning curves

marker usage statistics

intervention comparisons

state–action mutual information

marker role clustering summaries

Repository Structure
Code
MATE-v1/
│
├── mate/                 # Environment + PPO agent
├── configs/              # YAML configs for env/agent/eval
├── notebooks/            # Analysis notebooks
├── runs/                 # Raw + processed logs (ignored in repo)
├── tests/                # Unit tests
├── README.md
└── LICENSE

-----

Citation

If you use MATE‑v1 in academic work, please cite:

Frick, N. (2026). *MATE‑v1: Marker‑Affordance Test Environment* (v1.1_public). Zenodo. https://doi.org/10.5281/zenodo.20404687

### BibTeX

@software{frick_2026_matev1,
  author       = {Frick, Nathan},
  title        = {MATE‑v1: Marker‑Affordance Test Environment},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1_public},
  doi          = {10.5281/zenodo.20404687},
  url          = {https://doi.org/10.5281/zenodo.20404687}
}

License
This project is released under the MIT License.
