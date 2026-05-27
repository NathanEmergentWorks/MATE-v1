MATE v1
MATE (Marker‑Affordance Test Environment) is a minimal reinforcement‑learning benchmark for comparing a gridworld without persistent markers (C0) to a gridworld with episode‑persistent markers (C1). The repository includes a lightweight gridworld environment, a PyTorch PPO agent, training and evaluation scripts, CSV/JSONL logging utilities, and Jupyter notebooks for analysis.

Install
bash
pip install -r requirements.txt
pip install -e .
Train
bash
python -m mate.training.train --env-config configs/env_c0.yaml --agent-config configs/agent_ppo.yaml --condition C0
python -m mate.training.train --env-config configs/env_c1.yaml --agent-config configs/agent_ppo.yaml --condition C1
Checkpoints are saved under runs/raw/<run_id>/.
Episode‑level and training metrics are written to runs/processed/.

Evaluate
bash
python -m mate.training.eval --env-config configs/env_c1.yaml --agent-config configs/agent_ppo.yaml --eval-config configs/eval.yaml --condition C1
Evaluation rollouts and marker summaries are written to:

runs/processed/rollouts.csv

runs/processed/markers.csv

Notebooks
Launch Jupyter from the project root:

bash
jupyter notebook notebooks
The notebooks load data from runs/processed/ and generate:

learning curves

marker usage statistics

intervention comparisons

state–action mutual information

marker role summaries