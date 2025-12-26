# HAP-WPCN Hybrid SDMA/NOMA with SAC

This repository contains a complete research codebase for training a Soft Actor-Critic (SAC) agent to maximize system energy efficiency (EE) in a HAP-assisted WPCN scenario with Hybrid SDMA/NOMA uplink pairing. The environment supports dynamic user arrival/departure, beam assignment, NOMA pairing, WET/WIT time allocation, and energy causality constraints.

## What This Repository Contains

**Core source code**
- `envs/`: Gymnasium environment, channel models, SINR/rate model, WPCN energy causality
- `train/`: SAC/PPO training scripts and evaluation scripts
- `configs/`: YAML configuration files for experiments

**Experimental artifacts (intentionally included)**
- `logs/`: training logs, metrics CSVs, evaluation curves
- `models/`: SAC/PPO checkpoints and best models
- `figures/`: plots for convergence and analysis

## Research Scope

This code is intended for research use. The SAC policy is trained to maximize **system EE (bps/Hz/J)** under Hybrid SDMA/NOMA constraints with dynamic user activity. The solver-based scheduling in the environment selects candidate user groups, then evaluates WET/WIT allocation and NOMA power splitting to maximize EE.

## Official Results vs. Auxiliary Artifacts

**Official results** refer to metrics and figures reported in the associated paper or thesis.  
**Auxiliary artifacts** include intermediate logs, checkpoints, and exploratory plots in `logs/`, `models/`, and `figures/`.

Only **official results** should be considered authoritative for scientific claims.

## Disclaimer (Read Carefully)

- This repository intentionally includes **all artifacts** (logs, CSVs, checkpoints, and plots) to support transparency and reproducibility.
- These artifacts are **not** necessarily cleaned or curated for publication.
- Different seeds or configurations may lead to different outcomes.
- Use reported paper results as the official benchmark.

## Installation

```bash
pip install -r requirements.txt
```

## Training (SAC)

```bash
python -m train.train_sac --config configs/default_paper_scale.yaml
```

Optional (short run):
```bash
python -m train.train_sac --config configs/default_paper_scale.yaml --total-timesteps 50000
```

## Training (PPO)

```bash
python -m train.train_ppo
```

## Evaluation

```bash
python -m train.eval
```

## Scalability Evaluation (EE vs. Number of WDs)

```bash
python -m train.eval_ee_vs_wd
```

## Plotting

**EE convergence (evaluation curve)**
```bash
python -m scripts.plot_ee_vs_steps
```

**Training SINR distribution**
```bash
python scripts/plot_sinr_training.py --csv logs/runs/<run_id>/train_metrics.csv
```

**Dynamic EE (SAC vs. baseline)**
```bash
python -m scripts.plot_dynamic_ee
```

## Energy Causality Sanity Check

```bash
python -m scripts.check_energy_causality
```

## Reproducibility Notes

- Use the same config and seed for consistent comparisons.
- EE curves depend on solver settings (K/M/tau0 grid), channel scale, and energy parameters.
- This repository is designed for academic verification rather than production deployment.
