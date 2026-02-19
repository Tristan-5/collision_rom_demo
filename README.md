# Collision-Statistical Reduced-Order Model Demo

This repository contains a lightweight Python demonstration of a collision-statistical reduced-order model (ROM) for turbulence-like fluctuations. The model illustrates how small-scale stochastic effects can accumulate statistically and produce scale-dependent velocity fluctuations without resolving the full flow field.

The goal is not to replace CFD, but to demonstrate how fast, low-dimensional statistical models can capture key fluctuation behavior and scaling trends relevant to prediction, design, and control.

See https://doi.org/10.1515/tp-2026-0017 for more information on the theory. 

## Reproducibility

All stochastic components support explicit random seeding. Providing a fixed seed ensures deterministic and reproducible results across runs and platforms (subject to NumPy version consistency).

Overview:
- `collision_model.py`: Generates velocity fluctuations from a binomial collision-event random walk
- `rom.py`: Computes ROM parameters and predicts variance growth
- `plots.py`: Visualization utilities
- `demo.py`: End-to-end demonstration script

Usage:
Run the demo script from the command line to simulate a sequence of collision events:
```bash
py demo.py --N 5000 --p 0.55 --seed 1



