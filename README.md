## Collision-Statistical Reduced-Order Model Demo

This repository contains a lightweight Python demonstration of a collision-statistical reduced-order model (ROM) for turbulence-like fluctuations. The model illustrates how small-scale stochastic effects can accumulate statistically and produce scale-dependent velocity fluctuations without resolving the full flow field.

The goal is not to replace CFD, but to demonstrate how fast, low-dimensional statistical models can capture key fluctuation behaviour and scaling trends relevant to prediction, design and control. 

This implementation is intentionally minimal and designed for conceptual demonstration rather than high-fidelity simulation.

For theoretical background, see:
https://doi.org/10.1515/tp-2026-0017

## Installation

Clone the repository and install the required dependencies:
```bash
pip install numpy matplotlib


## Reproducibility

All stochastic components support explicit random seeding. Providing a fixed seed ensures deterministic and reproducible results accross runs and platforms (subject to numpy version).

## Repository Structure

`collision_model.py` generates velocity fluctuations from a binomial collision-event random walk.

`rom.py` computes ROM parameters and predicts variance growth.

`plots.py` gives visualization utilities.

`demo.py` gives end to end demonstration script.

## Usage

Run the demo script from the command line

```bash
py demo.py --N 5000 --p 0.55 --seed 1



