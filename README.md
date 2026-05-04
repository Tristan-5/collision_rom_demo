# Collision-Statistical Reduced-Order Model (CS ROM)

This repository contains a Python implementation of a collision-statistical reduced-order model (CS ROM) for velocity fluctuation and diffusion-like variance growth.

The model uses a binomial random-walk representation of molecular collision events, consistent with classical statistical mechanics of diffusion. The variance computed by individual collision steps is derived analytically,

$$
\sigma^2 = 4 p (1-p) \Delta v^2
$$

and is used to parameterize a reduced-order stochastic model without empirical tuning.

Ensemble simulations are compared directly against the analytical diffusion prediction,

$$
Var[v_N] = N \sigma^2
$$

and the expected Monte Carlo convergence behaviour (relative error $\sim M^{-1/2}$) is verified numerically. 

For background on the collision-statistical framework, see:

https://doi.org/10.1515/tp-2026-0017

Rather than resolving spatial flow fields or turbulence dynamics, the approach demonstrates:

- How microscopic collision statistics produce diffusion-like variance growth  
- How ensemble statistics align with analytic transport predictions 
- Recovery of the diffusion coefficient from simulation data
- Foundations of physics-parameterized stochastic reduced-order modeling

This implementation is intentionally lightweight. It does not resolve Navier–Stokes fields or turbulent cascades.

---

## Installation

Install dependencies:

```bash
pip install numpy matplotlib
```
## Reproducibility

All stochastic components accept explicit random seeds. Providing a fixed seed ensures deterministic ensemble behaviour (subject to Numpy version consistency).

## Repository Structure

The main code is contained in the folder `stochastic/`. 

`collision_model.py` 
Generates velocity increments via a binomial collision-event random walk.

`rom.py`
Computes statistical ROM parameters and evaluates predicted variance growth.

`plots.py`
Visualization utilities for ensemble statistics and scaling behaviour.

`demo.py`
End-to-end demonstration scritp reproducing full simulation and comparison against the analytical variance prediction.

## Usage
Run the demo script from the command line:

```bash
python demo.py --N 5000 --p 0.55 --M 300 --seed 1
```

Parameters:

`N`--number of collision events

`M`--number of ensemble realizations

`p`--collision asymettry probability

`seed`--random seed for reproducibility

## Experimental Folder
This folder contains experimental code that is still being developed.





