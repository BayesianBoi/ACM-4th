# ACM Assignment 4: Alien Categorization Model

The project investigates how participants learn to categorize alien stimuli based on visual features in a trial-by-trial learning task. We implement the Generalized Context Model (GCM) to simulate categorization behavior and compare model predictions against both simulated and empirical data.

## Repository Structure

* `data/` → empirical and simulated datasets
* `stan/` → model definitions
* `workbook.Rmd` → main analysis script 

## Model

We model categorization using the Generalized Context Model, an exemplar-based model where:

* Previous stimuli are stored in memory
* Similarity between current and past stimuli is calculated
* Feature attention weights determine how strongly each feature influences categorization
* A sensitivity parameter controls how quickly similarity decreases with distance
* A bias parameter captures baseline response tendencies

The model is implemented in STAN for Bayesian parameter estimation.

## Analyses

This repository includes:

* Simulation of artificial agents under known parameter settings
* Prior predictive checks
* Posterior parameter recovery
* Posterior predictive checks
* Model fitting on empirical participant data

## Authors

Group 2:
Barbora, Daniel, Mattis, Niels, Søren



