# Classical Learning

A lightweight Python project implementing **classical neural and adaptive learning algorithms from scratch**.
The goal is to understand the **foundations of neural computation** by implementing textbook models. 
This repository contains implementations of several classical learning algorithms used in early neural computation and soft computing literature.

## Implemented / Planned Models

* Hebbian Learning
* Perceptron
* Adaline
* Madaline
* Multilayer Perceptron (Backpropagation)
* Competitive Networks
* Kohonen Self-Organizing Map
* MAXNET

## Project Structure

```
classical_NNs/
│
├── models/        # Learning algorithm implementations
├── dataset/       # Small experimental datasets (logic gates, etc.)
├── activations/   # Activation functions
├── utils/         # Helper utilities
├── experiments/   # Model testing scripts
│
└── main.py        # Entry point
```

## Running Experiments

Example:

```
uv run python -m experiments.test_hebb
```

This runs the Hebbian learning experiment on a sample dataset.

## Motivation

Modern machine learning libraries and frameworks abstract away many of the mechanisms or the core algorithms behind neural learning.
This project focuses on building those mechanisms directly, providing insight into how classical neural algorithms operate internally.


Install dependencies:

```
pip install -r requirements.txt
```

## Status

WIP  — models and experiments are being added progressively.
