# NeuralNet Recognition Vault

This vault documents the RingWrapper dense recognition system: letter classification, MNIST, and Fashion-MNIST image recognition in C++17.

## Start Here

- [[Welcome|Welcome & Recognition Workflow]]
- [[00 - Overview & Architecture/Architecture Map|Architecture Map]]
- [[04 - Ring 3 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|Recognition Benchmarks]]

## Ring 0: Numerical Foundation

- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Matrix and Tensor Math]]
- [[01 - Ring 0 (Core Math & Hardware)/Activation Functions|Activation Functions]]
- [[01 - Ring 0 (Core Math & Hardware)/Loss Formulations & Calculus|Loss Functions and Gradients]]
- [[01 - Ring 0 (Core Math & Hardware)/Taylor Loss-Trajectory Predictor|Taylor Loss-Trajectory Predictor]]
- [[01 - Ring 0 (Core Math & Hardware)/Taylor Penalty Prediction & Confidence Gating|Taylor Penalty Prediction]]
- [[01 - Ring 0 (Core Math & Hardware)/Numerical Stability & NaN Prevention Physics|Numerical Stability]]
- [[01 - Ring 0 (Core Math & Hardware)/CUDA & Hardware Acceleration Engine|CUDA and CPU Acceleration]]
- [[01 - Ring 0 (Core Math & Hardware)/Config & Telemetry Systems|Runtime Configuration]]

## Ring 1: Layers and Adaptive Optimization

- [[02 - Ring 1 (Layers & Advanced Optimizers)/AdamW, Fisher Metric & Nesterov|AdamW, Fisher Metric, and Nesterov]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer|Meta-Loss Optimizer]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|Multi-Formula Weight Updates]]

## Ring 2: Dense Recognition Model

- [[03 - Ring 2 (Models & Transformers)/Recognition Network|Recognition Network and Dense Layers]]

## Ring 3: Datasets and Training

- [[04 - Ring 3 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|A-Z, MNIST, and Fashion-MNIST Benchmarks]]

## Theory and Practical Guides

- [[05 - Theoretical Foundations & Physics/Information Geometry & Loss Dynamics|Information Geometry and Loss Dynamics]]
- [[05 - Theoretical Foundations & Physics/Multi-Order Loss Derivatives & Optimization|Multi-Order Loss Derivatives]]
- [[05 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information|Fisher Information Geometry]]
- [[06 - Reference Dictionaries & Practical Guides/Loss Landscapes, Curvature & Optimization Physics|Loss Landscapes and Curvature]]
- [[06 - Reference Dictionaries & Practical Guides/Practical Guide - Why Neural Nets Overshoot & How to Stabilize|Why Training Overshoots]]
- [[06 - Reference Dictionaries & Practical Guides/Training Log Diagnostics & Troubleshooting Runbook|Training Diagnostics]]

## Source Entry Point

- `src/main.cpp` runs the A-Z, MNIST, and Fashion-MNIST benchmarks.
- `include/ring3/trainer.hpp` defines the adaptive dense recognition trainer.
- `include/ring0/config.hpp` owns recognition epochs, limits, learning rate, clipping, growth, meta-loss, and Taylor settings.
