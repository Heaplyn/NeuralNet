# Plain English Summary

This index is the master table of contents for the entire neural network knowledge base. It links to every component of the architecture, from Ring 0 (core math and hardware acceleration), Ring 1 (dense layers and meta-optimizers), Ring 2 (dense networks and growth controllers), Ring 3 (computer vision, 4D tensors, 2D convolutions, MaxPool, and auto gradient normalization), Ring 4 (datasets and training pipelines), up to theoretical physics and practical guides.

---

# NeuralNet Recognition & Computer Vision Vault

This vault documents the complete RingWrapper modular neural network and vision system: letter classification, convolutional neural networks (CNNs), MNIST, and Fashion-MNIST recognition in C++17.

## Start Here

- [[Welcome|Welcome & Recognition Workflow]]
- [[00 - Overview & Architecture/Architecture Map|Architecture Map]]
- [[00 - Overview & Architecture/Ring Dependency Hierarchy|Ring Dependency Hierarchy]]
- [[05 - Ring 4 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|Recognition Benchmarks]]

## Ring 0: Numerical Foundation & Math Engine

- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Matrix and Tensor Math]]
- [[01 - Ring 0 (Core Math & Hardware)/Activation Functions|Activation Functions]]
- [[01 - Ring 0 (Core Math & Hardware)/Taylor Loss-Trajectory Predictor|Taylor Loss-Trajectory Predictor]]
- [[01 - Ring 0 (Core Math & Hardware)/Taylor Penalty Prediction & Confidence Gating|Taylor Penalty Prediction]]

## Ring 1: Layers and Adaptive Optimization

- [[02 - Ring 1 (Layers & Advanced Optimizers)/AdamW, Fisher Metric & Nesterov|AdamW, Fisher Metric, and Nesterov]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer|Meta-Loss Optimizer]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|Multi-Formula Weight Updates]]

## Ring 2: Dense Recognition Models

- [[03 - Ring 2 (Models & Transformers)/Recognition Network|Recognition Network and Dense Layers]]

## Ring 3: Computer Vision & CNN Subsystem

- [[04 - Ring 3 (Computer Vision & CNN)/Tensor4D & Spatial Mechanics|Tensor4D & Spatial Mechanics]]
- [[04 - Ring 3 (Computer Vision & CNN)/2D Convolution & Pooling Engine|2D Convolution & Pooling Engine]]
- [[04 - Ring 3 (Computer Vision & CNN)/Modular CNN Architecture|Modular CNN Architecture]]
- [[04 - Ring 3 (Computer Vision & CNN)/CNN Trainer & Auto Gradient Normalization|CNN Trainer & Auto Gradient Normalization]]

## Ring 4: Datasets and Training Pipelines

- [[05 - Ring 4 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|A-Z, MNIST, and Fashion-MNIST Benchmarks]]

## Theory and Practical Guides

- [[06 - Theoretical Foundations & Physics/Information Geometry & Loss Dynamics|Information Geometry and Loss Dynamics]]
- [[06 - Theoretical Foundations & Physics/Multi-Order Loss Derivatives & Optimization|Multi-Order Loss Derivatives]]
- [[06 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information|Fisher Information Geometry]]
- [[07 - Reference Dictionaries & Practical Guides/Loss Landscapes, Curvature & Optimization Physics|Loss Landscapes and Curvature]]

## Source Entry Point

- `src/main.cpp` runs the A-Z Dense, A-Z CNN, MNIST, and Fashion-MNIST benchmarks.
- `include_new/ring3/` defines the CNN container, Conv2D, MaxPool2D, and CNNTrainer with auto-gradient normalization.
- `include_new/ring4/` defines datasets (LetterDataset, MnistDataset, etc.) and `RingTrainer`.
- `include_new/ring0/config.hpp` owns hyperparameters, Taylor settings, and hardware options.
