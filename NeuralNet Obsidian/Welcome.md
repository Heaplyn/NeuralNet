# Welcome to the NeuralNet Recognition Vault

This vault documents a focused C++17 recognition engine for labeled letters and images. The active application trains dense classifiers and evaluates them against held-out labels.

## What the Application Recognizes

- **A-Z letters**: 5 x 7 bitmap vectors with a harder, independently seeded noisy test split.
- **MNIST**: 28 x 28 grayscale handwritten digits loaded from IDX files.
- **Fashion-MNIST**: 28 x 28 grayscale clothing images loaded from IDX files.

## Recognition Workflow

```text
Dataset -> normalized Matrix -> dense forward pass -> cross-entropy
        -> backward gradients -> adaptive optimizer -> held-out accuracy
```

The final report shows the tested sample, its expected label, the guessed label, a PASS/FAIL match, and aggregate top-1 accuracy.

## Adaptive Training

The dense trainer combines the reusable RingWrapper optimization components:

1. Ring 0 computes matrix operations, cross-entropy, gradients, and Taylor loss forecasts.
2. Ring 1 applies AdamW, Fisher/Nesterov behavior, multi-formula routing, and online Meta-Loss control.
3. Ring 2 supplies the dense network and loss-guided growth controller.
4. Ring 3 loads datasets, batches samples, trains, and evaluates labels.

The effective learning rate combines the configured base rate with both Meta-LR and direct Taylor-LR multipliers. Taylor foresight also modulates curvature before AdamW updates parameters.

## First Reading Path

1. [[00 - Overview & Architecture/Architecture Map|Architecture Map]]
2. [[04 - Ring 3 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|Recognition Benchmarks]]
3. [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer|Meta-Loss Optimizer]]
4. [[01 - Ring 0 (Core Math & Hardware)/Taylor Loss-Trajectory Predictor|Taylor Predictor]]
5. [[02 - Ring 1 (Layers & Advanced Optimizers)/AdamW, Fisher Metric & Nesterov|AdamW and Fisher Metric]]

## Run the Benchmarks

```powershell
cmake --build build --config Release --target nn_demo
.\build\Release\nn_demo.exe --letter A
```
