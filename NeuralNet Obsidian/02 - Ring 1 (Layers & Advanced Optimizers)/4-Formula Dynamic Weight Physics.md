# Multi-Formula Weight Updates for Recognition

`ring1::MultiFormulaKernel` routes dense classifier parameter updates using a per-element importance score derived from gradient magnitude, parameter magnitude, and empirical Fisher information.

## Why Recognition Uses It

MNIST and Fashion-MNIST contain thousands of input features, while the A-Z classifier has small but noisy inputs. Treating every weight identically can make high-signal pixels and unstable low-signal pixels take the same update. Formula routing gives AdamW a bounded update choice for each parameter element.

## Four Update Families

1. **Natural-gradient style update** for high-importance parameters.
2. **Curvature-scaled Nesterov update** for strongly directional parameters.
3. **Variance-bounded AdamW update** for ordinary parameters.
4. **Inertial sparse decay** for low-importance parameters.

The router is not a separate training loop. It runs inside AdamW after `RingTrainer` has applied gradient clipping and blended Meta-LR/Taylor-LR controls.

## Recognition Flow

```text
Dense gradients -> importance/Fisher score -> formula selection -> AdamW update
```

The active routing mode is controlled by `RuntimeConfig` and `TrainingConfig::enable_multi_formula_opt`. Benchmark output should be interpreted together with held-out accuracy, loss, and Meta/Taylor LR telemetry.

## Related Notes

- [[02 - Ring 1 (Layers & Advanced Optimizers)/AdamW, Fisher Metric & Nesterov|AdamW and Fisher Metric]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer|Meta-Loss Optimizer]]
- [[04 - Ring 3 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|Recognition Benchmarks]]
