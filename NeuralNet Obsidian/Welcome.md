# 👋 Welcome to the NeuralNet Knowledge Base & System Vault

This is the central knowledge base and engineering manual for the **RingWrapper Causal Transformer LLM** — a dependency-free, high-performance, from-scratch C++17 language model engine built around a strict 5-ring architectural hierarchy.

This vault documents every mathematical formula, hardware layout, training dynamic, variable symbol, configuration parameter, and code snippet in the codebase. It is written with intellectual honesty: standard engineering is called standard engineering, and experimental heuristics are explained with their intended failure modes, stability guards, and fallback paths.

---

## 🧭 How to Navigate This Vault (If You're Confused or Stuck)

If you are trying to understand the system and feel overwhelmed by the moving parts, here is a mental model and directed reading path based on your exact question:

```
                            ┌────────────────────────────────────────┐
                            │  "What is the system doing right now?" │
                            └───────────────────┬────────────────────┘
                                                │
         ┌────────────────────────┬─────────────┴────────────┬────────────────────────┐
         ▼                        ▼                          ▼                        ▼
┌──────────────────┐    ┌───────────────────┐      ┌───────────────────┐    ┌───────────────────┐
│ "Why did loss    │    │ "What is this C++ │      │ "What are F1-F4   │    │ "What do these    │
│ climb or spike?" │    │ code doing?"      │      │ formulas doing?"  │    │ config knobs do?" │
└────────┬─────────┘    └─────────┬─────────┘      └─────────┬─────────┘    └─────────┬─────────┘
         ▼                        ▼                          ▼                        ▼
[[06 - Reference/       [[06 - Reference/          [[02 - Ring 1/           [[06 - Reference/
Practical Guide:        Beginner Line-by-Line      4-Formula Dynamic        Configuration Values
Why Overshoots Happen]] Code Annotation Guide]]    Weight Physics]]         Master Explainer]]
```

---

## 🔎 Problem-Driven Directory & Troubleshooting Compass

Use this table if something unexpected is happening in the logs or during training:

| Symptom or Question | What is happening | Where to read in depth |
|---|---|---|
| **"How does the C++ code actually work line-by-line?"** | 12 critical snippets annotated line-by-line with plain English beginner explanations. | [[06 - Reference Dictionaries & Practical Guides/Beginner Line-by-Line Code Annotation Guide\|Beginner Line-by-Line Code Annotation Guide]] |
| **"What are the plain English analogies for all concepts?"** | 20 core neural network concepts translated into real-world analogies. | [[06 - Reference Dictionaries & Practical Guides/Master Practical Concepts & Real-World Analogies\|Master Practical Concepts & Real-World Analogies]] |
| **"Why did loss spike or oscillate between 8 and 12?"** | Step size vs. parameter magnitude mismatch, Hessian curvature cliffs ($\lambda_{\max}$), or momentum buffer poisoning after a bad batch. | [[06 - Reference Dictionaries & Practical Guides/Practical Guide - Why Neural Nets Overshoot & How to Stabilize\|Practical Guide: Why Neural Networks Overshoot]] |
| **"What does each config knob in `RuntimeConfig` do?"** | Exhaustive field-by-field breakdown with default values, allowed ranges, and failure symptoms if set too high or low. | [[06 - Reference Dictionaries & Practical Guides/Configuration Values Master Explainer\|Configuration Values Master Explainer]] |
| **"What do all the mathematical symbols & variables mean?"** | Complete dictionary of every symbol ($\theta, g_t, m_t, v_t, S_A, S_B, S_C, \lambda_{\max}$, CoC sorts, etc.). | [[06 - Reference Dictionaries & Practical Guides/Mathematical & Systems Variables Dictionary\|Mathematical & Systems Variables Dictionary]] |
| **"How do I interpret the 11-line console telemetry block?"** | Complete field breakdown, warning flag definitions (`[BAD_BATCH_SKIPPED]`, `[WATCHDOG_TRIGGERED]`, etc.), and diagnosis runbook. | [[06 - Reference Dictionaries & Practical Guides/Training Log Diagnostics & Troubleshooting Runbook\|Training Log Diagnostics & Troubleshooting Runbook]] |
| **"How does attention & RoPE math work step-by-step?"** | Detailed matrix dimension tracing from token IDs to Q/K/V projections, RoPE 2D rotation blocks, ALiBi slopes, and SwiGLU. | [[06 - Reference Dictionaries & Practical Guides/Attention Mechanics Visualized & Head Math\|Attention Mechanics Visualized & Head Math]] |
| **"Why does the loss surface have sharp ravines?"** | Exploration of high-dimensional loss geometry, condition numbers, Rayleigh curvature, and Fisher Natural Gradients. | [[06 - Reference Dictionaries & Practical Guides/Loss Landscapes, Curvature & Optimization Physics\|Loss Landscapes, Curvature & Optimization Physics]] |
| **"How does BPE tokenizer merge bytes and expand vocab?"** | Non-destructive live vocabulary expansion algorithm ($256 \to 10\text{k} \to 32\text{k}$) and semantic cluster initialization. | [[06 - Reference Dictionaries & Practical Guides/Vocabulary Expansion & BPE Subword Mechanics\|Vocabulary Expansion & BPE Subword Mechanics]] |
| **"How do the 5 chrono background threads run concurrently?"** | Lock-free ring buffer streaming and asynchronous Co-Pilot architecture. | [[06 - Reference Dictionaries & Practical Guides/Asynchronous Chrono Co-Pilots & Background Streaming\|Asynchronous Chrono Co-Pilots & Background Streaming]] |

---

## 🧠 Mental Models: Understanding the Architecture

### 1. The 5-Ring Dependency Invariant
The codebase strictly prevents circular dependencies and architectural spaghetti through a unidirectional layered topology:
- **Ring 0 (Foundation)**: Raw matrix math, SIMD vectorization, cache blocking, hardware kernels, scalar loss history, and Taylor polynomial extrapolation.
- **Ring 1 (Physics & Layers)**: Attention (GQA, ALiBi, RoPE), RMSNorm, SwiGLU, AdamW, Meta-Loss Network, and 4-Formula Weight Physics.
- **Ring 2 (Models & Tokenization)**: Full 10-layer `TransformerLM`, BPE subword tokenizer, semantic concept clusters, and KV-cache generation.
- **Ring 3 (Data & Training Engine)**: `TextDataset`, `LLMTrainer`, asynchronous streaming, chrono scheduler, and checkpoint manager.
- **Ring 4 (Application & CLI)**: User interface, CLI interactive runtime, and benchmarking suite (`main.cpp`).

> **Invariant Rule:** Code in **Ring $N$** can include from **Ring $M$** if and only if **$M \le N$**.

---

## 📚 Complete Vault Sitemap

### 📌 Overview & Architecture
- [[00 - Overview & Architecture/Beginner Roadmap & Core Concepts|Beginner Roadmap & Core Concepts]] — Fundamental ML concepts and prerequisite roadmaps.
- [[00 - Overview & Architecture/Architecture Map|Architecture Map]] — Full system diagram and data flow.
- [[00 - Overview & Architecture/Ring Dependency Hierarchy|Ring Dependency Hierarchy]] — Dependency rules and modular boundaries.
- [[00 - Overview & Architecture/System Roadmap|Capabilities Roadmap]] — Evolutionary phases of the engine.

### ⚡ Ring 0 (Core Math & Hardware)
- [[01 - Ring 0 (Core Math & Hardware)/Tensor3D & Matrix Math|Tensor3D & Matrix Math]] — Contiguous memory, cache blocking, and vectorization.
- [[01 - Ring 0 (Core Math & Hardware)/Activation Functions|Activation Functions (GELU, SwiGLU, RMSNorm, Softmax)]] — Mathematical formulas and numerical stability.
- [[01 - Ring 0 (Core Math & Hardware)/Loss Formulations & Calculus|Loss Formulations & Calculus]] — Cross-entropy, Z-loss, and multi-order derivatives.
- [[01 - Ring 0 (Core Math & Hardware)/Loss Derivative Pyramid & Curvature Scaling|Loss Derivative Pyramid & Curvature Scaling]] — Rayleigh quotient curvature preconditioning.
- [[01 - Ring 0 (Core Math & Hardware)/Taylor Loss-Trajectory Predictor|Taylor Loss-Trajectory Predictor]] — Newton–Gregory loss foresight and trajectory rewards.
- [[01 - Ring 0 (Core Math & Hardware)/Taylor Penalty Prediction & Confidence Gating|Taylor Penalty Prediction & Confidence Gating]] — Loss penalization foresight.
- [[01 - Ring 0 (Core Math & Hardware)/Calculus of Constructions & Dependent-Typed Neural Reasoning|Calculus of Constructions & Dependent-Typed Neural Reasoning]] — Dependent type checking engine.
- [[01 - Ring 0 (Core Math & Hardware)/CUDA & Hardware Acceleration Engine|CUDA & Hardware Acceleration Engine]] — Hardware acceleration kernels and OpenMP fallback.
- [[01 - Ring 0 (Core Math & Hardware)/Numerical Stability & NaN Prevention Physics|Numerical Stability & NaN Prevention Physics]] — Logit soft-capping and gradient sanitization.
- [[01 - Ring 0 (Core Math & Hardware)/Config & Telemetry Systems|Config & Telemetry Systems]] — Global runtime configuration and telemetry hooks.

### 🔬 Ring 1 (Layers & Advanced Optimizers)
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi|Attention Mechanics & ALiBi]] — Multi-head, Grouped-Query Attention (GQA), ALiBi falloff, and RoPE.
- [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|4-Formula Dynamic Weight Physics]] — Riemannian Natural Grad, Nesterov, AdamW, and Sparse Decay.
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer|Meta-Neural Loss & Step Optimizer]] — Online meta-learning policy network.
- [[02 - Ring 1 (Layers & Advanced Optimizers)/AdamW, Fisher Metric & Nesterov|AdamW, Fisher Metric & Nesterov]] — Moment updates and empirical Fisher information.
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Training Stability & Fast-Start Descent|Training Stability & Fast-Start Descent]] — Unigram bias init, trust regions, and dimension damping.
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Hierarchical Recursive Thought Layer|Hierarchical Recursive Thought Layer]] — Deliberative cognitive reflection trees.

### 🏛️ Ring 2 (Models & Transformers)
- [[03 - Ring 2 (Models & Transformers)/TransformerLM Decoder (GQA + SwiGLU + RoPE)|TransformerLM Decoder]] — Full 10-layer causal transformer decoder pipeline.
- [[03 - Ring 2 (Models & Transformers)/BPE Tokenizer & Merging Engine|BPE Tokenizer & Merging Engine]] — Byte-pair encoding and merge table rules.
- [[03 - Ring 2 (Models & Transformers)/Semantic VocabManager & Lexicon Clusters|Semantic VocabManager & Lexicon Clusters]] — Semantic concept clusters and neurogenesis.
- [[03 - Ring 2 (Models & Transformers)/Dynamic Adaptive Vocabulary Sizing (10k Scaling)|Dynamic Adaptive Vocabulary Sizing (10k Scaling)]] — Progressive vocabulary expansion.
- [[03 - Ring 2 (Models & Transformers)/Autoregressive KV-Cache Generation|Autoregressive KV-Cache Generation]] — O(1) streaming generation inference.

### 📦 Ring 3 (Data & Training Pipelines)
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]] — Optimization loop, dynamic schedules, and telemetry dashboard.
- [[04 - Ring 3 (Data & Training Pipelines)/Mistake Checkpoint Memory & State Fingerprinting|Mistake Checkpoint Memory & State Fingerprinting]] — Episodic failure memory and similarity throttling.
- [[04 - Ring 3 (Data & Training Pipelines)/Concurrent Chrono Subsystems Engine|Concurrent Chrono Subsystems Engine]] — 5 concurrent background subsystem threads.
- [[04 - Ring 3 (Data & Training Pipelines)/Universal Data Ingestion (CSV, TXT, BIN)|Universal Data Ingestion & Background Streaming]] — Non-blocking data streaming from disk.
- [[04 - Ring 3 (Data & Training Pipelines)/Token Relevancy & Interpolated Parsing|Token Relevancy & Interpolated Parsing]] — Information entropy and non-linear parsing radius.
- [[04 - Ring 3 (Data & Training Pipelines)/Progressive Curriculum & Horizon Growth|Progressive Curriculum & Horizon Growth]] — Context length, dataset ratio, and depth ramp schedules.
- [[04 - Ring 3 (Data & Training Pipelines)/Real-Time Benchmark & Telemetry Dashboard|Real-Time Benchmark & Telemetry Dashboard]] — Live visual dashboard formatting.
- [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide|Debug Log Format & Reading Guide]] — Per-step diagnostic log anatomy.
- [[04 - Ring 3 (Data & Training Pipelines)/Evaluation & Checkpoint Lifecycle|Evaluation & Checkpoint Lifecycle]] — Loss evaluation and checkpoint bundle saving.

### 📐 Theoretical Foundations & Physics
- [[05 - Theoretical Foundations & Physics/Information Geometry & Loss Dynamics|Information Geometry & Loss Dynamics]] — Riemannian manifolds, Fisher metrics, and loss topologies.
- [[05 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information|Riemannian Manifolds & Fisher Information]] — Natural gradient derivations.
- [[05 - Theoretical Foundations & Physics/Adaptive Focal Loss Theory|Adaptive Focal Loss Theory]] — Mathematical proofs for plateau breakout.
- [[05 - Theoretical Foundations & Physics/Calculus of Constructions & Dependent Types|Calculus of Constructions & Dependent Types]] — Formal proof consistency in attention heads.
- [[05 - Theoretical Foundations & Physics/Multi-Order Loss Derivatives & Optimization|Multi-Order Loss Derivatives & Optimization]] — Empirical loss derivatives.

### 📖 Reference Dictionaries & Practical Guides
- [[06 - Reference Dictionaries & Practical Guides/Beginner Line-by-Line Code Annotation Guide|Beginner Line-by-Line Code Annotation Guide]] — 12 critical C++ functions explained line-by-line for beginners.
- [[06 - Reference Dictionaries & Practical Guides/Master Practical Concepts & Real-World Analogies|Master Practical Concepts & Real-World Analogies]] — 20 core concepts in plain English.
- [[06 - Reference Dictionaries & Practical Guides/Mathematical & Systems Variables Dictionary|Mathematical & Systems Variables Dictionary]] — Exhaustive dictionary of every symbol, coordinate, and metric.
- [[06 - Reference Dictionaries & Practical Guides/Configuration Values Master Explainer|Configuration Values Master Explainer]] — Complete guide to all config parameters, default values, and tuning symptoms.
- [[06 - Reference Dictionaries & Practical Guides/Practical Guide - Why Neural Nets Overshoot & How to Stabilize|Practical Guide: Why Neural Networks Overshoot & How RingWrapper Stabilizes Them]] — Deep dive into loss curvature, momentum poisoning, and overshoot physics.
- [[06 - Reference Dictionaries & Practical Guides/Training Log Diagnostics & Troubleshooting Runbook|Training Log Diagnostics & Troubleshooting Runbook]] — Troubleshooting runbook for logs, warning flags, and recovery steps.
- [[06 - Reference Dictionaries & Practical Guides/Attention Mechanics Visualized & Head Math|Attention Mechanics Visualized & Head Math]] — Worked numerical tensor examples of GQA, RoPE, and ALiBi.
- [[06 - Reference Dictionaries & Practical Guides/Loss Landscapes, Curvature & Optimization Physics|Loss Landscapes, Curvature & Optimization Physics]] — Non-Euclidean optimization and 4-formula routing.
- [[06 - Reference Dictionaries & Practical Guides/Vocabulary Expansion & BPE Subword Mechanics|Vocabulary Expansion & BPE Subword Mechanics]] — Live non-destructive subword dictionary expansion.
- [[06 - Reference Dictionaries & Practical Guides/Asynchronous Chrono Co-Pilots & Background Streaming|Asynchronous Chrono Co-Pilots & Background Streaming]] — 5 background co-pilot worker threads.
