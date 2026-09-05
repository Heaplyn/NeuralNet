# 📊 Real-Time Benchmark & Telemetry Dashboard

During deep neural network training, monitoring live hardware throughput, numerical convergence trajectories, and gradient dynamics is critical.

RingWrapper features an integrated **Real-Time Visual Benchmark & Telemetry Dashboard** in `Ring 3` (`LLMTrainer`) that updates live metrics every 50 steps.

> **Console vs. log file.** The dashboard here is the *live* view you watch scroll by; the same numbers plus a lot more state are written to `logs/debug_run_*.txt` in a structured block per step. See [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide|Debug Log Format & Reading Guide]] for the file format.

---

## 📋 Prerequisites

Before reading this, you should be comfortable with:
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]] — the loop that assembles and prints the dashboard
- [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide|Debug Log Format]] — the file version of everything shown here (with deltas)
- [[03 - Ring 2 (Models & Transformers)/Dynamic Adaptive Vocabulary Sizing (10k Scaling)|Dynamic Vocab Sizing]] — the "Active Vocab" line
- [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|4-Formula Dynamic Weight Physics]] — the F1/F2/F3/F4 row
- [[04 - Ring 3 (Data & Training Pipelines)/Evaluation & Checkpoint Lifecycle|Evaluation & Checkpoint Lifecycle]] — the accuracy metrics (Top-1, Top-20, Rank Score, PPL)

---

## 🧠 The Big Picture

At 3,000+ tokens/sec on a small transformer, staring at raw numbers is useless — you need at-a-glance density. The dashboard is designed so a glance answers three questions:

1. **Am I healthy?** Loss/PPL trend, gradient state, watchdog activity.
2. **Am I fast?** Tokens/sec, GFLOPs, ms/step — did something regress?
3. **What is the model actually doing right now?** Layer count, context length, vocab utilization, per-formula weight distribution.

If any of these three surprise you, drop to the debug log for full per-step forensic detail.

---

## 🖥️ Live Dashboard Output Visual Layout

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  ⚡ RINGWRAPPER NEURAL BENCHMARK & REAL-TIME TRAINING TELEMETRY DASHBOARD              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [Step & Progress]     Step   150 / 25000  [████░░░░░░░░░░░░░░░░░░░░] 0.6% | Time: 5.4s | ETA: 14m 32s
│  [Compute Speed]       3,480.2 tok/s | 14.28 GFLOPs/s | 9.19 ms/step (Batch: 32 x 64 = 2048 toks)
│  [Model Dimensions]    10/10 Layers Active | Embed: 128 | Heads: 8 (4 KV GQA) | FFN: 256
│  [Parameters Active]   1,428,960 float32 parameters (5.45 MB)
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [Loss & Convergence]  Loss: 2.3412 (EMA: 2.4105) | PPL: 10.40 | Min Loss: 2.1504
│  [Accuracy Gauges]     Top-1: 42.8% | Top-20: 89.4% | Rank-Score: 71.2%
│  [Dynamic LR & Scale]  LR: 0.003150 (gain: 1.25x) | Penalty: 0.120
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [Adaptive Vocab 10k]  Active Vocab: 4,820 / 10,000 subwords (48.2% capacity)
│  [Multi-Formula Split] F1(Natural): 6.2% | F2(Nesterov): 18.4% | F3(AdamW): 64.1% | F4(Sparse): 11.3%
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Metrics Tracked & Mathematical Definitions

| Dashboard Section | Metric Name | Mathematical Formula | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **Compute Speed** | `tokens_per_second` | $\frac{\text{Total Tokens Processed}}{\text{Elapsed Seconds}}$ | Raw token throughput through forward and backward passes. |
| **GFLOPs/s** | `gflops_estimate` | $\frac{6 \cdot P \cdot (\text{tok/s})}{10^9}$ | Floating-point compute operations executed per second ($6 \times \text{parameters}$ per token for training). |
| **Step Latency** | `ms_per_step` | $\frac{\Delta t_{\text{step}}}{10^{-3}}$ | Average wall-clock time required to compute forward, focal loss, backward, and AdamW update. |
| **Progress & ETA** | `ETA` | $\frac{\text{Elapsed Time}}{\text{Progress}} - \text{Elapsed Time}$ | Real-time estimated remaining time to reach total scheduled steps. |
| **Convergence** | `Loss` & `EMA` | $\alpha \mathcal{L}_t + (1 - \alpha) \text{EMA}_{t-1}$ | Instantaneous and smoothed Cross-Entropy + Z-Loss. |
| **Perplexity** | `PPL` | $e^{\min(\mathcal{L}, 10.0)}$ | Effective branching factor / token uncertainty of the model. |
| **Top-1 Accuracy** | `top1_accuracy` | $\frac{\text{Count}(\hat{y}_1 = y_{\text{true}})}{N} \times 100\%$ | Percentage of tokens where the model's highest logit was the exact target token. |
| **Top-20 Rank Score** | `rank_score` | $\frac{1}{N} \sum \frac{1}{\log_2(\text{Rank}_i + 1)} \times 100\%$ | Continuous soft ranking score honoring near-misses in top candidate positions. |
| **Adaptive Vocab 10k** | `Active Vocab` | $\frac{V_{\text{active}}}{10000} \times 100\%$ | Percentage of the 10,000 token capacity currently active and trained. |
| **Multi-Formula Physics** | `Formula %` | $\frac{\sum [w_i \in \text{Formula } k]}{P} \times 100\%$ | Real-time parameter percentage split across Riemannian ($F_1$), Nesterov ($F_2$), AdamW ($F_3$), and Sparse Decay ($F_4$). |

---

## 🔗 Related Notes
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]]
- [[03 - Ring 2 (Models & Transformers)/Dynamic Adaptive Vocabulary Sizing (10k Scaling)|Dynamic Adaptive Vocabulary Sizing (10k Scaling)]]
- [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|4-Formula Dynamic Weight Physics]]
- [[Index|Return to Master Index]]
