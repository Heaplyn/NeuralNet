# 👋 Welcome to the NeuralNet Vault

This is the knowledge base for the **RingWrapper Causal Transformer LLM** — a C++17, dependency-free, from-scratch language-model engine built around a strict 5-ring architectural hierarchy. The vault documents the code, the mathematics, the training dynamics, and — importantly — the *reasoning* behind each design choice, including the parts that are grounded engineering vs. the parts that are experimental heuristics.

Nothing here is written to sell you on the project. Where a mechanism is a plain re-application of a known technique, that's what the note says. Where a heuristic is experimental and could go wrong, the note says that too.

---

## 🚀 Where to start

- **If you want the map first:** [[Index|Master Index]] lists every note grouped by Ring.
- **If you want the "how does it fit together" view:** [[00 - Overview & Architecture/Architecture Map|Architecture Map]].
- **If you're chasing a specific problem:** the section below routes you.

---

## 🔎 Problem-driven starting points

| I want to understand… | Start here |
|---|---|
| Why training loss starts around 7 instead of the uniform 9.2 floor | [[02 - Ring 1 (Layers & Advanced Optimizers)/Training Stability & Fast-Start Descent\|Training Stability & Fast-Start Descent]] |
| Why loss sometimes climbs mid-run and how the engine catches it | Same note ↑ (watchdog + spike-skip sections) |
| The 4-formula routing (F1/F2/F3/F4) I see on the dashboard | [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics\|4-Formula Dynamic Weight Physics]] |
| The `META:` / `γ` line on the log | [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer\|Meta-Neural Loss & Step Optimizer]] |
| The `TAYL:` line and "trajectory reward" | [[01 - Ring 0 (Core Math & Hardware)/Taylor Loss-Trajectory Predictor\|Taylor Loss-Trajectory Predictor]] |
| The per-step `logs/debug_run_*.txt` block format | [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide\|Debug Log Format & Reading Guide]] |
| Attention (GQA, RoPE, ALiBi) | [[02 - Ring 1 (Layers & Advanced Optimizers)/Attention Mechanics & ALiBi\|Attention Mechanics & ALiBi]] |
| The training loop end-to-end | [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture\|LLMTrainer Architecture]] |
| The Fisher / natural-gradient / Riemannian language | [[05 - Theoretical Foundations & Physics/Riemannian Manifolds & Fisher Information\|Riemannian Manifolds & Fisher Information]] |

---

## 🧱 The Ring architecture in one paragraph

Ring 0 is pure math and hardware primitives (tensors, activations, loss, CUDA/OpenMP). Ring 1 adds neural layers and optimizers (attention, embeddings, AdamW, meta-network). Ring 2 assembles full models (`TransformerLM`, BPE tokenizer, vocab manager). Ring 3 is training pipelines, datasets, and checkpoint I/O. Ring 4 is the CLI application (`main.cpp`). A module in ring N may only include from rings ≤ N — full details in [[00 - Overview & Architecture/Ring Dependency Hierarchy|Ring Dependency Hierarchy]].

---

## 📎 Two conventions used throughout the vault

1. **Prerequisites blocks** — most notes start with a short list of concepts and other notes you should be comfortable with first. Skip them at your own risk, but they're there to save you from bouncing between pages.
2. **"Relation to prior work" callouts** — where a mechanism is close to something well-known in the literature (learned optimizers, log-frequency init, Newton–Gregory extrapolation, natural gradient), the note says so honestly instead of framing it as novel.

---

## 🔗 Next
- [[Index|Master Index]] — enumerate every note in the vault.
