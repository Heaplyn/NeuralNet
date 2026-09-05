# ⚙️ Runtime Configuration & Telemetry Systems

`ring0::RuntimeConfig` is a process-wide singleton that carries every runtime toggle that isn't part of a specific model / trainer config struct: debug logging, thought-chain recursion parameters, loss-descent acceleration knobs, and the token-relevance parsing hyperparameters. It's paired with `ring0::log_debug_file`, a single-file logging sink that every ring writes to.

> **In one sentence:** one shared place to ask "are we in debug mode?" / "should this optional adaptive feature fire?" / "write this line to the run log", without threading config pointers through every function signature.

---

## 📋 Prerequisites

Before reading this, you should be comfortable with:
- The [[00 - Overview & Architecture/Ring Dependency Hierarchy|Ring Dependency Hierarchy]] — this singleton lives in Ring 0 so *every* ring may read it
- Basic singleton pattern (why we use one here, and what breaks if two threads mutate it simultaneously)
- [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide|Debug Log Format]] — where the per-step lines this sink writes end up

---

## 🎛️ Configuration Structure

Every field is a plain scalar with a sensible default, so the singleton is safe to use without any explicit initialization.

```cpp
struct RuntimeConfig {
    // --- Debug & Logging Modes ---
    bool debug_mode = false;                ///< Verbose console + more log lines
    bool verbose_thought_chains = true;     ///< RecursiveLayer prints its per-step delta table
    bool print_relevance_details = false;   ///< TextDataset dumps its relevance windows

    // --- Thought-Chain Parameters (RecursiveLayer) ---
    size_t max_reflection_cycles = 3;       ///< how many self-reflection passes per query
    float  thought_residual_momentum = 0.35f;///< damping across reflection cycles

    // --- Loss Acceleration / Plateau Breakout ---
    bool  enable_loss_descent_acceleration = true;
    float plateau_breakout_loss = 2.0f;     ///< engage acceleration once loss drops below this
    float focal_gamma_max = 2.0f;           ///< ceiling on hand-computed focal γ (non-meta path)
    bool  fast_track_depth_unlock = true;   ///< allow depth ramp to bypass step boundaries

    // --- Token Relevance Algorithm ---
    bool   enable_token_relevance_parsing = true;
    size_t relevance_min_window = 8;
    size_t relevance_max_window = 64;
    float  relevance_power_alpha = 1.3f;
};
```

### Why it's a singleton and not a config struct
The five rings need to consult these flags in places where threading a config object would be unfeasible — for example, the numerical stability guards in Ring 0 tensor operations want to know `debug_mode` so they can log when they clamp a NaN, without every callee taking a `RuntimeConfig&` parameter. Making it a singleton keeps the API surface clean at the cost of one global. Access is documented; mutation is intentionally restricted to `get_mutable_config()` so grep can find every write.

### What is NOT in RuntimeConfig
Anything model-shape or training-run specific (batch size, LR, layer count, safe-mode, warmup ratio, etc.) belongs to `TransformerConfig` or `LLMTrainingConfig`, not here. Rule of thumb: if it's the same for every run of this binary, it can be in `RuntimeConfig`; if it changes per invocation, it belongs to a per-run config.

---

## 📡 Live Telemetry Access

Read the config anywhere; mutate it only from `main.cpp` / debug hooks.

```cpp
// Read (const, cheap, thread-safe for reads):
const auto& cfg = ring0::get_config();
if (cfg.debug_mode) { /* extra diagnostics */ }

// Write (should only happen from top-level app code):
ring0::get_mutable_config().debug_mode = true;
```

If you find yourself calling `get_mutable_config()` from inside a training step, that's usually a design smell — pass the value in as a parameter or fold it into the trainer's own state.

---

## 🪵 The Debug Log Sink (`log_debug_file`)

`log_debug_file(const char* tag, const std::string& body)` appends one entry to the current run's debug file (`logs/debug_run_<timestamp>.txt`). Format:

```
[HH:MM:SS.mmm] [TAG] body...
```

- **Opened lazily** the first time it's called each run, closed at program exit.
- **Thread-safe** via a small internal mutex (protects `ofstream` writes).
- **Cheap** — one `write` per call, no formatting overhead beyond what the caller already did.

Callers used across the codebase:
| Tag | Emitted from | Meaning |
|---|---|---|
| `TRAIN_STEP` | `LLMTrainer::train` | The 11-line per-step block; see [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide\|Debug Log Format]]. |
| `SESSION` | `main.cpp` startup | One-time header (git commit, config summary, device info). |
| `VOCAB` | `VocabManager` | Vocabulary expansion events. |
| `NEURO` | `TransformerLM::expand_capacity` | Neurogenesis / forced-neurogenesis fires. |
| `EVAL` | `LLMTrainer::evaluate_loss` | Periodic evaluation snapshots. |
| `CHECKPOINT` | `TransformerLM::save_checkpoint_bundle` | Milestone saves. |

Grep-friendly on purpose — a diagnostic session is usually `grep TRAIN_STEP` or `grep NEURO` on the file rather than opening it in an editor.

---

## 🧪 Debug-Mode Escalations

Turning on `debug_mode` doesn't just add prints — it also unlocks stricter internal checks:
- `Matrix` operations verify shape compatibility with asserts instead of relying on debug-build asserts alone.
- `LossDerivativePyramid` prints its per-layer variance table.
- `TaylorTrajectoryPredictor` logs its per-order trust factors each step.
- `RecursiveLayer` writes a Δ-magnitude / cosine-similarity table per reflection cycle.

Cost is nonzero (extra I/O, extra checks) — don't leave `debug_mode` on for long runs.

---

## 🔒 Threading Notes

- **Reads are safe** across threads (POD scalars, no locks).
- **Writes are NOT safe** while other threads are reading; only mutate before spawning worker threads or from the OMP master thread between steps.
- The debug-log mutex is separate from the config singleton — logging works fine from parallel regions.

---

## 🔗 Related Notes
- [[00 - Overview & Architecture/Architecture Map|Architecture Map]] — where the singleton sits in the system
- [[04 - Ring 3 (Data & Training Pipelines)/Debug Log Format & Reading Guide|Debug Log Format & Reading Guide]] — what the per-step lines this sink writes actually contain
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]] — biggest consumer of both the config flags and the log sink
- [[Index|Return to Index]]
