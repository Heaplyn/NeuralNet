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

### 🔍 Line-by-Line Beginner Breakdown of RuntimeConfig:
- `bool debug_mode = false;`: A Boolean flag (true/false) controlling whether extra debug metrics and console ASCII tables are printed.
- `bool verbose_thought_chains = true;`: When enabled, prints the delta convergence numbers for every reflection cycle in recursive layers.
- `size_t max_reflection_cycles = 3;`: An unsigned integer specifying the maximum number of multi-pass reasoning reflections allowed per recursive thought layer before passing forward.
- `float thought_residual_momentum = 0.35f;`: A 32-bit floating point number (the `f` suffix denotes a single-precision float literal) damping the vector update across reflection cycles: $h_{\text{new}} = (1 - 0.35) h + 0.35 f(h)$.
- `bool enable_loss_descent_acceleration = true;`: Toggles the adaptive focal gamma modulation when training encounters difficult loss plateaus.
- `float plateau_breakout_loss = 2.0f;`: The loss floor threshold. Once cross-entropy loss falls below 2.0, the training dynamics switch into high-precision fine-tuning mode.
- `float relevance_power_alpha = 1.3f;`: The non-linear exponent used to stretch context windows around rare, high-information tokens ($W(r) \propto r^\alpha$).

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

### 🔍 Line-by-Line Beginner Breakdown of Access Hooks:
- `const auto& cfg`: 
  - `const`: Guarantees this reference cannot modify the configuration.
  - `auto`: C++ automatically deduces the type as `RuntimeConfig`.
  - `&`: Creates an alias reference to the existing singleton object in RAM instead of allocating and copying memory.
- `ring0::get_config()`: Calls the static getter in namespace `ring0` that returns the global `RuntimeConfig` instance.
- `ring0::get_mutable_config().debug_mode = true;`: Retrieves the mutable reference to enable writing from top-level initialization routines.

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
