# ⚙️ Runtime Configuration & Telemetry Systems

The `ring0::RuntimeConfig` system provides a zero-overhead, centralized singleton for controlling training physics, acceleration toggles, thought chain recursion, and telemetry hooks.

---

## 🎛️ Configuration Structure

```cpp
struct RuntimeConfig {
    // Debug & Logging Modes
    bool debug_mode = false;
    bool verbose_thought_chains = true;
    bool print_relevance_details = false;

    // Thought Chain Parameters
    size_t max_reflection_cycles = 3;
    float thought_residual_momentum = 0.35f;

    // Loss Acceleration & Plateau Breakout
    bool enable_loss_descent_acceleration = true;
    float plateau_breakout_loss = 2.0f;
    float focal_gamma_max = 2.0f;
    bool fast_track_depth_unlock = true;

    // Token Relevance Algorithm
    bool enable_token_relevance_parsing = true;
    size_t relevance_min_window = 8;
    size_t relevance_max_window = 64;
    float relevance_power_alpha = 1.3f;
};
```

---

## 📡 Live Telemetry Access

Modules anywhere in the system can query or mutate runtime flags without passing config structs through dozens of constructors:

```cpp
// Query configuration
const auto& cfg = ring0::get_config();
if (cfg.debug_mode) {
    // Output deep tensor diagnostics
}

// Mutate configuration
ring0::get_mutable_config().debug_mode = true;
```

---

## 🔗 Related Notes
- [[00 - Overview & Architecture/Architecture Map|Architecture Map]]
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]]
- [[Index|Return to Index]]
