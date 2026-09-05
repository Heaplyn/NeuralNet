# 🪵 Debug Log Format & Reading Guide

Every training run writes a persistent, structured debug file to `logs/debug_run_<YYYYMMDD>_<HHMMSS>.txt`. This note documents its per-step block format so you can read a run after the fact and reconstruct exactly what happened — and, crucially, what *changed* between any two steps.

---

## 📋 Prerequisites

Before reading this, you should be comfortable with:
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]] — where the per-step block is emitted from
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Training Stability & Fast-Start Descent|Training Stability & Fast-Start Descent]] — what "spike step", "trust region", "watchdog" mean
- [[01 - Ring 0 (Core Math & Hardware)/Taylor Loss-Trajectory Predictor|Taylor Loss-Trajectory Predictor]] — the forecast values in the `TAYL` line
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Meta-Neural Loss & Step Optimizer|Meta-Neural Loss & Step Optimizer]] — the `META` line's four control knobs
- [[02 - Ring 1 (Layers & Advanced Optimizers)/4-Formula Dynamic Weight Physics|4-Formula Dynamic Weight Physics]] — the `FORM` line's F1/F2/F3/F4 breakdown

External familiarity: basic understanding of EMA (exponential moving average), L2 gradient norm, learning-rate scheduling.

---

## 🎯 Why a structured log (and not just prints)

An older version dumped one flat line per step. That was fine at 20 steps, useless at 2,000: you could not tell whether a spike was caused by the optimizer, the meta-network, an F4 collapse, or a curriculum event. The rewrite emits a *structured* block per step so a diagnostic session becomes `grep EVENT` or `grep WATCHDOG` on the file, not scrolling.

The design goals were:
1. **State + deltas.** Every quantity that moves gets a `dX=±...` field showing what changed since last step.
2. **Only meaningful lines emit.** If the meta-network didn't fire, the `META` line is omitted. Keeps noise down.
3. **Transitions are events.** State snapshots are boring; *changes* deserve a marker.

---

## 📦 The 11 possible lines per step

Only lines with meaningful content are emitted, so most steps produce ~7 lines.

```
Step   142/1000 | Loss=4.2103 | EMA=4.1876 | LR=0.001250 | |g|=0.847 | Top1=12.3% | PPL=67.4 | Penalty=0.850 | Ctx=32
    LOSS  cur=4.2103 ema_s=4.1876 ema_l=5.0334 min=3.9812 gap_from_min=0.2291 dLoss=+0.0341 dEMA=+0.0089
    LR    applied=1.250e-03 gain=0.987 (watchdog=off) dLR=-0.000012
    GRAD  pre_clip=0.847 clip_thresh=1 clipped=no d|g|=+0.032
    PEN   factor=0.8503 dL/dPen=+0.0142 EMA_dLdP=+0.0067 dPen=+0.0012
    META  scale=1.630x focal_gamma=0.842 dScale=-0.041 dGamma=+0.018
    TAYL  pred_dL=-0.0245 pred_net=-0.1420 reward=+0.0891 conf=0.412 order=4
    FORM  F1(NatGrad)=8.2% F2(NestCurv)=14.1% F3(AdamW)=63.7% F4(SparseDecay)=14.0%
    CURR  layers=10/10 seq_len=32
    WDOG  active=no bad_streak=0 recov_left=0 baseline=0.0000
```

### Line 1 — Summary (grep target)
`Step X/Y | Loss=... | EMA=... | LR=... | |g|=... | Top1=...% | PPL=... | Penalty=... | Ctx=...`
+ optional `[BAD_BATCH_SKIPPED]` / `[WATCHDOG_ACTIVE]` flags.

**Grep this line to plot the run.** Example: `grep "^Step " logs/debug_run_*.txt | awk -F'Loss=' '{print $2}'`.

### Line 2 — `LOSS`
Loss family with deltas.
- `cur` — this step's raw loss (post-ceiling clamp; a spike step will read at the ceiling ~18)
- `ema_s` — short-window EMA (α=0.05 by default)
- `ema_l` — long-window EMA (α=0.01) — used by `compute_loss_shrink` to detect a rising short-term trend
- `min` — best loss seen so far
- `gap_from_min` — how far above the best we are; a growing gap indicates drift
- `dLoss`, `dEMA` — step-to-step deltas

### Line 3 — `LR`
Learning-rate assembly. `applied` is what actually went to the optimizer. `gain` is the `dynamic_lr_gain` state (bounded to keep it from ratcheting). `watchdog=ON x0.25` means the stability watchdog is throttling LR.

### Line 4 — `GRAD`
Pre-clip global L2 gradient norm. Read this alongside the clip threshold and the `clipped=YES/no` flag. Two failure modes to watch for:
- `|g| ≈ 0` step after step → gradients vanishing → optimizer isn't moving weights
- `|g|` frequently *far* above `clip_thresh` → gradients regularly clipped, meaning your true LR is much lower than `applied` (clip is doing the LR-schedule's job)

### Line 5 — `PEN`
Penalty controller. `dL/dPen` is the instantaneous derivative of loss w.r.t. the penalty factor; `EMA_dLdP` is its smoothed version. Signs:
- `EMA_dLdP > 0` → raising penalty made loss worse → controller relaxes penalty
- `EMA_dLdP < 0` → raising penalty helped → controller keeps or boosts it

See [[05 - Theoretical Foundations & Physics/Multi-Order Loss Derivatives & Optimization|Multi-Order Loss Derivatives]].

### Line 6 — `META` (only when active)
The four meta-network knobs: `scale`, `focal_gamma`, and their deltas. Missing means the meta-network is disabled (safe mode) or frozen (watchdog).

### Line 7 — `TAYL` (only when forecast valid)
Taylor loss-trajectory prediction:
- `pred_dL` — predicted next-step loss change (negative = predicted drop)
- `pred_net` — predicted total change over the forecast horizon
- `reward` — discounted trajectory reward fed into the meta-network
- `conf` — forecast confidence in [0,1] (from the per-order trust)
- `order` — effective derivative order actually used this step

### Line 8 — `FORM` (only when 4-formula routing on)
Percentage of parameters routed through each update formula. Diagnostic reads:
- **F4 > 85% for many steps** → optimizer is pruning almost everything → capacity starvation → forced-neurogenesis trigger will fire
- **F1 near 0%** → nothing is being treated as high-importance → check the Fisher/Taylor salience computation

### Line 9 — `CURR`
Curriculum state — active layers vs total, current sequence length. Transitions render inline: `[LAYERS 4->10]`, `[CTX 32->64]`. Depth is monotonic, so you should never see `[LAYERS 10->4]` — if you do, it's a bug.

### Line 10 — `WDOG`
Watchdog state. Only emitted when the watchdog is active or its bad-step counter is nonzero.
- `bad_streak` — consecutive steps where `loss > ema_short + rise_gap`
- `recov_left` — remaining minimum-freeze steps before recovery is even checked
- `baseline` — the EMA loss the moment the watchdog fired

### Line 11 — `EVENT` (only on transitions)
The most important line for diagnosis. Only emits when something notable *changed* this step. Possible markers:
- `[WATCHDOG_TRIGGERED@<baseline>]` — sustained loss rise, freezing adaptive modules
- `[WATCHDOG_RECOVERED]` — loss returned to within `recover_gap` of baseline
- `[SPIKE_STEP_SKIPPED loss>N]` — this step's forward diverged; backward/optimizer bypassed
- `[ZERO_GRAD]` — gradients came back exactly zero (usually a sign of NaN sanitization)
- `[HUGE_GRAD_CLIPPED]` — pre-clip norm was more than 5× the clip threshold

**Grep `EVENT` to get a clean timeline of everything unusual in a run.**

---

## 🧭 A short reading recipe

Given a failing run, this is the fastest sequence:

1. `grep EVENT logs/debug_run_XXX.txt | head` — timeline of watchdog fires, spike-skips, huge grads
2. `grep "^Step " ... | awk -F'Loss=' '{print $2}' | awk -F' ' '{print $1}'` — plot loss
3. Around any spike or watchdog fire, look at the block **immediately before** the event and diff it against a block 20 steps earlier. The `dX=` fields on each line pinpoint what drifted.
4. If `|g|` was already large-and-growing before the spike, the divergence is in the model's weights themselves (bad LR, bad init, exploding activations). If `|g|` was small but a single step spiked, the divergence is a bad batch / edge-case token.
5. Check `FORM` — if F4 approached 100% before the spike, the model ran out of usable capacity and pruning saturation should have forced neurogenesis (see [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]]).

---

## ⚙️ Where the log is written

`ring0::log_debug_file(tag, body)` in [[01 - Ring 0 (Core Math & Hardware)/Config & Telemetry Systems|Config & Telemetry Systems]] appends `[<time>] [<tag>] <body>\n` to the run's file. The trainer calls it once per step from the `on_step` block in `LLMTrainer::train`. Cost is negligible compared to a training step, and the file is opened once for the session (not per-step).

---

## 🔗 Related Notes
- [[04 - Ring 3 (Data & Training Pipelines)/LLMTrainer Architecture|LLMTrainer Architecture]] — where the block is assembled
- [[04 - Ring 3 (Data & Training Pipelines)/Real-Time Benchmark & Telemetry Dashboard|Real-Time Benchmark & Telemetry Dashboard]] — the live console view (uses the same numbers)
- [[02 - Ring 1 (Layers & Advanced Optimizers)/Training Stability & Fast-Start Descent|Training Stability & Fast-Start Descent]] — for interpreting the `WDOG`/`EVENT` lines
- [[01 - Ring 0 (Core Math & Hardware)/Config & Telemetry Systems|Config & Telemetry Systems]] — the underlying `log_debug_file` helper
- [[Index|Return to Master Index]]
