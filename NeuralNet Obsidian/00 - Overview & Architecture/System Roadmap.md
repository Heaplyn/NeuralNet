# Recognition System Roadmap

## Current

- [x] Dense A-Z letter classification with an independently seeded hard noisy test split.
- [x] IDX loading for MNIST and Fashion-MNIST.
- [x] Held-out top-1 accuracy and tested-versus-guessed sample output.
- [x] AdamW with Fisher/Nesterov and multi-formula routing.
- [x] Online Meta-Loss LR and curvature modulation.
- [x] Direct Taylor predictive LR and curvature modulation.
- [x] Loss-guided hidden-layer growth with safety bounds.
- [x] Runtime recognition settings in `include/ring0/config.hpp`.

## Next Recognition Work

- [ ] Add a true handwritten A-Z dataset such as EMNIST Letters when a valid IDX source is available.
- [ ] Add confusion matrices and per-class precision/recall.
- [ ] Add model checkpointing for recognition runs.
- [ ] Compare fixed-LR, Meta-LR-only, Taylor-only, and blended ablations.
- [ ] Add image augmentation beyond the current dataset-provided variation.

## Longer-Term Engineering

- [ ] Add validation split selection to prevent tuning against the held-out test set.
- [ ] Add reproducible benchmark manifests containing config, dataset counts, and seed.
- [ ] Add CPU/GPU throughput comparisons for image batches.
