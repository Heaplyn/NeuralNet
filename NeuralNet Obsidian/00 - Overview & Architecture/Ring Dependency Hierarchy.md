# 🔒 Strict Ring Dependency Hierarchy

The codebase adheres to the strict **Layered Ring Architectural Rule**:
> **Core Constraint**: A module in **Ring $N$** can require/include modules from **Ring $M$** if and only if **$M \le N$**. Violating this constraint introduces circular dependencies, tight coupling, and silent compilation/loading failures.

---

## 📊 Dependency Topology Matrix

| Ring Layer | Directory Namespace | May Depend On | Prohibited From Accessing | Key Responsibilities |
| :--- | :--- | :--- | :--- | :--- |
| **Ring 0** | `ring0::` / `include/ring0` | Core STL only | Ring 1, Ring 2, Ring 3, Ring 4 | Pure mathematical primitives, `Tensor3D`, `Matrix`, `Loss`, `CUDA`, `Config` |
| **Ring 1** | `ring1::` / `include/ring1` | Ring 0, Ring 1 | Ring 2, Ring 3, Ring 4 | Neural network layers, attention, `AdamW`, `MetaLossOptimizer`, `MultiFormulaKernel`, `RecursiveLayer` |
| **Ring 2** | `ring2::` / `include/ring2` | Ring 0, Ring 1, Ring 2 | Ring 3, Ring 4 | Full model architectures, `TransformerLM`, `Tokenizer`, `VocabManager` |
| **Ring 3** | `ring3::` / `include/ring3` | Ring 0, Ring 1, Ring 2, Ring 3 | Ring 4 | Datasets (`TextDataset`, `MNIST`), Training loops (`LLMTrainer`), Checkpointing |
| **Ring 4** | `ring4::` / Application | All Rings (0 to 4) | None | CLI endpoints, streaming user interfaces, benchmarks, evaluation apps |

---

## 🛡️ Ring Invariance Enforcements

```mermaid
graph TD
    subgraph R0["Ring 0 (Root Level)"]
        tensor["tensor.hpp"]
        loss["loss.hpp"]
        cuda["cuda_backend.hpp"]
        config["config.hpp"]
        act["activations.hpp"]
    end

    subgraph R1["Ring 1 (Layer Level)"]
        layer["layer.hpp"]
        attn["attention.hpp"]
        adamw["adamw.hpp"]
        meta["meta_loss_optimizer.hpp"]
        multiformula["multi_formula_optimizer.hpp"]
        thought["recursive_layer.hpp"]
    end

    subgraph R2["Ring 2 (Model Level)"]
        transformer["transformer_lm.hpp"]
        tokenizer["tokenizer.hpp"]
        vocab["vocab_manager.hpp"]
        net["neural_net.hpp"]
    end

    subgraph R3["Ring 3 (Trainer Level)"]
        trainer["llm_trainer.hpp"]
        dataset["text_dataset.hpp"]
        mnist["mnist_dataset.hpp"]
    end

    subgraph R4["Ring 4 (App Level)"]
        main["main.cpp (Interactive CLI)"]
    end

    R0 --> R1
    R0 --> R2
    R0 --> R3
    R0 --> R4
    R1 --> R2
    R1 --> R3
    R1 --> R4
    R2 --> R3
    R2 --> R4
    R3 --> R4
```

---

## ⚠️ Anti-Patterns & Prohibitions

1. **Upward Dependency Violation**:
   - ❌ *Incorrect*: `#include "ring2/transformer_lm.hpp"` inside `include/ring1/adamw.hpp`.
   - ✅ *Correct*: `AdamW` operates strictly on `ring0::Matrix` references.
2. **Circular Inclusions**:
   - Forward declarations and clean header separation ensure zero compilation stalls.

---

## 🔗 Related Notes
- [[00 - Overview & Architecture/Architecture Map|Architecture Map]]
- [[Index|Return to Index]]
