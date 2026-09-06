#include "ring1/recursive_layer.hpp"
#include "ring0/activations.hpp"
#include "ring0/config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>

using namespace std;

namespace ring1 {

// Helper: L2 Frobenius norm
static float matrix_norm(const ring0::Matrix& M) {
    double sum = 0.0;
    for (float v : M.data) sum += static_cast<double>(v) * static_cast<double>(v);
    return static_cast<float>(sqrt(sum));
}

// Helper: Cosine similarity
static float matrix_cosine_sim(const ring0::Matrix& A, const ring0::Matrix& B) {
    if (A.data.size() != B.data.size() || A.data.empty()) return 1.0f;
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < A.data.size(); ++i) {
        dot += static_cast<double>(A.data[i]) * static_cast<double>(B.data[i]);
        normA += static_cast<double>(A.data[i]) * static_cast<double>(A.data[i]);
        normB += static_cast<double>(B.data[i]) * static_cast<double>(B.data[i]);
    }
    double denom = sqrt(normA) * sqrt(normB);
    return (denom > 1e-8) ? static_cast<float>(dot / denom) : 0.0f;
}

RecursiveLayer::RecursiveLayer(const string& layer_name, size_t in_dim, size_t out_dim, size_t depth)
    : name(layer_name),
      in_features(in_dim),
      out_features(out_dim),
      thinking_depth(depth),
      parent(nullptr) {

    float scale = sqrt(2.0f / static_cast<float>(in_dim));
    W_think = ring0::Matrix::random_normal(out_dim, out_dim, 0.0f, scale);
    b_think = ring0::Matrix::zeros(1, out_dim);
    W_context = ring0::Matrix::random_normal(in_dim, out_dim, 0.0f, scale);

    grad_W_think = ring0::Matrix::zeros(out_dim, out_dim);
    grad_b_think = ring0::Matrix::zeros(1, out_dim);
    grad_W_context = ring0::Matrix::zeros(in_dim, out_dim);
}

RecursiveLayer::RecursiveLayer(const RecursiveLayer& other)
    : name(other.name),
      in_features(other.in_features),
      out_features(other.out_features),
      thinking_depth(other.thinking_depth),
      parent(nullptr),
      W_think(other.W_think),
      b_think(other.b_think),
      W_context(other.W_context),
      grad_W_think(other.grad_W_think),
      grad_b_think(other.grad_b_think),
      grad_W_context(other.grad_W_context),
      last_input(other.last_input),
      step_H_cache(other.step_H_cache),
      step_linear_cache(other.step_linear_cache),
      thought_chain_history(other.thought_chain_history) {
    for (const auto& child : other.children) {
        if (child) {
            auto child_copy = std::make_unique<RecursiveLayer>(*child);
            child_copy->parent = this;
            children.push_back(std::move(child_copy));
        }
    }
}

RecursiveLayer& RecursiveLayer::operator=(const RecursiveLayer& other) {
    if (this == &other) return *this;
    name = other.name;
    in_features = other.in_features;
    out_features = other.out_features;
    thinking_depth = other.thinking_depth;
    parent = nullptr;
    W_think = other.W_think;
    b_think = other.b_think;
    W_context = other.W_context;
    grad_W_think = other.grad_W_think;
    grad_b_think = other.grad_b_think;
    grad_W_context = other.grad_W_context;
    last_input = other.last_input;
    step_H_cache = other.step_H_cache;
    step_linear_cache = other.step_linear_cache;
    thought_chain_history = other.thought_chain_history;
    children.clear();
    for (const auto& child : other.children) {
        if (child) {
            auto child_copy = std::make_unique<RecursiveLayer>(*child);
            child_copy->parent = this;
            children.push_back(std::move(child_copy));
        }
    }
    return *this;
}

// Attaches child sub-layer and links this layer as parent
void RecursiveLayer::add_child(unique_ptr<RecursiveLayer> child) {
    if (child) {
        child->parent = this;
        children.push_back(move(child));
    }
}

// Traverses up to the root to calculate depth in tree hierarchy
size_t RecursiveLayer::get_tree_depth() const {
    size_t d = 0;
    const RecursiveLayer* curr = parent;
    while (curr != nullptr) {
        d++;
        curr = curr->parent;
    }
    return d;
}

// Queries contextual representation from parent if available
ring0::Matrix RecursiveLayer::get_parent_context() const {
    if (parent != nullptr) {
        return parent->W_think;
    }
    return ring0::Matrix::zeros(1, out_features);
}

// Recursive forward pass: Performs K internal reasoning loops and delegates to child branches
ring0::Matrix RecursiveLayer::forward(const ring0::Matrix& X) {
    bool debug = ring0::is_debug_mode() || ring0::get_config().verbose_thought_chains;
    string indent(get_tree_depth() * 4, ' ');

    last_input = X;
    step_H_cache.clear();
    step_linear_cache.clear();

    // 1. Initial projection into layer's feature dimension
    ring0::Matrix H = (X.cols == in_features) ? X.matmul(W_context) : X;
    if (H.cols != out_features && W_think.rows == out_features) {
        // Project or pad if dimension differs
        if (H.cols < out_features) {
            ring0::Matrix H_pad(H.rows, out_features);
            for (size_t r = 0; r < H.rows; ++r) {
                for (size_t c = 0; c < H.cols; ++c) H_pad(r, c) = H(r, c);
            }
            H = H_pad;
        }
    }

    if (debug) {
        cout << indent << "🧠 [Thought Chain @ " << name << "] (Depth: " << get_tree_depth() 
             << ", Latent Dim: " << out_features << ", Input Norm: " 
             << fixed << setprecision(3) << matrix_norm(H) << ")\n";
    }

    // 2. Latent Recursive Thinking Loops: Refines hidden state over K iterations
    for (size_t step = 0; step < thinking_depth; ++step) {
        ring0::Matrix prev_H = H;
        step_H_cache.push_back(H);

        // Internal thinking step: H_{t+1} = GELU(H_t * W_think + b_think) + H_t (residual)
        ring0::Matrix linear = H.matmul(W_think).add_bias(b_think);
        step_linear_cache.push_back(linear);
        ring0::Matrix thought = ring0::Activations::gelu(linear);
        H = H + thought; // Residual thought accumulation

        // Telemetry metrics
        float delta = matrix_norm(thought);
        float energy = matrix_norm(H);
        float sim = matrix_cosine_sim(prev_H, H);

        string stage;
        ring0::CoCTermPtr witness = nullptr;
        ring0::CoCTermPtr prop = nullptr;
        static ring0::TypingContext logic_ctx = ring0::CoCTypeChecker::create_standard_logic_context();

        if (step == 0)
        {
            stage = "Perceptual Grounding & Feature Projection";
            prop = ring0::CoCTerm::make_universe(ring0::UniverseSort::PROP, 0);
            witness = ring0::CoCTerm::make_abstraction("p", prop, ring0::CoCTerm::make_var("p"));
        }
        else if (step == 1)
        {
            stage = "Latent Reasoning & Semantic Synthesis";
            auto prop_a = ring0::CoCTerm::make_universe(ring0::UniverseSort::PROP, 0);
            prop = ring0::CoCTerm::make_arrow(prop_a, prop_a);
            witness = ring0::CoCTerm::make_abstraction("a", prop_a, ring0::CoCTerm::make_var("a"));
        }
        else if (step == 2)
        {
            stage = "Hierarchical Hypothesis Formulation";
            auto type0 = ring0::CoCTerm::make_universe(ring0::UniverseSort::TYPE_0, 0);
            prop = ring0::CoCTerm::make_pi("T", type0, ring0::CoCTerm::make_arrow(ring0::CoCTerm::make_var("T"), ring0::CoCTerm::make_var("T")));
            witness = ring0::CoCTerm::make_abstraction("T", type0, ring0::CoCTerm::make_abstraction("t", ring0::CoCTerm::make_var("T"), ring0::CoCTerm::make_var("t")));
        }
        else
        {
            stage = "Self-Reflective Refinement & Deduction";
            auto prop_p = ring0::CoCTerm::make_universe(ring0::UniverseSort::PROP, 0);
            prop = prop_p;
            witness = ring0::CoCTerm::make_abstraction("h", prop_p, ring0::CoCTerm::make_var("h"));
        }

        ring0::ProofValidationResult proof_res = ring0::CoCTypeChecker::verify_proof(logic_ctx, witness, nullptr);

        ThoughtStep record{
            step,
            1,
            name,
            energy,
            delta,
            sim,
            stage,
            witness,
            prop,
            proof_res
        };
        thought_chain_history.push_back(record);

        if (debug) {
            cout << indent << "   ├── Step " << (step + 1) << "/" << thinking_depth 
                 << ": Energy=" << fixed << setprecision(3) << energy
                 << " | Δ=" << fixed << setprecision(3) << delta
                 << " | CosSim=" << fixed << setprecision(3) << sim
                 << " | CoC Proof=" << (proof_res.is_valid ? "VALID" : "UNVERIFIED")
                 << " -> [" << stage << "]\n";
        }
    }

    // 3. Hierarchical Child Sub-layer Execution (if any children exist)
    for (size_t c_idx = 0; c_idx < children.size(); ++c_idx) {
        auto& child = children[c_idx];
        if (debug) {
            cout << indent << "   └── Delegating Context to Child Branch [" 
                 << (c_idx + 1) << "/" << children.size() << "]: " << child->name << "...\n";
        }
        H = child->forward(H);
    }

    return H;
}

ring0::Matrix RecursiveLayer::backward(const ring0::Matrix& grad_output, float relevancy) {
    if (grad_W_think.rows != W_think.rows || grad_W_think.cols != W_think.cols) {
        grad_W_think = ring0::Matrix::zeros(W_think.rows, W_think.cols);
    }
    if (grad_b_think.rows != b_think.rows || grad_b_think.cols != b_think.cols) {
        grad_b_think = ring0::Matrix::zeros(b_think.rows, b_think.cols);
    }
    if (grad_W_context.rows != W_context.rows || grad_W_context.cols != W_context.cols) {
        grad_W_context = ring0::Matrix::zeros(W_context.rows, W_context.cols);
    }

    ring0::Matrix dH = grad_output;

    // 1. Backprop through children in reverse order
    for (int c_idx = static_cast<int>(children.size()) - 1; c_idx >= 0; --c_idx) {
        if (children[c_idx]) {
            dH = children[c_idx]->backward(dH, relevancy);
        }
    }

    // 2. Backprop through thinking steps in reverse order
    for (int step = static_cast<int>(step_H_cache.size()) - 1; step >= 0; --step) {
        const auto& H_t = step_H_cache[step];
        const auto& linear_t = step_linear_cache[step];

        ring0::Matrix d_linear = ring0::Activations::gelu_derivative(linear_t, dH);

        // dW_think += H_t^T * d_linear * relevancy
        ring0::Matrix H_t_T = H_t.transpose();
        ring0::Matrix step_grad_W = H_t_T.matmul(d_linear) * relevancy;
        grad_W_think = grad_W_think + step_grad_W;

        // db_think += col_sum(d_linear) * relevancy
        for (size_t r = 0; r < d_linear.rows; ++r) {
            for (size_t c = 0; c < d_linear.cols; ++c) {
                grad_b_think(0, c) += d_linear(r, c) * relevancy;
            }
        }

        // dH_t = dH_{t+1} + d_linear * W_think^T
        ring0::Matrix W_think_T = W_think.transpose();
        dH = dH + d_linear.matmul(W_think_T);
    }

    // 3. Backprop through context projection
    ring0::Matrix dX = dH;
    if (last_input.cols == in_features && W_context.rows == in_features && W_context.cols == out_features) {
        ring0::Matrix X_T = last_input.transpose();
        ring0::Matrix step_grad_W_ctx = X_T.matmul(dH) * relevancy;
        grad_W_context = grad_W_context + step_grad_W_ctx;

        ring0::Matrix W_ctx_T = W_context.transpose();
        dX = dH.matmul(W_ctx_T);
    }

    return dX;
}

void RecursiveLayer::reset_gradients() {
    grad_W_think = ring0::Matrix::zeros(W_think.rows, W_think.cols);
    grad_b_think = ring0::Matrix::zeros(b_think.rows, b_think.cols);
    grad_W_context = ring0::Matrix::zeros(W_context.rows, W_context.cols);
    for (auto& child : children) {
        if (child) child->reset_gradients();
    }
}

// Multi-Pass Thought Chain Looping: Recursively loops through and reflects on its own thought chain
ring0::Matrix RecursiveLayer::loop_thought_chain(const ring0::Matrix& X, size_t num_reflection_cycles, bool verbose_override) {
    bool debug = verbose_override || ring0::is_debug_mode() || ring0::get_config().verbose_thought_chains;
    string indent(get_tree_depth() * 4, ' ');

    if (debug) {
        cout << "\n" << indent << "🔄 [Thought Chain Multi-Pass Looping Engine: " << name << "]\n"
             << indent << "   Initiating " << num_reflection_cycles << " recursive reflection cycle(s)...\n";
    }

    ring0::Matrix current_thought = X;
    float damping = ring0::get_config().thought_damping;
    float tol = ring0::get_config().thought_convergence_tol;

    for (size_t cycle = 1; cycle <= num_reflection_cycles; ++cycle) {
        if (debug) {
            cout << indent << "  |-- [Reflection Cycle " << cycle << "/" << num_reflection_cycles << "]\n";
        }

        ring0::Matrix prev_thought = current_thought;
        ring0::Matrix new_thought = forward(current_thought);

        // Residual momentum damping across thought chain reflection loops
        if (cycle > 1 && current_thought.rows == new_thought.rows && current_thought.cols == new_thought.cols) {
            for (size_t i = 0; i < current_thought.data.size(); ++i) {
                current_thought.data[i] = damping * current_thought.data[i] + (1.0f - damping) * new_thought.data[i];
            }
        } else {
            current_thought = new_thought;
        }

        float cycle_delta = matrix_norm(current_thought - prev_thought);
        float cycle_sim = matrix_cosine_sim(prev_thought, current_thought);

        if (debug) {
            cout << indent << "  \\-- Cycle " << cycle << " Result: Energy=" << fixed << setprecision(3) << matrix_norm(current_thought)
                 << " | Reflection delta=" << fixed << setprecision(4) << cycle_delta
                 << " | Alignment=" << fixed << setprecision(4) << cycle_sim << "\n";
        }

        // Early convergence detection
        if (cycle > 1 && cycle_delta < tol) {
            if (debug) {
                cout << indent << "  ⭐ [Thought Chain Converged] Early exit at cycle " << cycle 
                     << " (Δ " << cycle_delta << " < " << tol << ")\n";
            }
            break;
        }
    }

    if (debug) {
        cout << indent << "✨ [Thought Chain Loop Complete: Refined Representation Dimension: (" 
             << current_thought.rows << "x" << current_thought.cols << ")]\n\n";
    }

    return current_thought;
}

// Prints formatted ASCII summary of thought chain history
void RecursiveLayer::print_thought_chain_summary() const {
    cout << "\n=========================================================\n";
    cout << "      THOUGHT CHAIN EXECUTION HISTORY & REASONING TRACE  \n";
    cout << "=========================================================\n";
    cout << " Layer: " << name << " (Thinking Depth: " << thinking_depth << ")\n";
    cout << " Children Sub-Reasoners: " << children.size() << "\n";
    cout << " Total Recorded Thought Steps: " << thought_chain_history.size() << "\n";
    cout << "---------------------------------------------------------\n";

    for (size_t i = 0; i < thought_chain_history.size(); ++i) {
        const auto& s = thought_chain_history[i];
        cout << " Step " << setw(2) << (i + 1)
             << " | Cycle " << s.loop_cycle
             << " | " << setw(20) << left << s.layer_name << right
             << " | Energy: " << fixed << setprecision(3) << s.energy_norm
             << " | Δ: " << fixed << setprecision(3) << s.delta_magnitude
             << " | Sim: " << fixed << setprecision(3) << s.cosine_similarity
             << " | [" << s.stage_description << "]\n";
    }
    cout << "=========================================================\n";
}

} // namespace ring1
