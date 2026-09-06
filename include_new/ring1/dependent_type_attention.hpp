#pragma once

#include "ring0/tensor.hpp"
#include "ring0/calculus_of_constructions.hpp"
#include <vector>
#include <cmath>

namespace ring1
{

    /**
     * @struct TypeEmbeddingSignature
     * @brief Continuous type vector embedding representing a term's CoC type signature.
     */
    struct TypeEmbeddingSignature
    {
        std::vector<float> type_vector;     ///< (1 x type_dim) Continuous semantic type embedding
        ring0::UniverseSort universe_level; ///< CoC Universe rank (Prop, Type_0, Type_1, etc.)
        float proof_certainty = 1.0f;       ///< Proof verification confidence factor in [0, 1]
    };

    /**
     * @class DependentTypeAttention
     * @brief Computes constructive type-directed attention compatible with the Calculus of Constructions.
     * Modulates standard dot-product attention scores by dependent type unification:
     * A_ij = Softmax( (Q_i K_j^T / sqrt(d)) + alpha * TypeMatch(tau(q_i), tau(k_j)) )
     */
    class DependentTypeAttention
    {
    public:
        size_t embed_dim;
        size_t type_dim;
        float type_guidance_alpha = 0.25f; ///< Weight of type-consistency prior

        ring0::Matrix W_type_q; ///< Query type projection (embed_dim -> type_dim)
        ring0::Matrix W_type_k; ///< Key type projection (embed_dim -> type_dim)

        ring0::Matrix grad_W_type_q;
        ring0::Matrix grad_W_type_k;

        explicit DependentTypeAttention(size_t dim = 256, size_t t_dim = 64, float alpha = 0.25f);

        /// Computes type compatibility matrix (B, T, T) from query and key tensor sequences
        ring0::Tensor3D compute_type_compatibility(const ring0::Tensor3D &Q, const ring0::Tensor3D &K);

        /// Combines raw dot-product attention logits with CoC type-unification priors
        void apply_type_guidance(ring0::Tensor3D &attn_weights, const ring0::Tensor3D &type_compatibility);

        /// Resets type projection gradients
        void reset_gradients();

        /// Updates parameters with optimizer
        void update_parameters(float lr);
    };

} // namespace ring1
