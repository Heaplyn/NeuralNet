#pragma once

/**
 * @file ringwrapper.hpp
 * @brief Master umbrella header for the entire RingWrapper Neural Network & Causal Transformer LLM system.
 * 
 * Ring Architecture Hierarchy:
 * - Ring 0 (ring0): Core primitive mathematical utilities, linear algebra, tensors, activations, and loss functions.
 * - Ring 1 (ring1): Transformer subcomponents (Embedding, Causal Multi-Head Attention, Pre-LN Decoder Block) & Optimizers (AdamW, Momentum SGD).
 * - Ring 2 (ring2): Network architectures (TransformerLM, Tokenizer, Feedforward NeuralNet, Dynamic GrowthController).
 * - Ring 3 (ring3): Datasets (Text Next-Token Dataset, MNIST, Fashion-MNIST, Letter Bitmaps) & Training engines (LLMTrainer, RingTrainer).
 * - Ring 4 (apps): Application layer, demos, and end-user interactive inference scripts.
 */

// Ring 0: Core Math, Tensors, Activations, Losses, Configuration
#include "ring0/tensor.hpp"
#include "ring0/activations.hpp"
#include "ring0/loss.hpp"
#include "ring0/config.hpp"

// Ring 1: Transformer Layers, Recursive Thinking & Optimizers
#include "ring1/embedding.hpp"
#include "ring1/attention.hpp"
#include "ring1/transformer_block.hpp"
#include "ring1/recursive_layer.hpp"
#include "ring1/adamw.hpp"
#include "ring1/layer.hpp"
#include "ring1/optimizer.hpp"

// Ring 2: LLM Network Topology & Tokenizer
#include "ring2/tokenizer.hpp"
#include "ring2/transformer_lm.hpp"
#include "ring2/neural_net.hpp"
#include "ring2/growth_controller.hpp"

// Ring 3: Datasets & LLM Trainer
#include "ring3/text_dataset.hpp"
#include "ring3/llm_trainer.hpp"
#include "ring3/letter_dataset.hpp"
#include "ring3/mnist_dataset.hpp"
#include "ring3/image_dataset.hpp"
#include "ring3/trainer.hpp"

using namespace std;
