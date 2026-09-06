#pragma once

/**
 * @file ringwrapper.hpp
 * @brief Master umbrella header for the entire RingWrapper Neural Network, Vision CNN & Causal Transformer LLM system.
 * 
 * Ring Architecture Hierarchy:
 * - Ring 0 (ring0): Core primitive mathematical utilities, linear algebra, 2D/3D tensors, activations, Taylor predictor, and loss functions.
 * - Ring 1 (ring1): Transformer subcomponents (Embedding, GQA Attention, RMSNorm, SwiGLU, Pre-LN Block, RecursiveLayer) & Optimizers (AdamW, MetaLossOptimizer, MultiFormula).
 * - Ring 2 (ring2): Network architectures (TransformerLM, Tokenizer, VocabManager, Feedforward NeuralNet, GrowthController).
 * - Ring 3 (ring3): Computer Vision & CNNs (Tensor4D, Conv2D, MaxPool2D, AvgPool2D, CNN, CNNTrainer with Taylor foresight, Meta-LR, mistake memory sizing, and auto grad norm).
 * - Ring 4 (ring4): Datasets (Text Next-Token Dataset, MNIST, Fashion-MNIST, Letter Bitmaps) & Pipeline trainers (LLMTrainer, RingTrainer, ChronoScheduler, DataLoader).
 * - Ring 5 (apps) : Application layer, demos, and interactive inference scripts.
 */

// Ring 0: Core Math, Tensors, Activations, Losses, Configuration, Taylor Predictor, CoC
#include "ring0/tensor.hpp"
#include "ring0/activations.hpp"
#include "ring0/loss.hpp"
#include "ring0/config.hpp"
#include "ring0/taylor_predictor.hpp"
#include "ring0/cuda_backend.hpp"
#include "ring0/calculus_of_constructions.hpp"

// Ring 1: Transformer Layers, Recursive Thinking & Optimizers
#include "ring1/embedding.hpp"
#include "ring1/attention.hpp"
#include "ring1/transformer_block.hpp"
#include "ring1/recursive_layer.hpp"
#include "ring1/adamw.hpp"
#include "ring1/layer.hpp"
#include "ring1/optimizer.hpp"
#include "ring1/meta_loss_optimizer.hpp"
#include "ring1/multi_formula_optimizer.hpp"
#include "ring1/dependent_type_attention.hpp"

// Ring 2: LLM Network Topology, Vocab & Tokenizer
#include "ring2/tokenizer.hpp"
#include "ring2/transformer_lm.hpp"
#include "ring2/neural_net.hpp"
#include "ring2/growth_controller.hpp"
#include "ring2/vocab_manager.hpp"

// Ring 3: Computer Vision, 4D Tensors & Convolutional Networks
#include "ring3/tensor4d.hpp"
#include "ring3/conv2d.hpp"
#include "ring3/maxpool2d.hpp"
#include "ring3/cnn.hpp"
#include "ring3/cnn_trainer.hpp"

// Ring 4: Datasets, DataLoader, Pipelines & LLM Trainer
#include "ring4/text_dataset.hpp"
#include "ring4/llm_trainer.hpp"
#include "ring4/letter_dataset.hpp"
#include "ring4/mnist_dataset.hpp"
#include "ring4/image_dataset.hpp"
#include "ring4/trainer.hpp"
#include "ring4/data_loader.hpp"
#include "ring4/chrono_scheduler.hpp"

using namespace std;
