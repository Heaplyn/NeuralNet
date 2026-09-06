#include "ring2/neural_net.hpp"
#include <iostream>

using namespace std;

namespace ring2
{

    // Appends an existing layer to the sequential stack
    void NeuralNet::add_layer(const ring1::DenseLayer &layer)
    {
        layers.push_back(layer);
    }

    // Constructs and appends a dense layer
    void NeuralNet::add_dense(size_t in_dim, size_t out_dim, ring0::ActivationType act)
    {
        layers.emplace_back(in_dim, out_dim, act);
    }

    // Forward propagation: X_0 -> Layer_0 -> Layer_1 -> ... -> Output
    ring0::Matrix NeuralNet::forward(const ring0::Matrix &input)
    {
        ring0::Matrix current = input;
        for (auto &layer : layers)
        {
            current = layer.forward(current);
        }
        return current;
    }

    // Backward propagation: dOut -> Layer_N -> Layer_{N-1} -> ... -> dIn
    ring0::Matrix NeuralNet::backward(const ring0::Matrix &grad_output, float relevancy)
    {
        ring0::Matrix current_grad = grad_output * relevancy;
        int size = static_cast<int>(layers.size() * relevancy);
        for (int i = static_cast<int>(size) - 1; i >= 0; --i)
        {
            current_grad = layers[i].backward(current_grad, relevancy / (1.0f + (float)i * 7.0f / sqrt((float)size)));
        }
        return current_grad;
    }

    // Resets parameter gradients across all dense layers
    void NeuralNet::reset_gradients()
    {
        for (auto &layer : layers)
        {
            layer.reset_gradients();
        }
    }

    // Dynamically expands hidden layer output width and next layer's input width
    bool NeuralNet::expand_hidden_layer(size_t layer_index, size_t additional_neurons)
    {
        if (layer_index >= layers.size() - 1 || additional_neurons == 0)
        {
            return false;
        }

        // Expand current layer's output
        layers[layer_index].expand_output_dim(additional_neurons);
        // Expand next layer's input
        layers[layer_index + 1].expand_input_dim(additional_neurons);
        return true;
    }

    // Computes sum of all weights and biases across the network
    size_t NeuralNet::get_total_parameters() const
    {
        size_t total = 0;
        for (const auto &layer : layers)
        {
            total += layer.weights.data.size() + layer.biases.data.size();
        }
        return total;
    }

    // Prints architecture layout
    void NeuralNet::print_architecture() const
    {
        cout << "--- Neural Network Architecture ---\n";
        for (size_t i = 0; i < layers.size(); ++i)
        {
            cout << "  Layer " << i << ": (" << layers[i].in_features
                 << " -> " << layers[i].out_features << ")";
            if (i == layers.size() - 1)
            {
                cout << " [Output Layer]";
            }
            else
            {
                cout << " [Hidden Layer]";
            }
            cout << " | Parameters: " << (layers[i].weights.data.size() + layers[i].biases.data.size()) << "\n";
        }
        cout << "  Total Parameters: " << get_total_parameters() << "\n";
        cout << "-----------------------------------\n";
    }

} // namespace ring2
