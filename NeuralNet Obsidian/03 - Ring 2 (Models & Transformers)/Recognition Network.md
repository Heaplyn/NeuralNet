# Recognition Network and Dense Layers

The recognition model is a small sequential dense network built from `ring2::NeuralNet` and `ring1::DenseLayer`.

## Shapes

```text
A-Z:          35 -> 64 -> 26
MNIST:       784 -> 128 -> 10
Fashion-MNIST: 784 -> 128 -> 10
```

The hidden layer uses ReLU. The output layer is linear and produces one logit per class. Cross-entropy applies a numerically stable softmax interpretation to those logits.

## Forward Pass

For a batch matrix $X$ and layer weights $W$:

$$
Z = XW + b, \qquad A = \mathrm{ReLU}(Z)
$$

The output logits are compared with one-hot labels. The largest output logit is the predicted class.

## Backward Pass

`DenseLayer::backward()` caches the input and output from the forward pass, computes weight and bias gradients, and returns the input gradient to the previous layer. `RingTrainer` then sends copied gradients through clipping and adaptive scaling before AdamW updates the actual parameters.

## Structural Growth

The Ring 2 growth controller monitors epoch loss. If a plateau is detected, it can expand hidden layers. The trainer then re-registers all tensors with AdamW so moment, Fisher, and shift buffers match the new shapes. Growth is bounded by configured width ceilings and rejects non-finite requests.

## Evaluation

Recognition is evaluated with top-1 label accuracy:

$$
\mathrm{accuracy} = \frac{\#\{\operatorname{argmax}(\hat{y}) = \operatorname{argmax}(y)\}}{N}
$$

See [[04 - Ring 3 (Data & Training Pipelines)/Recognition Benchmarks - Letters MNIST Fashion-MNIST|Recognition Benchmarks]] for dataset paths and measured results.
