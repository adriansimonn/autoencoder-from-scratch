#pragma once

#include "nn/network.h"
#include "nn/dense.h"
#include "nn/relu.h"
#include "nn/sigmoid.h"
#include <vector>

class Autoencoder {
public:
    Autoencoder();

    // Forward pass through full autoencoder (encode then decode)
    Tensor forward(const Tensor& input);

    // Encode input to latent space
    Tensor encode(const Tensor& input);

    // Decode latent vector to reconstruction
    Tensor decode(const Tensor& latent);

    // Backward pass through full autoencoder
    Tensor backward(const Tensor& grad_output);

    // Get all trainable parameters (encoder + decoder)
    std::vector<Parameter> parameters();

    // Zero all gradients
    void zero_gradients();

private:
    Network encoder_;
    Network decoder_;
};
