#pragma once

#include "nn/layer.h"
#include <vector>
#include <memory>

class Network {
public:
    void add_layer(std::shared_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    std::vector<Parameter> parameters();
    void zero_gradients();

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};
