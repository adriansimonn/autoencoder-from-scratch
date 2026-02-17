#pragma once

#include "nn/layer.h"

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "ReLU"; }

private:
    Tensor input_cache_;
};
