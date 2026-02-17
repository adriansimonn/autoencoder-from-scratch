#pragma once

#include "nn/layer.h"

class Sigmoid : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Sigmoid"; }

private:
    Tensor output_cache_;
};
