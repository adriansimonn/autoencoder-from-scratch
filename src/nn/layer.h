#pragma once

#include "math/tensor.h"
#include <vector>
#include <string>

struct Parameter {
    Tensor* value;
    Tensor* gradient;
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Parameter> parameters() { return {}; }
    virtual std::string name() const = 0;
};
