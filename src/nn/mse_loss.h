#pragma once

#include "math/tensor.h"

class MSELoss {
public:
    float forward(const Tensor& prediction, const Tensor& target);
    Tensor backward();

private:
    Tensor prediction_cache_;
    Tensor target_cache_;
    size_t n_;
};
