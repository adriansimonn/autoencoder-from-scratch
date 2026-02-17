#pragma once

#include "nn/layer.h"
#include <vector>

class Adam {
public:
    Adam(std::vector<Parameter> params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    void step();

private:
    std::vector<Parameter> params_;
    std::vector<Tensor> m_;  // First moment estimates
    std::vector<Tensor> v_;  // Second moment estimates
    float lr_, beta1_, beta2_, epsilon_;
    int t_;  // Timestep
};
