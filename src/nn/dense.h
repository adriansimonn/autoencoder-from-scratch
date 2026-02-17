#pragma once

#include "nn/layer.h"

enum class InitMethod { He, Xavier };

class DenseLayer : public Layer {
public:
    DenseLayer(size_t in_features, size_t out_features, InitMethod init = InitMethod::He);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Parameter> parameters() override;
    std::string name() const override { return "Dense"; }

private:
    size_t in_features_, out_features_;
    Tensor W_, b_;
    Tensor dW_, db_;
    Tensor input_cache_;
};
