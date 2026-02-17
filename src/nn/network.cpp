#include "nn/network.h"

void Network::add_layer(std::shared_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Tensor Network::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }
    return x;
}

Tensor Network::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        grad = layers_[i]->backward(grad);
    }
    return grad;
}

std::vector<Parameter> Network::parameters() {
    std::vector<Parameter> params;
    for (auto& layer : layers_) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void Network::zero_gradients() {
    for (auto& param : parameters()) {
        param.gradient->zero();
    }
}
