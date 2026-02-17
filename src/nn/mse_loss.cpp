#include "nn/mse_loss.h"

float MSELoss::forward(const Tensor& prediction, const Tensor& target) {
    prediction_cache_ = prediction;
    target_cache_ = target;
    n_ = prediction.size();

    float sum = 0.0f;
    for (size_t i = 0; i < n_; ++i) {
        float diff = prediction[i] - target[i];
        sum += diff * diff;
    }
    return sum / static_cast<float>(n_);
}

Tensor MSELoss::backward() {
    Tensor grad(prediction_cache_.rows, prediction_cache_.cols);
    float scale = 2.0f / static_cast<float>(n_);
    for (size_t i = 0; i < n_; ++i) {
        grad[i] = scale * (prediction_cache_[i] - target_cache_[i]);
    }
    return grad;
}
