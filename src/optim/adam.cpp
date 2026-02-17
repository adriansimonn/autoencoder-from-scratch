#include "optim/adam.h"
#include <cmath>

Adam::Adam(std::vector<Parameter> params, float lr, float beta1, float beta2, float epsilon)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      epsilon_(epsilon), t_(0) {

    for (auto& p : params_) {
        m_.push_back(Tensor::zeros(p.value->rows, p.value->cols));
        v_.push_back(Tensor::zeros(p.value->rows, p.value->cols));
    }
}

void Adam::step() {
    t_++;
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& param = *params_[i].value;
        Tensor& grad = *params_[i].gradient;
        Tensor& m = m_[i];
        Tensor& v = v_[i];

        for (size_t j = 0; j < param.size(); ++j) {
            // Update biased first moment: m = beta1 * m + (1 - beta1) * g
            m[j] = beta1_ * m[j] + (1.0f - beta1_) * grad[j];
            // Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
            v[j] = beta2_ * v[j] + (1.0f - beta2_) * grad[j] * grad[j];
            // Bias-corrected estimates
            float m_hat = m[j] / bc1;
            float v_hat = v[j] / bc2;
            // Update parameter
            param[j] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}
