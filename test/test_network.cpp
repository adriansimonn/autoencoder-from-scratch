#include "nn/network.h"
#include "nn/dense.h"
#include "nn/relu.h"
#include "nn/sigmoid.h"
#include "nn/mse_loss.h"
#include "optim/adam.h"
#include "io/model_io.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

void test_network_forward_backward() {
    Network net;
    net.add_layer(std::make_shared<DenseLayer>(3, 4, InitMethod::He));
    net.add_layer(std::make_shared<ReLU>());
    net.add_layer(std::make_shared<DenseLayer>(4, 2, InitMethod::He));

    Tensor x(1, 3);
    x[0] = 0.5f; x[1] = -0.3f; x[2] = 0.8f;

    auto y = net.forward(x);
    assert(y.rows == 1 && y.cols == 2);

    // Backward with unit gradient
    Tensor grad(1, 2, 1.0f);
    auto dx = net.backward(grad);
    assert(dx.rows == 1 && dx.cols == 3);

    // Verify parameters are collected
    auto params = net.parameters();
    assert(params.size() == 4); // 2 Dense layers x (W + b)

    printf("  PASS: network forward/backward\n");
}

void test_mse_loss() {
    MSELoss loss;

    Tensor pred(1, 4);
    pred[0]=1; pred[1]=2; pred[2]=3; pred[3]=4;
    Tensor target(1, 4);
    target[0]=1; target[1]=2; target[2]=3; target[3]=4;

    // Perfect prediction -> loss = 0
    float l = loss.forward(pred, target);
    assert(approx(l, 0.0f));

    // Known loss: pred=[1,2,3,4], target=[0,0,0,0] -> MSE = (1+4+9+16)/4 = 7.5
    target[0]=0; target[1]=0; target[2]=0; target[3]=0;
    l = loss.forward(pred, target);
    assert(approx(l, 7.5f));

    // Backward: 2*(pred-target)/N = 2*[1,2,3,4]/4 = [0.5, 1.0, 1.5, 2.0]
    auto grad = loss.backward();
    assert(approx(grad[0], 0.5f));
    assert(approx(grad[1], 1.0f));
    assert(approx(grad[2], 1.5f));
    assert(approx(grad[3], 2.0f));

    printf("  PASS: MSE loss\n");
}

void test_mse_gradient_check() {
    MSELoss loss;
    Tensor pred(1, 4);
    pred[0]=0.5f; pred[1]=-0.3f; pred[2]=0.8f; pred[3]=-0.1f;
    Tensor target(1, 4);
    target[0]=0.2f; target[1]=0.1f; target[2]=-0.5f; target[3]=0.7f;

    loss.forward(pred, target);
    auto grad = loss.backward();

    float eps = 1e-4f;
    for (size_t i = 0; i < pred.size(); ++i) {
        float orig = pred[i];

        pred[i] = orig + eps;
        float loss_plus = loss.forward(pred, target);

        pred[i] = orig - eps;
        float loss_minus = loss.forward(pred, target);

        pred[i] = orig;

        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        assert(approx(grad[i], numerical, 1e-3f));
    }

    printf("  PASS: MSE gradient check\n");
}

void test_tiny_autoencoder_convergence() {
    // Build tiny autoencoder: 4 -> 3 -> 4 (identity mapping)
    Network net;
    net.add_layer(std::make_shared<DenseLayer>(4, 3, InitMethod::He));
    net.add_layer(std::make_shared<ReLU>());
    net.add_layer(std::make_shared<DenseLayer>(3, 4, InitMethod::Xavier));
    net.add_layer(std::make_shared<Sigmoid>());

    MSELoss loss;
    Adam optimizer(net.parameters(), 0.01f);

    // Target: identity-like mapping with values in [0,1] (for sigmoid output)
    Tensor x(1, 4);
    x[0]=0.2f; x[1]=0.8f; x[2]=0.5f; x[3]=0.3f;

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int step = 0; step < 200; ++step) {
        net.zero_gradients();
        auto pred = net.forward(x);
        float l = loss.forward(pred, x);
        auto grad = loss.backward();
        net.backward(grad);
        optimizer.step();

        if (step == 0) initial_loss = l;
        if (step == 199) final_loss = l;
    }

    assert(final_loss < 0.001f);
    assert(final_loss < initial_loss);

    printf("  PASS: tiny autoencoder convergence (loss: %.6f -> %.6f)\n",
           initial_loss, final_loss);
}

void test_model_save_load() {
    // Build a network and do a forward pass
    Network net;
    net.add_layer(std::make_shared<DenseLayer>(4, 3, InitMethod::He));
    net.add_layer(std::make_shared<ReLU>());
    net.add_layer(std::make_shared<DenseLayer>(3, 2, InitMethod::He));

    Tensor x(1, 4);
    x[0]=0.5f; x[1]=-0.3f; x[2]=0.8f; x[3]=-0.1f;
    auto y_before = net.forward(x);

    // Save model
    auto params = net.parameters();
    ModelIO::save(params, "/tmp/test_model.bin");

    // Build a new network with same architecture
    Network net2;
    net2.add_layer(std::make_shared<DenseLayer>(4, 3, InitMethod::He));
    net2.add_layer(std::make_shared<ReLU>());
    net2.add_layer(std::make_shared<DenseLayer>(3, 2, InitMethod::He));

    // Load saved weights
    auto params2 = net2.parameters();
    ModelIO::load(params2, "/tmp/test_model.bin");

    // Forward pass should produce identical output
    auto y_after = net2.forward(x);
    assert(y_after.rows == y_before.rows && y_after.cols == y_before.cols);
    for (size_t i = 0; i < y_before.size(); ++i) {
        assert(approx(y_before[i], y_after[i], 1e-6f));
    }

    printf("  PASS: model save/load round-trip\n");
}

void test_zero_gradients() {
    Network net;
    net.add_layer(std::make_shared<DenseLayer>(3, 2, InitMethod::He));

    Tensor x(1, 3);
    x[0]=1; x[1]=2; x[2]=3;

    // Forward + backward to populate gradients
    net.forward(x);
    Tensor grad(1, 2, 1.0f);
    net.backward(grad);

    // Verify gradients are non-zero
    auto params = net.parameters();
    bool has_nonzero = false;
    for (auto& p : params) {
        for (size_t i = 0; i < p.gradient->size(); ++i) {
            if ((*p.gradient)[i] != 0.0f) has_nonzero = true;
        }
    }
    assert(has_nonzero);

    // Zero gradients
    net.zero_gradients();

    // Verify all gradients are zero
    for (auto& p : params) {
        for (size_t i = 0; i < p.gradient->size(); ++i) {
            assert((*p.gradient)[i] == 0.0f);
        }
    }

    printf("  PASS: zero gradients\n");
}

int main() {
    printf("Running network tests...\n");
    test_network_forward_backward();
    test_mse_loss();
    test_mse_gradient_check();
    test_zero_gradients();
    test_tiny_autoencoder_convergence();
    test_model_save_load();
    printf("All network tests passed!\n");
    return 0;
}
