#include "nn/relu.h"
#include "nn/sigmoid.h"
#include <cassert>
#include <cmath>
#include <cstdio>

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// --- ReLU Tests ---

void test_relu_forward() {
    ReLU relu;
    Tensor x(1, 5);
    x[0]=-2; x[1]=-0.5f; x[2]=0; x[3]=0.5f; x[4]=3;

    auto y = relu.forward(x);
    assert(approx(y[0], 0));
    assert(approx(y[1], 0));
    assert(approx(y[2], 0));
    assert(approx(y[3], 0.5f));
    assert(approx(y[4], 3));

    printf("  PASS: relu forward\n");
}

void test_relu_backward() {
    ReLU relu;
    Tensor x(1, 5);
    x[0]=-2; x[1]=-0.5f; x[2]=0; x[3]=0.5f; x[4]=3;
    relu.forward(x);

    Tensor grad(1, 5, 1.0f);
    auto dx = relu.backward(grad);

    assert(approx(dx[0], 0));
    assert(approx(dx[1], 0));
    assert(approx(dx[2], 0));
    assert(approx(dx[3], 1));
    assert(approx(dx[4], 1));

    printf("  PASS: relu backward\n");
}

void test_relu_gradient_check() {
    ReLU relu;
    Tensor x(1, 4);
    x[0]=-1.0f; x[1]=0.5f; x[2]=2.0f; x[3]=-0.3f;

    relu.forward(x);
    Tensor grad(1, 4, 1.0f);
    auto dx = relu.backward(grad);

    float eps = 1e-4f;
    for (size_t i = 0; i < x.size(); ++i) {
        // Skip values near zero where ReLU is non-differentiable
        if (std::fabs(x[i]) < 0.01f) continue;

        float orig = x[i];

        x[i] = orig + eps;
        auto y_plus = relu.forward(x);
        float loss_plus = 0;
        for (size_t j = 0; j < y_plus.size(); ++j) loss_plus += y_plus[j];

        x[i] = orig - eps;
        auto y_minus = relu.forward(x);
        float loss_minus = 0;
        for (size_t j = 0; j < y_minus.size(); ++j) loss_minus += y_minus[j];

        x[i] = orig;

        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        assert(approx(dx[i], numerical, 1e-3f));
    }

    printf("  PASS: relu gradient check\n");
}

// --- Sigmoid Tests ---

void test_sigmoid_forward() {
    Sigmoid sig;
    Tensor x(1, 5);
    x[0]=-100; x[1]=-1; x[2]=0; x[3]=1; x[4]=100;

    auto y = sig.forward(x);
    // sigmoid(-100) ≈ 0, sigmoid(-1) ≈ 0.2689, sigmoid(0) = 0.5,
    // sigmoid(1) ≈ 0.7311, sigmoid(100) ≈ 1
    assert(y[0] < 1e-6f);
    assert(approx(y[1], 0.2689f, 1e-3f));
    assert(approx(y[2], 0.5f));
    assert(approx(y[3], 0.7311f, 1e-3f));
    assert(y[4] > 1.0f - 1e-6f);

    printf("  PASS: sigmoid forward\n");
}

void test_sigmoid_backward() {
    Sigmoid sig;
    Tensor x(1, 3);
    x[0]=-1; x[1]=0; x[2]=1;

    auto y = sig.forward(x);

    Tensor grad(1, 3, 1.0f);
    auto dx = sig.backward(grad);

    // sigmoid'(x) = sig(x) * (1 - sig(x))
    for (size_t i = 0; i < 3; ++i) {
        float expected = y[i] * (1.0f - y[i]);
        assert(approx(dx[i], expected, 1e-5f));
    }

    printf("  PASS: sigmoid backward\n");
}

void test_sigmoid_gradient_check() {
    Sigmoid sig;
    Tensor x(1, 4);
    x[0]=-2.0f; x[1]=-0.5f; x[2]=0.5f; x[3]=2.0f;

    sig.forward(x);
    Tensor grad(1, 4, 1.0f);
    auto dx = sig.backward(grad);

    float eps = 1e-4f;
    for (size_t i = 0; i < x.size(); ++i) {
        float orig = x[i];

        x[i] = orig + eps;
        auto y_plus = sig.forward(x);
        float loss_plus = 0;
        for (size_t j = 0; j < y_plus.size(); ++j) loss_plus += y_plus[j];

        x[i] = orig - eps;
        auto y_minus = sig.forward(x);
        float loss_minus = 0;
        for (size_t j = 0; j < y_minus.size(); ++j) loss_minus += y_minus[j];

        x[i] = orig;

        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        assert(approx(dx[i], numerical, 1e-3f));
    }

    printf("  PASS: sigmoid gradient check\n");
}

void test_sigmoid_numerical_stability() {
    Sigmoid sig;
    Tensor x(1, 2);
    x[0] = -200.0f;  // Very negative
    x[1] = 200.0f;   // Very positive

    auto y = sig.forward(x);
    // Should not produce NaN or Inf
    assert(!std::isnan(y[0]) && !std::isinf(y[0]));
    assert(!std::isnan(y[1]) && !std::isinf(y[1]));
    assert(y[0] >= 0.0f && y[0] <= 1.0f);
    assert(y[1] >= 0.0f && y[1] <= 1.0f);

    printf("  PASS: sigmoid numerical stability\n");
}

int main() {
    printf("Running activation tests...\n");
    test_relu_forward();
    test_relu_backward();
    test_relu_gradient_check();
    test_sigmoid_forward();
    test_sigmoid_backward();
    test_sigmoid_gradient_check();
    test_sigmoid_numerical_stability();
    printf("All activation tests passed!\n");
    return 0;
}
