#include "nn/dense.h"
#include <cassert>
#include <cmath>
#include <cstdio>

static bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

void test_dense_forward() {
    // Create a dense layer 3 -> 2, set weights manually
    DenseLayer dense(3, 2, InitMethod::He);

    // Override weights for deterministic test
    auto params = dense.parameters();
    Tensor& W = *params[0].value;   // (3, 2)
    Tensor& b = *params[1].value;   // (1, 2)

    // W = [[1, 2], [3, 4], [5, 6]]
    W(0,0)=1; W(0,1)=2; W(1,0)=3; W(1,1)=4; W(2,0)=5; W(2,1)=6;
    // b = [0.1, 0.2]
    b(0,0)=0.1f; b(0,1)=0.2f;

    // input x = [1, 1, 1] -> y = [1,1,1]*W + b = [9+0.1, 12+0.2] = [9.1, 12.2]
    Tensor x(1, 3);
    x(0,0)=1; x(0,1)=1; x(0,2)=1;

    auto y = dense.forward(x);
    assert(y.rows == 1 && y.cols == 2);
    assert(approx(y(0,0), 9.1f));
    assert(approx(y(0,1), 12.2f));

    printf("  PASS: dense forward\n");
}

void test_dense_backward() {
    DenseLayer dense(3, 2, InitMethod::He);
    auto params = dense.parameters();
    Tensor& W = *params[0].value;
    Tensor& b = *params[1].value;

    W(0,0)=1; W(0,1)=2; W(1,0)=3; W(1,1)=4; W(2,0)=5; W(2,1)=6;
    b(0,0)=0.1f; b(0,1)=0.2f;

    Tensor x(1, 3);
    x(0,0)=1; x(0,1)=2; x(0,2)=3;

    auto y = dense.forward(x);
    // y = x*W + b = [1*1+2*3+3*5+0.1, 1*2+2*4+3*6+0.2] = [22.1, 28.2]
    assert(approx(y(0,0), 22.1f));
    assert(approx(y(0,1), 28.2f));

    // grad_output = [1, 1]
    Tensor grad(1, 2, 1.0f);
    auto dx = dense.backward(grad);

    // dx = grad * W^T = [1,1] * [[1,3,5],[2,4,6]] = [3, 7, 11]
    assert(dx.rows == 1 && dx.cols == 3);
    assert(approx(dx(0,0), 3.0f));
    assert(approx(dx(0,1), 7.0f));
    assert(approx(dx(0,2), 11.0f));

    // dW = x^T * grad = [[1],[2],[3]] * [[1,1]] = [[1,1],[2,2],[3,3]]
    Tensor& dW = *params[0].gradient;
    assert(approx(dW(0,0), 1.0f) && approx(dW(0,1), 1.0f));
    assert(approx(dW(1,0), 2.0f) && approx(dW(1,1), 2.0f));
    assert(approx(dW(2,0), 3.0f) && approx(dW(2,1), 3.0f));

    // db = grad = [1, 1]
    Tensor& db = *params[1].gradient;
    assert(approx(db(0,0), 1.0f) && approx(db(0,1), 1.0f));

    printf("  PASS: dense backward\n");
}

void test_dense_gradient_check() {
    // Numerical gradient check using finite differences
    DenseLayer dense(4, 3, InitMethod::He);
    Tensor x(1, 4);
    x[0]=0.5f; x[1]=-0.3f; x[2]=0.8f; x[3]=-0.1f;

    // Forward pass
    auto y = dense.forward(x);

    // Use sum of outputs as scalar loss
    Tensor grad_out(1, 3, 1.0f);
    dense.backward(grad_out);

    auto params = dense.parameters();
    float eps = 1e-4f;

    // Check weight gradients
    Tensor& W = *params[0].value;
    Tensor& dW = *params[0].gradient;
    for (size_t i = 0; i < W.size(); ++i) {
        float orig = W[i];

        W[i] = orig + eps;
        auto y_plus = dense.forward(x);
        float loss_plus = 0;
        for (size_t j = 0; j < y_plus.size(); ++j) loss_plus += y_plus[j];

        W[i] = orig - eps;
        auto y_minus = dense.forward(x);
        float loss_minus = 0;
        for (size_t j = 0; j < y_minus.size(); ++j) loss_minus += y_minus[j];

        W[i] = orig;

        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        assert(approx(dW[i], numerical, 1e-2f));
    }

    printf("  PASS: dense gradient check\n");
}

int main() {
    printf("Running dense layer tests...\n");
    test_dense_forward();
    test_dense_backward();
    test_dense_gradient_check();
    printf("All dense tests passed!\n");
    return 0;
}
