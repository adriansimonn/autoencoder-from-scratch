#include "math/tensor.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>

static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_construction() {
    Tensor a(2, 3);
    assert(a.rows == 2 && a.cols == 3 && a.size() == 6);
    for (size_t i = 0; i < a.size(); ++i) assert(a[i] == 0.0f);

    Tensor b(2, 3, 1.5f);
    for (size_t i = 0; i < b.size(); ++i) assert(b[i] == 1.5f);

    auto c = Tensor::from_vector({1, 2, 3, 4});
    assert(c.rows == 1 && c.cols == 4);
    assert(c[0] == 1.0f && c[3] == 4.0f);

    printf("  PASS: construction\n");
}

void test_element_access() {
    Tensor a(2, 3);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    assert(a(0, 0) == 1 && a(1, 2) == 6);
    assert(a[0] == 1 && a[5] == 6);
    printf("  PASS: element access\n");
}

void test_matmul() {
    // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] * [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    Tensor A(2, 2);
    A(0,0)=1; A(0,1)=2; A(1,0)=3; A(1,1)=4;
    Tensor B(2, 2);
    B(0,0)=5; B(0,1)=6; B(1,0)=7; B(1,1)=8;

    auto C = Tensor::matmul(A, B);
    assert(C.rows == 2 && C.cols == 2);
    assert(approx(C(0,0), 19) && approx(C(0,1), 22));
    assert(approx(C(1,0), 43) && approx(C(1,1), 50));

    // Non-square: (1,3) * (3,2) = (1,2)
    Tensor D(1, 3);
    D(0,0)=1; D(0,1)=2; D(0,2)=3;
    Tensor E(3, 2);
    E(0,0)=1; E(0,1)=2; E(1,0)=3; E(1,1)=4; E(2,0)=5; E(2,1)=6;
    auto F = Tensor::matmul(D, E);
    assert(F.rows == 1 && F.cols == 2);
    assert(approx(F(0,0), 22) && approx(F(0,1), 28));

    printf("  PASS: matmul\n");
}

void test_transpose() {
    Tensor A(2, 3);
    A(0,0)=1; A(0,1)=2; A(0,2)=3;
    A(1,0)=4; A(1,1)=5; A(1,2)=6;
    auto T = Tensor::transpose(A);
    assert(T.rows == 3 && T.cols == 2);
    assert(approx(T(0,0), 1) && approx(T(0,1), 4));
    assert(approx(T(1,0), 2) && approx(T(1,1), 5));
    assert(approx(T(2,0), 3) && approx(T(2,1), 6));
    printf("  PASS: transpose\n");
}

void test_add_broadcast() {
    // Same shape
    Tensor A(2, 3, 1.0f);
    Tensor B(2, 3, 2.0f);
    auto C = Tensor::add(A, B);
    for (size_t i = 0; i < C.size(); ++i) assert(approx(C[i], 3.0f));

    // Broadcast: (2,3) + (1,3)
    Tensor bias(1, 3);
    bias[0] = 10; bias[1] = 20; bias[2] = 30;
    auto D = Tensor::add(A, bias);
    assert(approx(D(0,0), 11) && approx(D(0,1), 21) && approx(D(0,2), 31));
    assert(approx(D(1,0), 11) && approx(D(1,1), 21) && approx(D(1,2), 31));

    printf("  PASS: add with broadcast\n");
}

void test_elementwise_ops() {
    Tensor A(1, 4);
    A[0]=1; A[1]=4; A[2]=9; A[3]=16;
    Tensor B(1, 4);
    B[0]=1; B[1]=2; B[2]=3; B[3]=4;

    auto sub = Tensor::subtract(A, B);
    assert(approx(sub[0], 0) && approx(sub[1], 2) && approx(sub[2], 6) && approx(sub[3], 12));

    auto mul = Tensor::multiply(A, B);
    assert(approx(mul[0], 1) && approx(mul[1], 8) && approx(mul[2], 27) && approx(mul[3], 64));

    auto sc = Tensor::scale(B, 3.0f);
    assert(approx(sc[0], 3) && approx(sc[1], 6) && approx(sc[2], 9) && approx(sc[3], 12));

    auto sq = Tensor::sqrt_elem(A);
    assert(approx(sq[0], 1) && approx(sq[1], 2) && approx(sq[2], 3) && approx(sq[3], 4));

    auto dv = Tensor::divide_elem(A, B);
    assert(approx(dv[0], 1) && approx(dv[1], 2) && approx(dv[2], 3) && approx(dv[3], 4));

    printf("  PASS: elementwise ops\n");
}

void test_inplace() {
    Tensor A(1, 3, 1.0f);
    Tensor B(1, 3, 2.0f);
    A.add_inplace(B);
    for (size_t i = 0; i < 3; ++i) assert(approx(A[i], 3.0f));

    A.scale_inplace(2.0f);
    for (size_t i = 0; i < 3; ++i) assert(approx(A[i], 6.0f));

    A.zero();
    for (size_t i = 0; i < 3; ++i) assert(approx(A[i], 0.0f));

    printf("  PASS: in-place ops\n");
}

void test_randn() {
    auto R = Tensor::randn(100, 100, 0.0f, 1.0f);
    assert(R.rows == 100 && R.cols == 100);
    float sum = 0;
    for (size_t i = 0; i < R.size(); ++i) sum += R[i];
    float mean = sum / R.size();
    // Mean should be roughly 0 for 10000 samples
    assert(std::fabs(mean) < 0.1f);
    printf("  PASS: randn\n");
}

void test_save_load() {
    Tensor A(3, 4);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<float>(i) * 0.5f;

    {
        std::ofstream out("/tmp/test_tensor.bin", std::ios::binary);
        A.save(out);
    }
    {
        std::ifstream in("/tmp/test_tensor.bin", std::ios::binary);
        auto B = Tensor::load(in);
        assert(B.rows == 3 && B.cols == 4);
        for (size_t i = 0; i < A.size(); ++i) {
            assert(approx(A[i], B[i]));
        }
    }
    printf("  PASS: save/load round-trip\n");
}

int main() {
    printf("Running tensor tests...\n");
    test_construction();
    test_element_access();
    test_matmul();
    test_transpose();
    test_add_broadcast();
    test_elementwise_ops();
    test_inplace();
    test_randn();
    test_save_load();
    printf("All tensor tests passed!\n");
    return 0;
}
