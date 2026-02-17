#pragma once

#include <vector>
#include <cstddef>
#include <fstream>
#include <string>

class Tensor {
public:
    std::vector<float> data;
    size_t rows, cols;

    // Construction
    Tensor();
    Tensor(size_t rows, size_t cols);
    Tensor(size_t rows, size_t cols, float val);
    static Tensor from_vector(const std::vector<float>& data);

    // Element access
    float& operator()(size_t r, size_t c);
    const float& operator()(size_t r, size_t c) const;
    float& operator[](size_t i);
    const float& operator[](size_t i) const;
    size_t size() const;

    // Math operations (return new Tensors)
    static Tensor matmul(const Tensor& A, const Tensor& B);
    static Tensor transpose(const Tensor& A);
    static Tensor add(const Tensor& A, const Tensor& B);
    static Tensor subtract(const Tensor& A, const Tensor& B);
    static Tensor multiply(const Tensor& A, const Tensor& B);
    static Tensor scale(const Tensor& A, float scalar);
    static Tensor sqrt_elem(const Tensor& A);
    static Tensor divide_elem(const Tensor& A, const Tensor& B);

    // In-place operations
    void add_inplace(const Tensor& other);
    void scale_inplace(float scalar);
    void zero();

    // Initialization
    static Tensor randn(size_t rows, size_t cols, float mean, float stddev);
    static Tensor zeros(size_t rows, size_t cols);

    // Serialization
    void save(std::ofstream& out) const;
    static Tensor load(std::ifstream& in);
};
