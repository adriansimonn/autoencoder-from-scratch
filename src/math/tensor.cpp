#include "math/tensor.h"
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

Tensor::Tensor() : rows(0), cols(0) {}

Tensor::Tensor(size_t rows, size_t cols)
    : data(rows * cols, 0.0f), rows(rows), cols(cols) {}

Tensor::Tensor(size_t rows, size_t cols, float val)
    : data(rows * cols, val), rows(rows), cols(cols) {}

Tensor Tensor::from_vector(const std::vector<float>& vec) {
    Tensor t(1, vec.size());
    t.data = vec;
    return t;
}

float& Tensor::operator()(size_t r, size_t c) {
    return data[r * cols + c];
}

const float& Tensor::operator()(size_t r, size_t c) const {
    return data[r * cols + c];
}

float& Tensor::operator[](size_t i) {
    return data[i];
}

const float& Tensor::operator[](size_t i) const {
    return data[i];
}

size_t Tensor::size() const {
    return data.size();
}

Tensor Tensor::matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("matmul: incompatible shapes (" +
            std::to_string(A.rows) + "x" + std::to_string(A.cols) + ") * (" +
            std::to_string(B.rows) + "x" + std::to_string(B.cols) + ")");
    }
    Tensor C(A.rows, B.cols);
    // i,k,j loop order for cache locality
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            float a_ik = A.data[i * A.cols + k];
            for (size_t j = 0; j < B.cols; ++j) {
                C.data[i * B.cols + j] += a_ik * B.data[k * B.cols + j];
            }
        }
    }
    return C;
}

Tensor Tensor::transpose(const Tensor& A) {
    Tensor T(A.cols, A.rows);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < A.cols; ++j) {
            T.data[j * A.rows + i] = A.data[i * A.cols + j];
        }
    }
    return T;
}

Tensor Tensor::add(const Tensor& A, const Tensor& B) {
    // Support broadcast: if B is (1, cols) and A is (rows, cols)
    if (A.rows == B.rows && A.cols == B.cols) {
        Tensor C(A.rows, A.cols);
        for (size_t i = 0; i < A.size(); ++i) {
            C.data[i] = A.data[i] + B.data[i];
        }
        return C;
    }
    if (B.rows == 1 && A.cols == B.cols) {
        Tensor C(A.rows, A.cols);
        for (size_t i = 0; i < A.rows; ++i) {
            for (size_t j = 0; j < A.cols; ++j) {
                C.data[i * A.cols + j] = A.data[i * A.cols + j] + B.data[j];
            }
        }
        return C;
    }
    if (A.rows == 1 && A.cols == B.cols) {
        Tensor C(B.rows, B.cols);
        for (size_t i = 0; i < B.rows; ++i) {
            for (size_t j = 0; j < B.cols; ++j) {
                C.data[i * B.cols + j] = A.data[j] + B.data[i * B.cols + j];
            }
        }
        return C;
    }
    throw std::invalid_argument("add: incompatible shapes");
}

Tensor Tensor::subtract(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("subtract: shapes must match");
    }
    Tensor C(A.rows, A.cols);
    for (size_t i = 0; i < A.size(); ++i) {
        C.data[i] = A.data[i] - B.data[i];
    }
    return C;
}

Tensor Tensor::multiply(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("multiply: shapes must match");
    }
    Tensor C(A.rows, A.cols);
    for (size_t i = 0; i < A.size(); ++i) {
        C.data[i] = A.data[i] * B.data[i];
    }
    return C;
}

Tensor Tensor::scale(const Tensor& A, float scalar) {
    Tensor C(A.rows, A.cols);
    for (size_t i = 0; i < A.size(); ++i) {
        C.data[i] = A.data[i] * scalar;
    }
    return C;
}

Tensor Tensor::sqrt_elem(const Tensor& A) {
    Tensor C(A.rows, A.cols);
    for (size_t i = 0; i < A.size(); ++i) {
        C.data[i] = std::sqrt(A.data[i]);
    }
    return C;
}

Tensor Tensor::divide_elem(const Tensor& A, const Tensor& B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("divide_elem: shapes must match");
    }
    Tensor C(A.rows, A.cols);
    for (size_t i = 0; i < A.size(); ++i) {
        C.data[i] = A.data[i] / B.data[i];
    }
    return C;
}

void Tensor::add_inplace(const Tensor& other) {
    if (rows == other.rows && cols == other.cols) {
        for (size_t i = 0; i < size(); ++i) {
            data[i] += other.data[i];
        }
    } else if (other.rows == 1 && cols == other.cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i * cols + j] += other.data[j];
            }
        }
    } else {
        throw std::invalid_argument("add_inplace: incompatible shapes");
    }
}

void Tensor::scale_inplace(float scalar) {
    for (size_t i = 0; i < size(); ++i) {
        data[i] *= scalar;
    }
}

void Tensor::zero() {
    std::fill(data.begin(), data.end(), 0.0f);
}

Tensor Tensor::randn(size_t rows, size_t cols, float mean, float stddev) {
    static std::mt19937 gen(42);
    std::normal_distribution<float> dist(mean, stddev);
    Tensor t(rows, cols);
    for (size_t i = 0; i < t.size(); ++i) {
        t.data[i] = dist(gen);
    }
    return t;
}

Tensor Tensor::zeros(size_t rows, size_t cols) {
    return Tensor(rows, cols, 0.0f);
}

void Tensor::save(std::ofstream& out) const {
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

Tensor Tensor::load(std::ifstream& in) {
    size_t r, c;
    in.read(reinterpret_cast<char*>(&r), sizeof(r));
    in.read(reinterpret_cast<char*>(&c), sizeof(c));
    Tensor t(r, c);
    in.read(reinterpret_cast<char*>(t.data.data()), t.data.size() * sizeof(float));
    return t;
}
