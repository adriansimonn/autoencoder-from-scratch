#include "io/model_io.h"
#include <fstream>
#include <stdexcept>

void ModelIO::save(const std::vector<Parameter>& params, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    size_t num_params = params.size();
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    for (auto& p : params) {
        p.value->save(out);
    }
}

void ModelIO::load(std::vector<Parameter>& params, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    size_t num_params;
    in.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

    if (num_params != params.size()) {
        throw std::runtime_error("Parameter count mismatch: file has " +
            std::to_string(num_params) + ", model has " + std::to_string(params.size()));
    }

    for (size_t i = 0; i < num_params; ++i) {
        Tensor loaded = Tensor::load(in);
        if (loaded.rows != params[i].value->rows || loaded.cols != params[i].value->cols) {
            throw std::runtime_error("Shape mismatch for parameter " + std::to_string(i));
        }
        *params[i].value = loaded;
    }
}
