#pragma once

#include "nn/layer.h"
#include <string>
#include <vector>

class ModelIO {
public:
    // Save parameters to binary file: [num_params][rows,cols,float_data]...
    static void save(const std::vector<Parameter>& params, const std::string& path);

    // Load parameters from binary file into existing parameter list
    static void load(std::vector<Parameter>& params, const std::string& path);
};
