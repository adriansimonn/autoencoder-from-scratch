#pragma once

#include "math/tensor.h"
#include <string>

class ImageIO {
public:
    static constexpr int TARGET_SIZE = 64;
    static constexpr int CHANNELS = 3;
    static constexpr int FLAT_SIZE = TARGET_SIZE * TARGET_SIZE * CHANNELS;

    // Load image, resize to 64x64, normalize to [0,1], flatten to (1, 12288)
    static Tensor load(const std::string& path);

    // Denormalize from [0,1], reshape, save as PNG
    static void save(const Tensor& tensor, const std::string& path);
};
