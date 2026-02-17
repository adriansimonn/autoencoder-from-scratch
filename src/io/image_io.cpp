#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#include "io/image_io.h"
#include <stdexcept>
#include <algorithm>

Tensor ImageIO::load(const std::string& path) {
    int w, h, c;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, CHANNELS);
    if (!raw) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    // Resize to 64x64
    std::vector<unsigned char> resized(TARGET_SIZE * TARGET_SIZE * CHANNELS);
    stbir_resize_uint8_linear(
        raw, w, h, 0,
        resized.data(), TARGET_SIZE, TARGET_SIZE, 0,
        static_cast<stbir_pixel_layout>(CHANNELS));
    stbi_image_free(raw);

    // Normalize to [0,1] and flatten
    Tensor tensor(1, FLAT_SIZE);
    for (int i = 0; i < FLAT_SIZE; ++i) {
        tensor[i] = static_cast<float>(resized[i]) / 255.0f;
    }
    return tensor;
}

void ImageIO::save(const Tensor& tensor, const std::string& path) {
    if (tensor.size() != static_cast<size_t>(FLAT_SIZE)) {
        throw std::runtime_error("Tensor size mismatch for image save: expected " +
            std::to_string(FLAT_SIZE) + ", got " + std::to_string(tensor.size()));
    }

    // Denormalize to [0, 255]
    std::vector<unsigned char> pixels(FLAT_SIZE);
    for (int i = 0; i < FLAT_SIZE; ++i) {
        float val = std::clamp(tensor[i], 0.0f, 1.0f);
        pixels[i] = static_cast<unsigned char>(val * 255.0f + 0.5f);
    }

    if (!stbi_write_png(path.c_str(), TARGET_SIZE, TARGET_SIZE, CHANNELS,
                        pixels.data(), TARGET_SIZE * CHANNELS)) {
        throw std::runtime_error("Failed to write image: " + path);
    }
}
