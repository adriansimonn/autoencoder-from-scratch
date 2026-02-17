#include "models/autoencoder.h"
#include "nn/mse_loss.h"
#include "io/image_io.h"
#include "io/model_io.h"

#include <iostream>
#include <cmath>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_path> <input_image> <output_image>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];

    // Build model and load weights
    Autoencoder model;
    auto params = model.parameters();
    ModelIO::load(params, model_path);
    std::cout << "Loaded model from " << model_path << std::endl;

    // Load input image
    Tensor input = ImageIO::load(input_path);
    std::cout << "Loaded image: " << input_path << std::endl;

    // Encode to latent space
    Tensor latent = model.encode(input);

    // Compute latent vector statistics
    float lat_min = latent[0], lat_max = latent[0];
    float lat_sum = 0.0f;
    for (size_t i = 0; i < latent.size(); ++i) {
        float v = latent[i];
        lat_min = std::min(lat_min, v);
        lat_max = std::max(lat_max, v);
        lat_sum += v;
    }
    float lat_mean = lat_sum / static_cast<float>(latent.size());

    float lat_var_sum = 0.0f;
    for (size_t i = 0; i < latent.size(); ++i) {
        float diff = latent[i] - lat_mean;
        lat_var_sum += diff * diff;
    }
    float lat_std = std::sqrt(lat_var_sum / static_cast<float>(latent.size()));

    // Decode from latent space
    Tensor output = model.decode(latent);

    // Compute reconstruction loss
    MSELoss loss_fn;
    float loss = loss_fn.forward(output, input);

    // Save reconstruction
    ImageIO::save(output, output_path);

    // Print results
    std::cout << std::endl;
    std::cout << "Reconstruction loss (MSE): " << loss << std::endl;
    std::cout << std::endl;
    std::cout << "Latent vector (" << latent.size() << " dims):" << std::endl;
    std::cout << "  min:  " << lat_min << std::endl;
    std::cout << "  max:  " << lat_max << std::endl;
    std::cout << "  mean: " << lat_mean << std::endl;
    std::cout << "  std:  " << lat_std << std::endl;
    std::cout << std::endl;
    std::cout << "Saved reconstruction to " << output_path << std::endl;

    return 0;
}
