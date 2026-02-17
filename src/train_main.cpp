#include "models/autoencoder.h"
#include "nn/mse_loss.h"
#include "optim/adam.h"
#include "io/image_io.h"
#include "io/model_io.h"

#include <iostream>
#include <string>
#include <chrono>
#include <cstring>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <input_image> <output_model_path> [--epochs N] [--lr F]"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = argv[2];
    int epochs = 500;
    float lr = 0.001f;

    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lr = std::atof(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Load image
    std::cout << "Loading image: " << image_path << std::endl;
    Tensor input = ImageIO::load(image_path);

    std::cout << "Training for " << epochs << " epochs with lr=" << lr << std::endl;
    std::cout << std::endl;

    // Build model and optimizer
    Autoencoder model;
    auto params = model.parameters();
    Adam optimizer(params, lr);
    MSELoss loss_fn;

    std::cout << "Model parameters: " << params.size() << " tensors" << std::endl;
    size_t total_params = 0;
    for (const auto& p : params) {
        total_params += p.value->size();
    }
    std::cout << "Total trainable values: " << total_params << std::endl;
    std::cout << std::endl;

    // Training loop
    auto total_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::steady_clock::now();

        // Forward pass
        model.zero_gradients();
        Tensor output = model.forward(input);
        float loss = loss_fn.forward(output, input);

        // Backward pass
        Tensor grad = loss_fn.backward();
        model.backward(grad);

        // Update weights
        optimizer.step();

        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start).count();

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << "  loss=" << loss
                  << "  time=" << epoch_ms << "ms"
                  << std::endl;
    }

    auto total_end = std::chrono::steady_clock::now();
    auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(
        total_end - total_start).count();
    std::cout << std::endl;
    std::cout << "Training complete in " << total_sec << "s" << std::endl;

    // Save model
    ModelIO::save(model.parameters(), model_path);
    std::cout << "Model saved to " << model_path << std::endl;

    return 0;
}
