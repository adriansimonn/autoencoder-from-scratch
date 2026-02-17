#include "models/autoencoder.h"

// Architecture:
// Encoder: Input(12288) -> Dense(512) -> ReLU -> Dense(128) -> ReLU -> Dense(64) [latent]
// Decoder: Latent(64) -> Dense(128) -> ReLU -> Dense(512) -> ReLU -> Dense(12288) -> Sigmoid

Autoencoder::Autoencoder() {
    // Encoder layers (He init for ReLU layers)
    encoder_.add_layer(std::make_shared<DenseLayer>(12288, 512, InitMethod::He));
    encoder_.add_layer(std::make_shared<ReLU>());
    encoder_.add_layer(std::make_shared<DenseLayer>(512, 128, InitMethod::He));
    encoder_.add_layer(std::make_shared<ReLU>());
    encoder_.add_layer(std::make_shared<DenseLayer>(128, 64, InitMethod::He));

    // Decoder layers (He init for ReLU layers, Xavier for Sigmoid output)
    decoder_.add_layer(std::make_shared<DenseLayer>(64, 128, InitMethod::He));
    decoder_.add_layer(std::make_shared<ReLU>());
    decoder_.add_layer(std::make_shared<DenseLayer>(128, 512, InitMethod::He));
    decoder_.add_layer(std::make_shared<ReLU>());
    decoder_.add_layer(std::make_shared<DenseLayer>(512, 12288, InitMethod::Xavier));
    decoder_.add_layer(std::make_shared<Sigmoid>());
}

Tensor Autoencoder::forward(const Tensor& input) {
    Tensor latent = encoder_.forward(input);
    return decoder_.forward(latent);
}

Tensor Autoencoder::encode(const Tensor& input) {
    return encoder_.forward(input);
}

Tensor Autoencoder::decode(const Tensor& latent) {
    return decoder_.forward(latent);
}

Tensor Autoencoder::backward(const Tensor& grad_output) {
    Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

std::vector<Parameter> Autoencoder::parameters() {
    auto enc_params = encoder_.parameters();
    auto dec_params = decoder_.parameters();
    enc_params.insert(enc_params.end(), dec_params.begin(), dec_params.end());
    return enc_params;
}

void Autoencoder::zero_gradients() {
    encoder_.zero_gradients();
    decoder_.zero_gradients();
}
