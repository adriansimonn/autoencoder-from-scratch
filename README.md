# Autoencoder From Scratch

A fully-connected autoencoder implemented in pure C++ with no ML frameworks. All neural network primitives (matrix math, layers, activations, backpropagation, and the Adam optimizer) are built from scratch. The only external dependency is [stb](https://github.com/nothings/stb) (header-only) for image I/O.

Given an image, the autoencoder encodes it into a 64-dimensional latent space, decodes it back, and saves the reconstruction.

## Architecture

```
Encoder: Input(12288) -> Dense(512) -> ReLU -> Dense(128) -> ReLU -> Dense(64) [latent]
Decoder: Latent(64) -> Dense(128) -> ReLU -> Dense(512) -> ReLU -> Dense(12288) -> Sigmoid
```

- **Input**: Any image, resized to 64x64x3 (RGB), normalized to [0,1], flattened to 12,288 dimensions
- **Latent space**: 64 dimensions (~192:1 compression)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001, beta1=0.9, beta2=0.999)
- **Weight init**: He for ReLU layers, Xavier for Sigmoid output layer
- **Total parameters**: ~12.7M

## Building

Requires CMake 3.16+ and a C++17 compiler.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

### Train

Train the autoencoder on a single image:

```bash
./build/train <input_image> <output_model_path> [--epochs N] [--lr F]
```

Example:

```bash
./build/train images/sample_01.jpg model.bin --epochs 500
```

Default: 500 epochs, lr=0.001. Training prints per-epoch loss and timing.

### Reconstruct

Load a trained model and reconstruct an image:

```bash
./build/reconstruct <model_path> <input_image> <output_image>
```

Example:

```bash
./build/reconstruct model.bin images/sample_01.jpg output.png
```

Prints reconstruction loss (MSE) and latent vector statistics (min, max, mean, std).

## Tests

```bash
cd build && ctest
```

Runs unit tests for tensor math, dense layers, activations, and network convergence.

## Project Structure

```
src/
  math/     Tensor class: matrix ops, serialization
  nn/       Dense, ReLU, Sigmoid layers, MSE loss, Network container
  optim/    Adam optimizer
  io/       Image loading/saving (stb), model serialization
  models/   Autoencoder (encoder + decoder wiring)
test/       Unit tests
third_party/stb/  stb image headers
```
