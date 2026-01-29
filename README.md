# EfficientNetSeq2label

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/EfficientNetSeq2label.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/EfficientNetSeq2label.jl/dev/)
[![Build Status](https://github.com/kchu25/EfficientNetSeq2label.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/EfficientNetSeq2label.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/EfficientNetSeq2label.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/EfficientNetSeq2label.jl)

A Julia package for building EfficientNet-style convolutional neural networks for biological sequence-to-label prediction tasks. Supports DNA/RNA nucleotide sequences and amino acid (protein) sequences.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/kchu25/EfficientNetSeq2label.jl")
```

## Quick Start

### Basic Model Creation and Inference

```julia
using EfficientNetSeq2label
using Random

# Set seed for reproducibility
Random.seed!(42)

# Generate random hyperparameters using nucleotide-optimized ranges
ranges = nucleotide_ranges_simple()
hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)

# Create model for DNA sequences (alphabet size 4, sequence length 41, 10 outputs)
# Use pwm_dropout_p=0.0f0 for CPU-only mode
model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)

# Create random input: (alphabet_size, seq_length, 1, batch_size)
sequences = randn(Float32, 4, 41, 1, 8)

# Forward pass (use training=false for inference)
predictions = predict_from_sequences(model, sequences; training=false)
println("Predictions shape: ", size(predictions))  # (10, 8)
```

### Using Different Hyperparameter Ranges

```julia
using EfficientNetSeq2label
using Random

Random.seed!(123)

# For nucleotide sequences (DNA/RNA)
nuc_ranges = nucleotide_ranges()
hp_nuc = generate_random_hyperparameters(batch_size=64, ranges=nuc_ranges)
model_nuc = SeqCNN(hp_nuc, (4, 100), 5; use_cuda=false, pwm_dropout_p=0.0f0)

# For amino acid sequences (proteins) - alphabet size 20
aa_ranges = amino_acid_ranges()
hp_aa = generate_random_hyperparameters(batch_size=32, ranges=aa_ranges)
model_aa = SeqCNN(hp_aa, (20, 50), 3; use_cuda=false, pwm_dropout_p=0.0f0)

# Check model architecture
println("Nucleotide model layers: ", num_layers(hp_nuc))
println("Amino acid model layers: ", num_layers(hp_aa))
```

### Computing Loss

```julia
using EfficientNetSeq2label
using Random

Random.seed!(42)

# Create model
ranges = nucleotide_ranges_simple()
hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)

# Create dummy data
sequences = randn(Float32, 4, 41, 1, 16)
targets = randn(Float32, 10, 16)

# Compute Huber loss (handles NaN values automatically)
predictions = predict_from_sequences(model, sequences; training=false)
loss = huber_loss(predictions, targets)
println("Loss: ", loss)
```

### Extracting Intermediate Features

```julia
using EfficientNetSeq2label
using Random

Random.seed!(42)

ranges = nucleotide_ranges_simple()
hp = generate_random_hyperparameters(batch_size=32, ranges=ranges)
model = SeqCNN(hp, (4, 41), 10; use_cuda=false, pwm_dropout_p=0.0f0)

sequences = randn(Float32, 4, 41, 1, 4)

# Extract code at different layers
code_pwm = compute_code_at_layer(model, sequences, 0; training=false)   # Base PWM layer
code_l1 = compute_code_at_layer(model, sequences, 1; training=false)    # After 1st conv

println("PWM code shape: ", size(code_pwm))
println("Layer 1 code shape: ", size(code_l1))
```

### Using the Convenience Constructor

```julia
using EfficientNetSeq2label

# Create model with default nucleotide ranges (tanh final activation)
model = create_model_nucleotides((4, 41), 10, 32; use_cuda=false)

# Or with simple ranges for faster experimentation
model_simple = create_model_nucleotides_simple((4, 41), 10, 32; use_cuda=false)
```

## API Reference

### Hyperparameter Ranges

- `nucleotide_ranges()` - Optimized for DNA/RNA sequences
- `nucleotide_ranges_simple()` - Simplified ranges for testing
- `amino_acid_ranges()` - Optimized for protein sequences
- `nucleotide_ranges_fixed_pool_stride()` - Fixed pooling/stride for controlled experiments

### Model Creation

- `SeqCNN(hp, input_dims, output_dim; kwargs...)` - Create model from hyperparameters
- `create_model(input_dims, output_dim, batch_size; kwargs...)` - Convenience constructor
- `generate_random_hyperparameters(; batch_size, ranges)` - Generate random architecture

### Forward Pass

- `predict_from_sequences(model, sequences; training=false)` - Get predictions
- `compute_code_at_layer(model, sequences, layer; training=false)` - Get intermediate representations
- `extract_features(model, sequences)` - Extract CNN features

### Loss Functions

- `huber_loss(predictions, targets; delta=0.85)` - Robust loss with NaN handling
- `masked_mse(predictions, targets, mask)` - MSE on valid entries only
- `compute_training_loss(model, sequences, targets)` - Full training loss computation

### Utilities

- `model2cpu(model)` - Move model to CPU
- `model2gpu(model)` - Move model to GPU
- `num_layers(hp)` - Get number of layers in architecture

## License

MIT
