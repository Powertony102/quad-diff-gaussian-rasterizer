# Quad-Diff-Gaussian-Rasterizer

A high-performance CUDA-accelerated differentiable Gaussian rasterization library with quad box optimization for 3D Gaussian Splatting. This is an enhanced version of the original differential Gaussian rasterization that incorporates advanced quad box techniques for improved tile-based rendering efficiency.

## Overview

This project implements a CUDA-accelerated differentiable rasterizer specifically designed for 3D Gaussian Splatting applications. The key innovation is the integration of **quad box optimization** that provides more precise tile-Gaussian intersection calculations compared to traditional circular bounding approaches.

### Key Features

- **Quad Box Optimization**: Advanced tile intersection algorithm using quad boxes instead of circular bounding
- **CUDA Acceleration**: Highly optimized CUDA kernels for maximum GPU performance
- **Differentiable Rendering**: Full backward pass support for neural network training
- **Memory Efficient**: Optimized memory management with safety limits for large scenes
- **Adam Optimizer Integration**: Built-in CUDA Adam optimizer for Gaussian parameters
- **Spherical Harmonics Support**: Efficient SH coefficient to RGB conversion

## Technical Architecture

### Quad Box

**Quad Box System** (`cuda_rasterizer/auxiliary.h`)
   - Advanced ellipse-to-quad box conversion
   - Optimized tile intersection generation
   - Tilt angle and eccentricity calculations

### Quad Box Innovation

The quad box system represents Gaussian ellipses using four rectangular boxes that provide:
- More accurate tile intersection detection
- Reduced over-estimation of affected tiles  
- Better performance for elongated Gaussians
- Improved memory efficiency

## Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/Powertony102/quad-diff-gaussian-rasterizer.git
cd quad-diff-gaussian-rasterizer

# Install the Python package
pip install .
```