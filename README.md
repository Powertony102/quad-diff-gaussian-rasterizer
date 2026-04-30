# Quad-Diff-Gaussian-Rasterizer

A CUDA-accelerated differentiable Gaussian rasterizer with **quad box** tile culling for 3D Gaussian Splatting.

## Key Features

- **Quad Box Optimization**: Replaces circular bounding with quad boxes for more accurate tile-Gaussian intersection, especially for elongated Gaussians
- **Differentiable Rendering**: Full forward/backward pass for end-to-end training
- **CUDA Adam Optimizer**: Built-in sparse Adam update for Gaussian parameters

## Installation

```bash
git clone https://github.com/Powertony102/quad-diff-gaussian-rasterizer.git
cd quad-diff-gaussian-rasterizer
pip install .
```

## Acknowledgement

Based on [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) by Inria GRAPHDECO.
