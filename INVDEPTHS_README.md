# InvDepths 功能实现说明

## 概述

本次修改为 `diff_gaussian_rasterization` 添加了 `invdepths`（逆深度期望值）的计算和传回功能。`invdepths` 是逆深度的期望值，在3D高斯光栅化过程中计算每个像素的逆深度期望。

## 修改内容

### 1. 数据结构修改

#### `rasterizer_impl.h`
- 在 `ImageState` 结构中添加了 `float* invdepths` 字段

#### `rasterizer_impl.cu`
- 在 `ImageState::fromChunk` 函数中添加了 `invdepths` 的内存分配

### 2. CUDA 核心计算修改

#### `forward.h`
- 在 `FORWARD::render` 函数签名中添加了 `float* invdepths` 参数

#### `forward.cu`
- 在 `renderCUDA` 模板函数中添加了 `depths` 和 `invdepths` 参数
- 实现了逆深度期望值的计算逻辑：
  ```cuda
  // invdepths = Σ (1/depth_i) × α_i × T_i
  expected_invdepth += (1.f / depths[collected_id[j]]) * alpha * T;
  ```
- 在 `FORWARD::render` 函数中传递了相应的参数

### 3. 接口层修改

#### `rasterizer.h`
- 在 `Rasterizer::forward` 函数签名中添加了 `float* invdepths` 参数

#### `rasterizer_impl.cu`
- 在 `Rasterizer::forward` 函数实现中添加了 `invdepths` 参数
- 在调用 `FORWARD::render` 时传递了 `geomState.depths` 和 `invdepths` 参数

### 4. C++ 接口修改

#### `rasterize_points.h`
- 修改了 `RasterizeGaussiansCUDA` 函数的返回类型，添加了 `invdepths` 张量

#### `rasterize_points.cu`
- 在 `RasterizeGaussiansCUDA` 函数中创建了 `invdepths` 张量
- 将 `invdepths` 传递给 `CudaRasterizer::Rasterizer::forward`
- 在返回值中包含 `invdepths`

### 5. Python 接口修改

#### `__init__.py`
- 修改了 `_RasterizeGaussians.forward` 方法，添加 `invdepths` 到返回值
- 修改了 `_RasterizeGaussians.backward` 方法，调整参数数量以匹配新的返回值
- 修改了 `GaussianRasterizer.forward` 方法，返回 `invdepths`

### 6. 测试文件修改

#### `test.cu`
- 更新了 `ImageState::fromChunk` 实现以包含 `invdepths` 内存分配

## 使用方法

### Python 接口

```python
from diff_gaussian_rasterization import GaussianRasterizationSettings, rasterize_gaussians

# 创建光栅化设置
raster_settings = GaussianRasterizationSettings(
    image_height=512,
    image_width=512,
    tanfovx=0.5,
    tanfovy=0.5,
    bg=torch.zeros(3),
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=0,
    campos=torch.tensor([0.0, 0.0, -5.0]),
    prefiltered=False,
    debug=False
)

# 调用光栅化函数
color, radii, kernel_times, invdepths = rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    scores,
    raster_settings
)

# invdepths 现在包含每个像素的逆深度期望值
print(f"invdepths shape: {invdepths.shape}")  # 应该是 (1, H, W)
print(f"invdepths range: [{invdepths.min():.6f}, {invdepths.max():.6f}]")
```

### C++ 接口

```cpp
// 调用 C++ 接口
auto result = RasterizeGaussiansCUDA(
    background, means3D, colors, opacity, scales, rotations,
    scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
    tan_fovx, tan_fovy, image_height, image_width,
    sh, degree, campos, prefiltered, debug
);

// 解包结果
auto [rendered, color, radii, kernel_times, geomBuffer, binningBuffer, imgBuffer, invdepths] = result;
```

## 计算原理

`invdepths` 的计算遵循以下公式：

**invdepths = Σ (1/depth_i) × α_i × T_i**

其中：
- `depth_i` 是第 i 个高斯点在相机坐标系中的深度（z 坐标）
- `α_i` 是第 i 个高斯点在该像素位置的透明度贡献
- `T_i` 是到第 i 个高斯点时的累积透明度
- 求和是对所有影响该像素的高斯点进行的

## 张量维度

`invdepths` 是一个3D张量，维度为 `{1, H, W}`，其中：
- 第一个维度固定为1
- H 是图像高度
- W 是图像宽度

这种设计与其他输出张量（如 `out_color` 的 `{NUM_CHANNELS, H, W}`）保持一致。

## 测试

运行测试脚本验证功能：

```bash
cd submodules/diff_gaussian_rasterization
python test_invdepths.py
```

## 注意事项

1. `invdepths` 的计算需要有效的深度值，确保输入的高斯点在相机视锥体内
2. 逆深度值可能包含零值或无穷大值，使用时需要进行适当的处理
3. 该功能与现有的光栅化流程完全兼容，不会影响其他功能
4. `invdepths` 是3D张量 `{1, H, W}`，使用时可以通过 `invdepths[0]` 或 `invdepths.squeeze(0)` 转换为2D张量

## 编译

确保在修改后重新编译CUDA扩展：

```bash
cd submodules/diff_gaussian_rasterization
python setup.py build_ext --inplace
```