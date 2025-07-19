#!/usr/bin/env python3
"""
测试invdepths功能的简单脚本
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, rasterize_gaussians
    print("✓ 成功导入diff_gaussian_rasterization模块")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_invdepths():
    """测试invdepths功能"""
    print("\n=== 测试invdepths功能 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    num_points = 1000
    image_height = 512
    image_width = 512
    
    # 创建3D高斯点
    means3D = torch.randn(num_points, 3, device=device) * 2.0
    means2D = torch.randn(num_points, 3, device=device)
    
    # 创建其他参数
    colors_precomp = torch.rand(num_points, 3, device=device)
    opacities = torch.sigmoid(torch.randn(num_points, 1, device=device))
    scales = torch.rand(num_points, 3, device=device) * 0.1
    rotations = torch.randn(num_points, 4, device=device)
    rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)  # 归一化四元数
    
    # 创建相机参数
    viewmatrix = torch.eye(4, device=device)
    viewmatrix[2, 3] = -5.0  # 相机位置
    
    projmatrix = torch.eye(4, device=device)
    projmatrix[0, 0] = 1.0
    projmatrix[1, 1] = 1.0
    projmatrix[2, 2] = -1.0
    projmatrix[2, 3] = -0.1
    projmatrix[3, 2] = -1.0
    
    # 创建光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=torch.tensor([0.0, 0.0, -5.0], device=device),
        prefiltered=False,
        debug=False
    )
    
    # 创建空的张量
    sh = torch.empty(0, device=device)
    cov3Ds_precomp = torch.empty(0, device=device)
    scores = torch.ones(num_points, device=device)
    
    try:
        # 调用光栅化函数
        print("调用rasterize_gaussians...")
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
        
        print("✓ 光栅化成功完成!")
        print(f"  - color shape: {color.shape}")
        print(f"  - radii shape: {radii.shape}")
        print(f"  - kernel_times shape: {kernel_times.shape}")
        print(f"  - invdepths shape: {invdepths.shape} (应该是 (1, H, W))")
        
        # 检查invdepths的值范围
        invdepths_min = invdepths.min().item()
        invdepths_max = invdepths.max().item()
        invdepths_mean = invdepths.mean().item()
        
        print(f"  - invdepths 范围: [{invdepths_min:.6f}, {invdepths_max:.6f}]")
        print(f"  - invdepths 均值: {invdepths_mean:.6f}")
        
        # 检查是否有非零值
        non_zero_count = (invdepths > 0).sum().item()
        total_pixels = invdepths.numel()
        print(f"  - 非零像素数量: {non_zero_count}/{total_pixels} ({non_zero_count/total_pixels*100:.2f}%)")
        
        # 验证维度
        expected_shape = (1, image_height, image_width)
        if invdepths.shape == expected_shape:
            print("✓ invdepths维度正确")
        else:
            print(f"⚠ invdepths维度错误: 期望 {expected_shape}, 实际 {invdepths.shape}")
            
        if non_zero_count > 0:
            print("✓ invdepths计算成功，包含有效的逆深度值")
        else:
            print("⚠ invdepths全为零，可能需要检查输入参数")
            
    except Exception as e:
        print(f"✗ 光栅化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_invdepths()
    if success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n❌ 测试失败!")
        sys.exit(1) 