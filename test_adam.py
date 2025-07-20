#!/usr/bin/env python3

"""
测试 SparseGaussianAdam 优化器的简单脚本
"""

import torch
import numpy as np
import sys
import os

# 添加路径以导入自定义的 rasterizer
sys.path.append('/Users/lixinze/Documents/DashGaussian/diff_gaussian_rasterization')

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    print("✓ 成功导入 SparseGaussianAdam")
except ImportError as e:
    print(f"✗ 导入 SparseGaussianAdam 失败: {e}")
    exit(1)

def test_adam_optimizer():
    """测试 Adam 优化器的基本功能"""
    print("\n开始测试 SparseGaussianAdam 优化器...")
    
    # 创建测试参数
    N = 100  # Gaussian 数量
    M = 3    # 每个 Gaussian 的参数维度
    
    # 创建测试张量
    param = torch.randn(N * M, requires_grad=True, device='cuda')
    
    # 创建优化器
    optimizer = SparseGaussianAdam([param], lr=0.01, eps=1e-8)
    
    # 模拟梯度
    param.grad = torch.randn_like(param)
    
    # 创建 visibility 掩码（只有一半的 Gaussian 是可见的）
    visibility = torch.zeros(N, dtype=torch.bool, device='cuda')
    visibility[:N//2] = True
    
    print(f"参数张量形状: {param.shape}")
    print(f"可见 Gaussian 数量: {visibility.sum().item()}/{N}")
    
    # 保存更新前的参数值
    param_before = param.clone()
    
    # 执行优化步骤
    try:
        optimizer.step(visibility, N)
        print("✓ 优化器步骤执行成功")
    except Exception as e:
        print(f"✗ 优化器步骤执行失败: {e}")
        return False
    
    # 检查参数是否发生变化
    param_changed = not torch.allclose(param, param_before)
    print(f"参数是否发生变化: {param_changed}")
    
    # 检查只有可见的 Gaussian 参数发生了变化
    param_diff = (param - param_before).view(N, M)
    visible_changed = torch.any(param_diff[:N//2] != 0)
    invisible_unchanged = torch.all(param_diff[N//2:] == 0)
    
    print(f"可见 Gaussian 参数已更新: {visible_changed}")
    print(f"不可见 Gaussian 参数未更新: {invisible_unchanged}")
    
    return visible_changed and invisible_unchanged

if __name__ == "__main__":
    if torch.cuda.is_available():
        success = test_adam_optimizer()
        if success:
            print("\n✓ 所有测试通过！SparseGaussianAdam 优化器工作正常。")
        else:
            print("\n✗ 测试失败！")
    else:
        print("CUDA 不可用，无法测试 CUDA 优化器")
