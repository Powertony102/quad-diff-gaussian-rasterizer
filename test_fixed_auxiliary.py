import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import random
import math

def generate_ellipse_in_first_quadrant(grid_size=20):
    """Generate an ellipse within the first quadrant"""
    while True:
        center_x = random.uniform(5, grid_size - 5)
        center_y = random.uniform(5, grid_size - 5)
        a = random.uniform(2, 5)
        b = random.uniform(1, a)
        theta = random.uniform(0, 180)
        
        theta_rad = math.radians(theta)
        t_values = np.linspace(0, 2*np.pi, 1000)
        x_points = center_x + a * np.cos(t_values) * np.cos(theta_rad) - b * np.sin(t_values) * np.sin(theta_rad)
        y_points = center_y + a * np.cos(t_values) * np.sin(theta_rad) + b * np.sin(t_values) * np.cos(theta_rad)
        
        if np.min(x_points) > 0 and np.min(y_points) > 0:
            return center_x, center_y, a, b, theta

def calculate_bounding_rectangle(center_x, center_y, a, b, theta):
    """Calculate the bounding rectangle of the ellipse"""
    theta_rad = math.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    dx = np.sqrt((a * cos_theta)**2 + (b * sin_theta)**2)
    dy = np.sqrt((a * sin_theta)**2 + (b * cos_theta)**2)
    
    x_min = center_x - dx
    x_max = center_x + dx
    y_min = center_y - dy
    y_max = center_y + dy
    
    return x_min, y_min, x_max - x_min, y_max - y_min

def calculate_eccentricity(a, b):
    """Calculate ellipse eccentricity"""
    if a > b:
        e = np.sqrt(1 - (b/a)**2)
    else:
        e = np.sqrt(1 - (a/b)**2)
    return e

def f_function(e, theta_rad):
    """Calculate extension function f(e,theta)"""
    return 1 / np.sqrt(1 + (e**4 / (4 * (1 - e**2))) * (np.sin(2 * theta_rad))**2)

# ============================================================================
# 修复后的 auxiliary.h 方法实现
# ============================================================================

def compute_tilt_angle(cov2d):
    """Compute tilt angle θ using covariance matrix eigenvalue approach"""
    numerator = 2.0 * cov2d[1]
    denominator = cov2d[0] - cov2d[2]
    return 0.5 * math.atan2(numerator, denominator)

def compute_eccentricity_auxiliary(con_o):
    """Compute eccentricity of a 2D Gaussian ellipse from auxiliary.h"""
    A = con_o[0]
    B = con_o[1]
    C = con_o[2]
    
    diff_AC = A - C
    term_under_sqrt = diff_AC * diff_AC + 4.0 * B * B
    term_sqrt = math.sqrt(term_under_sqrt)
    
    sum_AC = A + C
    lambda_max = (sum_AC + term_sqrt) / 2.0
    lambda_min = (sum_AC - term_sqrt) / 2.0
    
    ratio = lambda_min / lambda_max
    return math.sqrt(1.0 - ratio)

def compute_bounding_rectangle_auxiliary(center, a, b, theta_rad):
    """Compute bounding rectangle of ellipse using geometric approach"""
    # Calculate ellipse boundaries after rotation
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    
    # Ellipse boundary calculation
    dx = math.sqrt((a * cos_theta) * (a * cos_theta) + (b * sin_theta) * (b * sin_theta))
    dy = math.sqrt((a * sin_theta) * (a * sin_theta) + (b * cos_theta) * (b * cos_theta))
    
    x_min = center[0] - dx
    x_max = center[0] + dx
    y_min = center[1] - dy
    y_max = center[1] + dy
    
    return {
        'x_extremes': (x_min, x_max),
        'y_extremes': (y_min, y_max),
        'x_coords_at_y_extremes': (x_min, x_max),
        'y_coords_at_x_extremes': (y_min, y_max)
    }

def construct_dual_boxes(extremes, center, theta, eccentricity):
    """Construct dual asymmetric AABBs using extreme points and center"""
    snug_min_x = extremes['x_extremes'][0]
    snug_max_x = extremes['x_extremes'][1]
    snug_min_y = extremes['y_extremes'][0]
    snug_max_y = extremes['y_extremes'][1]
    
    e_sq = eccentricity * eccentricity
    if e_sq >= 1.0:
        e_sq = 0.999
    sin_2theta = math.sin(2.0 * theta)
    sin_2theta_sq = sin_2theta * sin_2theta
    stretch_factor = 1.0 / math.sqrt(1.0 + (e_sq * e_sq / (4.0 * (1.0 - e_sq))) * sin_2theta_sq)
    
    if theta >= 0:
        left_rect_x = snug_min_x
        left_rect_y = snug_min_y
        left_rect_width = center[0] - snug_min_x
        left_rect_height = center[1] - snug_min_y
        
        right_rect_x = center[0]
        right_rect_y = center[1]
        right_rect_width = snug_max_x - center[0]
        right_rect_height = snug_max_y - center[1]
    else:
        left_rect_x = snug_min_x
        left_rect_y = center[1]
        left_rect_width = center[0] - snug_min_x
        left_rect_height = snug_max_y - center[1]
        
        right_rect_x = center[0]
        right_rect_y = snug_min_y
        right_rect_width = snug_max_x - center[0]
        right_rect_height = center[1] - snug_min_y
    
    if left_rect_width > 0:
        left_extension = left_rect_width * stretch_factor
        left_rect_width += left_extension
    
    if right_rect_width > 0:
        right_extension = right_rect_width * stretch_factor
        right_rect_x -= right_extension
        right_rect_width += right_extension
    
    left_box = (left_rect_x, left_rect_y, left_rect_x + left_rect_width, left_rect_y + left_rect_height)
    right_box = (right_rect_x, right_rect_y, right_rect_x + right_rect_width, right_rect_y + right_rect_height)
    
    return left_box, right_box

def create_coverage_rectangles_original(center_x, center_y, x_min, y_min, width, height, theta, e):
    """Create coverage rectangles using original method"""
    theta_rad = math.radians(theta)
    
    x_max = x_min + width
    y_max = y_min + height
    
    if 0 <= theta <= 90:
        left_rect_x = x_min
        left_rect_y = y_min
        left_rect_width = center_x - x_min
        left_rect_height = center_y - y_min
        
        right_rect_x = center_x
        right_rect_y = center_y
        right_rect_width = x_max - center_x
        right_rect_height = y_max - center_y
    else:
        left_rect_x = x_min
        left_rect_y = center_y
        left_rect_width = center_x - x_min
        left_rect_height = y_max - center_y
        
        right_rect_x = center_x
        right_rect_y = y_min
        right_rect_width = x_max - center_x
        right_rect_height = center_y - y_min
    
    f_val = f_function(e, theta_rad)
    
    left_extension = left_rect_width * f_val
    left_rect_width += left_extension
    
    right_extension = right_rect_width * f_val
    right_rect_x -= right_extension
    right_rect_width += right_extension
    
    return (left_rect_x, left_rect_y, left_rect_width, left_rect_height), \
           (right_rect_x, right_rect_y, right_rect_width, right_rect_height)

def calculate_display_bounds(left_rect_params, right_rect_params, x_min, y_min, width, height):
    """Calculate the optimal display bounds for the figure"""
    left_x_min = left_rect_params[0]
    left_x_max = left_rect_params[0] + left_rect_params[2]
    left_y_min = left_rect_params[1]
    left_y_max = left_rect_params[1] + left_rect_params[3]
    
    right_x_min = right_rect_params[0]
    right_x_max = right_rect_params[0] + right_rect_params[2]
    right_y_min = right_rect_params[1]
    right_y_max = right_rect_params[1] + right_rect_params[3]
    
    bound_x_min = x_min
    bound_x_max = x_min + width
    bound_y_min = y_min
    bound_y_max = y_min + height
    
    overall_x_min = min(left_x_min, right_x_min, bound_x_min)
    overall_x_max = max(left_x_max, right_x_max, bound_x_max)
    overall_y_min = min(left_y_min, right_y_min, bound_y_min)
    overall_y_max = max(left_y_max, right_y_max, bound_y_max)
    
    padding_x = (overall_x_max - overall_x_min) * 0.1
    padding_y = (overall_y_max - overall_y_min) * 0.1
    
    return (overall_x_min - padding_x, overall_x_max + padding_x,
            overall_y_min - padding_y, overall_y_max + padding_y)

def test_fixed_auxiliary():
    """测试修复后的auxiliary.h方法"""
    print("=" * 80)
    print("测试修复后的 auxiliary.h 方法")
    print("=" * 80)
    
    # 生成测试用例
    test_cases = []
    for i in range(3):
        center_x, center_y, a, b, theta = generate_ellipse_in_first_quadrant(30)
        test_cases.append((center_x, center_y, a, b, theta))
    
    results = []
    
    # 创建可视化图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('修复后的 auxiliary.h DualBox 创建方法验证', fontsize=16, fontweight='bold')
    
    for i, (center_x, center_y, a, b, theta) in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"  椭圆参数: 中心({center_x:.2f}, {center_y:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.1f}°")
        
        # 计算偏心率
        e = calculate_eccentricity(a, b)
        print(f"  偏心率: e = {e:.3f}")
        
        # 计算边界矩形
        x_min, y_min, width, height = calculate_bounding_rectangle(center_x, center_y, a, b, theta)
        
        # 原始方法
        left_orig, right_orig = create_coverage_rectangles_original(
            center_x, center_y, x_min, y_min, width, height, theta, e)
        
        # 修复后的 auxiliary.h 方法
        # 计算 con_o (从椭圆参数推导)
        cos_theta = math.cos(math.radians(theta))
        sin_theta = math.sin(math.radians(theta))
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        Sigma = np.array([[a*a, 0], [0, b*b]])
        Sigma_rotated = R @ Sigma @ R.T
        Sigma_inv = np.linalg.inv(Sigma_rotated)
        
        A = Sigma_inv[0, 0]
        B = Sigma_inv[0, 1]
        C = Sigma_inv[1, 1]
        opacity = 0.5
        con_o = (A, B, C, opacity)
        
        # 计算倾斜角和偏心率
        cov2d = (A, B, C)
        theta_rad = compute_tilt_angle(cov2d)
        eccentricity = compute_eccentricity_auxiliary(con_o)
        
        # 从 con_o 提取 a 和 b
        det_inv = A * C - B * B
        cov_xx = C / det_inv
        cov_xy = -B / det_inv
        cov_yy = A / det_inv
        
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        discriminant = trace * trace - 4.0 * det
        sqrt_disc = math.sqrt(discriminant)
        
        lambda_max = (trace + sqrt_disc) / 2.0
        lambda_min = (trace - sqrt_disc) / 2.0
        
        a_extracted = math.sqrt(lambda_max)
        b_extracted = math.sqrt(lambda_min)
        
        print(f"  提取的参数: a={a_extracted:.6f}, b={b_extracted:.6f}")
        print(f"  原始参数: a={a:.6f}, b={b:.6f}")
        print(f"  参数差异: a_diff={abs(a-a_extracted):.6f}, b_diff={abs(b-b_extracted):.6f}")
        
        # 使用边界矩形
        extremes = compute_bounding_rectangle_auxiliary((center_x, center_y), a_extracted, b_extracted, theta_rad)
        
        left_fixed, right_fixed = construct_dual_boxes(extremes, (center_x, center_y), theta_rad, eccentricity)
        
        # 计算扩展系数
        f_orig = f_function(e, math.radians(theta))
        f_fixed = 1.0 / math.sqrt(1.0 + (eccentricity**4 / (4.0 * (1.0 - eccentricity**2))) * (math.sin(2.0 * theta_rad))**2)
        
        print(f"  原始方法扩展系数: f(e,θ) = {f_orig:.6f}")
        print(f"  修复后扩展系数: f(e,θ) = {f_fixed:.6f}")
        print(f"  扩展系数差异: {abs(f_orig - f_fixed):.6f}")
        
        # 计算矩形面积
        left_orig_area = left_orig[2] * left_orig[3]
        right_orig_area = right_orig[2] * right_orig[3]
        left_fixed_area = (left_fixed[2] - left_fixed[0]) * (left_fixed[3] - left_fixed[1])
        right_fixed_area = (right_fixed[2] - right_fixed[0]) * (right_fixed[3] - right_fixed[1])
        
        total_orig_area = left_orig_area + right_orig_area
        total_fixed_area = left_fixed_area + right_fixed_area
        
        print(f"  原始方法总面积: {total_orig_area:.2f}")
        print(f"  修复后总面积: {total_fixed_area:.2f}")
        print(f"  面积差异: {abs(total_orig_area - total_fixed_area):.2f}")
        
        results.append({
            'case': i+1,
            'f_orig': f_orig,
            'f_fixed': f_fixed,
            'f_diff': abs(f_orig - f_fixed),
            'area_orig': total_orig_area,
            'area_fixed': total_fixed_area,
            'area_diff': abs(total_orig_area - total_fixed_area),
            'a_diff': abs(a-a_extracted),
            'b_diff': abs(b-b_extracted)
        })
        
        # 可视化当前测试用例
        ax = axes[i]
        
        # 绘制椭圆
        ellipse = Ellipse((center_x, center_y), 2*a, 2*b, angle=theta, 
                         facecolor='lightblue', edgecolor='blue', alpha=0.6, linewidth=2)
        ax.add_patch(ellipse)
        
        # 绘制边界矩形
        bound_rect = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
        ax.add_patch(bound_rect)
        
        # 绘制原始方法的覆盖矩形
        left_orig_rect = patches.Rectangle((left_orig[0], left_orig[1]), left_orig[2], left_orig[3], 
                                         linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
        right_orig_rect = patches.Rectangle((right_orig[0], right_orig[1]), right_orig[2], right_orig[3], 
                                          linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(left_orig_rect)
        ax.add_patch(right_orig_rect)
        
        # 绘制修复后方法的覆盖矩形
        left_fixed_rect = patches.Rectangle((left_fixed[0], left_fixed[1]), 
                                          left_fixed[2] - left_fixed[0], left_fixed[3] - left_fixed[1], 
                                          linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.3)
        right_fixed_rect = patches.Rectangle((right_fixed[0], right_fixed[1]), 
                                           right_fixed[2] - right_fixed[0], right_fixed[3] - right_fixed[1], 
                                           linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.3)
        ax.add_patch(left_fixed_rect)
        ax.add_patch(right_fixed_rect)
        
        # 绘制中心点
        ax.plot(center_x, center_y, 'ko', markersize=8, label='椭圆中心')
        
        # 设置图表属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'测试用例 {i+1}\n中心({center_x:.1f}, {center_y:.1f}), a={a:.1f}, b={b:.1f}, θ={theta:.1f}°')
        ax.grid(True, alpha=0.3)
        ax.legend(['椭圆中心', '椭圆', '边界矩形', '原始覆盖矩形', '修复后覆盖矩形'], 
                 loc='upper right', fontsize=8)
        
        # 设置坐标轴范围
        display_bounds = calculate_display_bounds(left_orig, right_orig, x_min, y_min, width, height)
        ax.set_xlim(display_bounds[0], display_bounds[1])
        ax.set_ylim(display_bounds[2], display_bounds[3])
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('auxiliary_fix_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计分析
    print("\n" + "=" * 80)
    print("统计分析结果:")
    print("=" * 80)
    
    f_diffs = [r['f_diff'] for r in results]
    area_diffs = [r['area_diff'] for r in results]
    a_diffs = [r['a_diff'] for r in results]
    b_diffs = [r['b_diff'] for r in results]
    
    print(f"扩展系数差异统计:")
    print(f"  平均差异: {np.mean(f_diffs):.6f}")
    print(f"  最大差异: {np.max(f_diffs):.6f}")
    print(f"  最小差异: {np.min(f_diffs):.6f}")
    print(f"  标准差: {np.std(f_diffs):.6f}")
    
    print(f"\n面积差异统计:")
    print(f"  平均差异: {np.mean(area_diffs):.2f}")
    print(f"  最大差异: {np.max(area_diffs):.2f}")
    print(f"  最小差异: {np.min(area_diffs):.2f}")
    print(f"  标准差: {np.std(area_diffs):.2f}")
    
    print(f"\n参数提取差异统计:")
    print(f"  a参数平均差异: {np.mean(a_diffs):.6f}")
    print(f"  b参数平均差异: {np.mean(b_diffs):.6f}")
    
    # 检查是否修复成功
    max_f_diff = np.max(f_diffs)
    max_area_diff = np.max(area_diffs)
    max_param_diff = max(np.max(a_diffs), np.max(b_diffs))
    
    if max_f_diff < 1e-6 and max_area_diff < 1e-2 and max_param_diff < 1e-6:
        print(f"\n✅ 修复成功! auxiliary.h 中的 DualBox 创建方法现在与原始方法完全一致!")
        print(f"   扩展系数最大差异: {max_f_diff:.6f} < 1e-6")
        print(f"   面积最大差异: {max_area_diff:.2f} < 1e-2")
        print(f"   参数最大差异: {max_param_diff:.6f} < 1e-6")
    else:
        print(f"\n❌ 修复可能存在问题!")
        print(f"   扩展系数最大差异: {max_f_diff:.6f}")
        print(f"   面积最大差异: {max_area_diff:.2f}")
        print(f"   参数最大差异: {max_param_diff:.6f}")
    
    return results

if __name__ == "__main__":
    results = test_fixed_auxiliary() 