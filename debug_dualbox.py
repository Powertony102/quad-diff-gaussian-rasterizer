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
# auxiliary.h 方法实现
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

def compute_stretching_factor(theta, eccentricity):
    """Compute stretching factor based on tilt angle and eccentricity"""
    sin_2theta = math.sin(2.0 * theta)
    sin_2theta_sq = sin_2theta * sin_2theta
    
    e_sq = eccentricity * eccentricity
    e_4 = e_sq * e_sq
    
    stretch_modifier = e_4 / (4.0 * (1.0 - e_sq))
    return 1.0 / math.sqrt(1.0 + sin_2theta_sq * stretch_modifier)

def compute_extreme_points(con_o, disc, t, p):
    """Compute extreme points of ellipse using analytical, numerically-stable math"""
    A = con_o[0]
    B = con_o[1]
    C = con_o[2]
    
    ext = {
        'x_extremes': (p[0], p[0]),
        'y_extremes': (p[1], p[1]),
        'x_coords_at_y_extremes': (p[0], p[0]),
        'y_coords_at_x_extremes': (p[1], p[1])
    }
    
    if A <= 0.0 or C <= 0.0 or disc >= 0.0 or t <= 0.0:
        return ext
    
    denomX = A - (B * B) / C
    denomY = C - (B * B) / A
    
    if denomX <= 0.0 or denomY <= 0.0:
        return ext
    
    u_max = math.sqrt(t / denomX)
    v_max = math.sqrt(t / denomY)
    
    B_over_C = B / C
    B_over_A = B / A
    
    y_at_xmin = p[1] + B_over_C * u_max
    y_at_xmax = p[1] - B_over_C * u_max
    
    ext['x_extremes'] = (p[0] - u_max, p[0] + u_max)
    ext['y_coords_at_x_extremes'] = (y_at_xmin, y_at_xmax)
    
    x_at_ymin = p[0] + B_over_A * v_max
    x_at_ymax = p[0] - B_over_A * v_max
    
    ext['y_extremes'] = (p[1] - v_max, p[1] + v_max)
    ext['x_coords_at_y_extremes'] = (x_at_ymin, x_at_ymax)
    
    return ext

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

def debug_single_case():
    """调试单个测试用例"""
    print("=" * 80)
    print("调试单个测试用例")
    print("=" * 80)
    
    # 使用固定的测试用例以便调试
    center_x, center_y, a, b, theta = 10.0, 10.0, 3.0, 1.5, 45.0
    print(f"椭圆参数: 中心({center_x}, {center_y}), a={a}, b={b}, θ={theta}°")
    
    # 计算偏心率
    e = calculate_eccentricity(a, b)
    print(f"偏心率: e = {e:.3f}")
    
    # 计算边界矩形
    x_min, y_min, width, height = calculate_bounding_rectangle(center_x, center_y, a, b, theta)
    print(f"边界矩形: x_min={x_min:.2f}, y_min={y_min:.2f}, width={width:.2f}, height={height:.2f}")
    
    # 原始方法
    left_orig, right_orig = create_coverage_rectangles_original(
        center_x, center_y, x_min, y_min, width, height, theta, e)
    
    print(f"\n原始方法结果:")
    print(f"  左矩形: x={left_orig[0]:.2f}, y={left_orig[1]:.2f}, w={left_orig[2]:.2f}, h={left_orig[3]:.2f}")
    print(f"  右矩形: x={right_orig[0]:.2f}, y={right_orig[1]:.2f}, w={right_orig[2]:.2f}, h={right_orig[3]:.2f}")
    
    # auxiliary.h 方法
    # 计算 con_o
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
    
    print(f"\nAuxiliary方法参数:")
    print(f"  con_o: A={A:.6f}, B={B:.6f}, C={C:.6f}")
    
    disc = B * B - A * C
    threshold = opacity * 255.0
    t = 2.0 * math.log(threshold)
    
    print(f"  判别式: {disc:.6f}")
    print(f"  阈值 t: {t:.6f}")
    
    p = (center_x, center_y)
    extremes = compute_extreme_points(con_o, disc, t, p)
    
    print(f"\n极值点:")
    print(f"  x_extremes: {extremes['x_extremes']}")
    print(f"  y_extremes: {extremes['y_extremes']}")
    print(f"  x_coords_at_y_extremes: {extremes['x_coords_at_y_extremes']}")
    print(f"  y_coords_at_x_extremes: {extremes['y_coords_at_x_extremes']}")
    
    cov2d = (A, B, C)
    theta_rad = compute_tilt_angle(cov2d)
    eccentricity = compute_eccentricity_auxiliary(con_o)
    
    print(f"\n计算参数:")
    print(f"  theta_rad: {theta_rad:.6f} (弧度)")
    print(f"  eccentricity: {eccentricity:.6f}")
    
    left_aux, right_aux = construct_dual_boxes(extremes, p, theta_rad, eccentricity)
    
    print(f"\nAuxiliary方法结果:")
    print(f"  左矩形: x={left_aux[0]:.2f}, y={left_aux[1]:.2f}, w={left_aux[2]-left_aux[0]:.2f}, h={left_aux[3]-left_aux[1]:.2f}")
    print(f"  右矩形: x={right_aux[0]:.2f}, y={right_aux[1]:.2f}, w={right_aux[2]-right_aux[0]:.2f}, h={right_aux[3]-right_aux[1]:.2f}")
    
    # 计算扩展系数
    f_orig = f_function(e, math.radians(theta))
    f_aux = compute_stretching_factor(theta_rad, eccentricity)
    
    print(f"\n扩展系数比较:")
    print(f"  原始方法: f(e,θ) = {f_orig:.6f}")
    print(f"  Auxiliary方法: f(e,θ) = {f_aux:.6f}")
    print(f"  差异: {abs(f_orig - f_aux):.6f}")
    
    # 计算面积
    left_orig_area = left_orig[2] * left_orig[3]
    right_orig_area = right_orig[2] * right_orig[3]
    left_aux_area = (left_aux[2] - left_aux[0]) * (left_aux[3] - left_aux[1])
    right_aux_area = (right_aux[2] - right_aux[0]) * (right_aux[3] - right_aux[1])
    
    total_orig_area = left_orig_area + right_orig_area
    total_aux_area = left_aux_area + right_aux_area
    
    print(f"\n面积比较:")
    print(f"  原始方法总面积: {total_orig_area:.2f}")
    print(f"  Auxiliary方法总面积: {total_aux_area:.2f}")
    print(f"  面积差异: {abs(total_orig_area - total_aux_area):.2f}")
    
    # 问题分析
    print(f"\n问题分析:")
    print(f"  原始方法左矩形面积: {left_orig_area:.2f}")
    print(f"  Auxiliary方法左矩形面积: {left_aux_area:.2f}")
    print(f"  原始方法右矩形面积: {right_orig_area:.2f}")
    print(f"  Auxiliary方法右矩形面积: {right_aux_area:.2f}")
    
    # 检查极值点计算是否正确
    print(f"\n极值点验证:")
    print(f"  边界矩形: x_min={x_min:.2f}, x_max={x_min+width:.2f}, y_min={y_min:.2f}, y_max={y_min+height:.2f}")
    print(f"  Auxiliary极值点: x_min={extremes['x_extremes'][0]:.2f}, x_max={extremes['x_extremes'][1]:.2f}")
    print(f"  Auxiliary极值点: y_min={extremes['y_extremes'][0]:.2f}, y_max={extremes['y_extremes'][1]:.2f}")

if __name__ == "__main__":
    debug_single_case() 