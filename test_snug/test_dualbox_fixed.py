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
# auxiliary.h 方法实现（修正版本）
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

def main():
    # Set up the figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Generate ellipse with larger size
    center_x, center_y, a, b, theta = generate_ellipse_in_first_quadrant(30)
    
    # Calculate eccentricity
    e = calculate_eccentricity(a, b)
    
    # Calculate bounding rectangle
    x_min, y_min, width, height = calculate_bounding_rectangle(center_x, center_y, a, b, theta)
    
    # ============================================================================
    # 左侧图：原始 test.py 方法
    # ============================================================================
    
    # Create coverage rectangles using original method
    left_rect_params, right_rect_params = create_coverage_rectangles_original(
        center_x, center_y, x_min, y_min, width, height, theta, e)
    
    # Calculate optimal display bounds
    x_min_display, x_max_display, y_min_display, y_max_display = calculate_display_bounds(
        left_rect_params, right_rect_params, x_min, y_min, width, height)
    
    # Set up adaptive grid and limits
    ax1.set_xlim(x_min_display, x_max_display)
    ax1.set_ylim(y_min_display, y_max_display)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Draw large rectangle (light blue)
    bounding_rect = patches.Rectangle((x_min, y_min), width, height, 
                                    linewidth=2, edgecolor='blue', 
                                    facecolor='lightblue', alpha=0.5)
    ax1.add_patch(bounding_rect)
    
    # Draw left rectangle (light yellow)
    left_rect = patches.Rectangle((left_rect_params[0], left_rect_params[1]), 
                                left_rect_params[2], left_rect_params[3],
                                linewidth=2, edgecolor='orange', 
                                facecolor='lightyellow', alpha=0.7)
    ax1.add_patch(left_rect)
    
    # Draw right rectangle (light green)
    right_rect = patches.Rectangle((right_rect_params[0], right_rect_params[1]), 
                                 right_rect_params[2], right_rect_params[3],
                                 linewidth=2, edgecolor='green', 
                                 facecolor='lightgreen', alpha=0.7)
    ax1.add_patch(right_rect)
    
    # Draw ellipse
    ellipse = Ellipse((center_x, center_y), 2*a, 2*b, angle=theta,
                     linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(ellipse)
    
    # Mark ellipse center
    ax1.plot(center_x, center_y, 'ro', markersize=8, label='Ellipse Center')
    
    # Add title and labels
    ax1.set_title(f'Original Method (test.py)\nEllipse: a={a:.2f}, b={b:.2f}, θ={theta:.1f}°, e={e:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Axis', fontsize=10)
    ax1.set_ylabel('Y Axis', fontsize=10)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', alpha=0.5, label='Bounding Rectangle'),
        patches.Patch(color='lightyellow', alpha=0.7, label='Left Coverage Rectangle'),
        patches.Patch(color='lightgreen', alpha=0.7, label='Right Coverage Rectangle'),
        patches.Patch(color='red', alpha=0, label='Ellipse')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # ============================================================================
    # 中间图：auxiliary.h 方法（使用边界矩形作为极值点）
    # ============================================================================
    
    # 使用边界矩形作为极值点，而不是计算椭圆方程
    extremes_boundary = {
        'x_extremes': (x_min, x_min + width),
        'y_extremes': (y_min, y_min + height),
        'x_coords_at_y_extremes': (x_min, x_min + width),
        'y_coords_at_x_extremes': (y_min, y_min + height)
    }
    
    # 计算倾斜角和偏心率
    cos_theta = math.cos(math.radians(theta))
    sin_theta = math.sin(math.radians(theta))
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    Sigma = np.array([[a*a, 0], [0, b*b]])
    Sigma_rotated = R @ Sigma @ R.T
    Sigma_inv = np.linalg.inv(Sigma_rotated)
    
    A = Sigma_inv[0, 0]
    B = Sigma_inv[0, 1]
    C = Sigma_inv[1, 1]
    cov2d = (A, B, C)
    theta_rad = compute_tilt_angle(cov2d)
    eccentricity = compute_eccentricity_auxiliary((A, B, C, 0.5))
    
    left_box_boundary, right_box_boundary = construct_dual_boxes(
        extremes_boundary, (center_x, center_y), theta_rad, eccentricity)
    
    # 计算显示边界
    left_rect_boundary = (left_box_boundary[0], left_box_boundary[1], 
                         left_box_boundary[2] - left_box_boundary[0], 
                         left_box_boundary[3] - left_box_boundary[1])
    right_rect_boundary = (right_box_boundary[0], right_box_boundary[1], 
                          right_box_boundary[2] - right_box_boundary[0], 
                          right_box_boundary[3] - right_box_boundary[1])
    
    x_min_display2, x_max_display2, y_min_display2, y_max_display2 = calculate_display_bounds(
        left_rect_boundary, right_rect_boundary, x_min, y_min, width, height)
    
    # Set up adaptive grid and limits
    ax2.set_xlim(x_min_display2, x_max_display2)
    ax2.set_ylim(y_min_display2, y_max_display2)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Draw large rectangle (light blue)
    bounding_rect2 = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=2, edgecolor='blue', 
                                     facecolor='lightblue', alpha=0.5)
    ax2.add_patch(bounding_rect2)
    
    # Draw left rectangle (light yellow)
    left_rect2 = patches.Rectangle((left_box_boundary[0], left_box_boundary[1]), 
                                 left_box_boundary[2] - left_box_boundary[0], 
                                 left_box_boundary[3] - left_box_boundary[1],
                                 linewidth=2, edgecolor='orange', 
                                 facecolor='lightyellow', alpha=0.7)
    ax2.add_patch(left_rect2)
    
    # Draw right rectangle (light green)
    right_rect2 = patches.Rectangle((right_box_boundary[0], right_box_boundary[1]), 
                                  right_box_boundary[2] - right_box_boundary[0], 
                                  right_box_boundary[3] - right_box_boundary[1],
                                  linewidth=2, edgecolor='green', 
                                  facecolor='lightgreen', alpha=0.7)
    ax2.add_patch(right_rect2)
    
    # Draw ellipse
    ellipse2 = Ellipse((center_x, center_y), 2*a, 2*b, angle=theta,
                      linewidth=3, edgecolor='red', facecolor='none')
    ax2.add_patch(ellipse2)
    
    # Mark ellipse center
    ax2.plot(center_x, center_y, 'ro', markersize=8, label='Ellipse Center')
    
    # Add title and labels
    ax2.set_title(f'Auxiliary.h Method (Fixed)\nUsing Boundary Rectangle as Extremes', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Axis', fontsize=10)
    ax2.set_ylabel('Y Axis', fontsize=10)
    
    # Add legend
    ax2.legend(handles=legend_elements, loc='upper right')
    
    # ============================================================================
    # 右侧图：auxiliary.h 原始方法（使用椭圆方程极值点）
    # ============================================================================
    
    # auxiliary.h 原始方法
    opacity = 0.5
    con_o = (A, B, C, opacity)
    disc = B * B - A * C
    threshold = opacity * 255.0
    t = 2.0 * math.log(threshold)
    
    p = (center_x, center_y)
    extremes = compute_extreme_points(con_o, disc, t, p)
    
    left_aux, right_aux = construct_dual_boxes(extremes, p, theta_rad, eccentricity)
    
    # 计算显示边界
    left_rect_aux = (left_aux[0], left_aux[1], left_aux[2] - left_aux[0], left_aux[3] - left_aux[1])
    right_rect_aux = (right_aux[0], right_aux[1], right_aux[2] - right_aux[0], right_aux[3] - right_aux[1])
    
    x_min_display3, x_max_display3, y_min_display3, y_max_display3 = calculate_display_bounds(
        left_rect_aux, right_rect_aux, x_min, y_min, width, height)
    
    # Set up adaptive grid and limits
    ax3.set_xlim(x_min_display3, x_max_display3)
    ax3.set_ylim(y_min_display3, y_max_display3)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Draw large rectangle (light blue)
    bounding_rect3 = patches.Rectangle((x_min, y_min), width, height, 
                                     linewidth=2, edgecolor='blue', 
                                     facecolor='lightblue', alpha=0.5)
    ax3.add_patch(bounding_rect3)
    
    # Draw left rectangle (light yellow)
    left_rect3 = patches.Rectangle((left_aux[0], left_aux[1]), 
                                 left_aux[2] - left_aux[0], left_aux[3] - left_aux[1],
                                 linewidth=2, edgecolor='orange', 
                                 facecolor='lightyellow', alpha=0.7)
    ax3.add_patch(left_rect3)
    
    # Draw right rectangle (light green)
    right_rect3 = patches.Rectangle((right_aux[0], right_aux[1]), 
                                  right_aux[2] - right_aux[0], right_aux[3] - right_aux[1],
                                  linewidth=2, edgecolor='green', 
                                  facecolor='lightgreen', alpha=0.7)
    ax3.add_patch(right_rect3)
    
    # Draw ellipse
    ellipse3 = Ellipse((center_x, center_y), 2*a, 2*b, angle=theta,
                      linewidth=3, edgecolor='red', facecolor='none')
    ax3.add_patch(ellipse3)
    
    # Mark ellipse center
    ax3.plot(center_x, center_y, 'ro', markersize=8, label='Ellipse Center')
    
    # Add title and labels
    ax3.set_title(f'Auxiliary.h Method (Original)\nUsing Ellipse Equation Extremes', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('X Axis', fontsize=10)
    ax3.set_ylabel('Y Axis', fontsize=10)
    
    # Add legend
    ax3.legend(handles=legend_elements, loc='upper right')
    
    # Add parameter information
    info_text1 = f'Original f(e,θ) = {f_function(e, math.radians(theta)):.3f}'
    ax1.text(0.02, 0.98, info_text1, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    info_text2 = f'Fixed f(e,θ) = {compute_stretching_factor(theta_rad, eccentricity):.3f}'
    ax2.text(0.02, 0.98, info_text2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    info_text3 = f'Original f(e,θ) = {compute_stretching_factor(theta_rad, eccentricity):.3f}'
    ax3.text(0.02, 0.98, info_text3, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save image
    plt.tight_layout()
    plt.savefig('dualbox_comparison_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Image saved as 'dualbox_comparison_fixed.png'")
    print(f"Ellipse parameters: Center({center_x:.2f}, {center_y:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.1f}°")
    print(f"Eccentricity: e = {e:.3f}")
    print(f"Original extension coefficient: f(e,θ) = {f_function(e, math.radians(theta)):.3f}")
    print(f"Fixed extension coefficient: f(e,θ) = {compute_stretching_factor(theta_rad, eccentricity):.3f}")

if __name__ == "__main__":
    main() 