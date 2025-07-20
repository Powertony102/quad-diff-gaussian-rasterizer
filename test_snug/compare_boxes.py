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
    """Calculate the bounding rectangle of the ellipse (SnugBox)"""
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

def create_dual_boxes(center_x, center_y, x_min, y_min, width, height, theta, e):
    """Create DualBox Left & Right boxes"""
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

def compare_boxes():
    """Compare SnugBox vs DualBox Left & Right boxes"""
    print("=" * 80)
    print("SnugBox vs DualBox Left & Right Boxes Comparison")
    print("=" * 80)
    
    # Generate test cases
    test_cases = []
    for i in range(3):
        center_x, center_y, a, b, theta = generate_ellipse_in_first_quadrant(30)
        test_cases.append((center_x, center_y, a, b, theta))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SnugBox vs DualBox Left & Right Boxes', fontsize=16, fontweight='bold')
    
    for i, (center_x, center_y, a, b, theta) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"  Ellipse: center({center_x:.2f}, {center_y:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.1f}°")
        
        # Calculate eccentricity
        e = calculate_eccentricity(a, b)
        print(f"  Eccentricity: e = {e:.3f}")
        
        # Calculate SnugBox (bounding rectangle)
        x_min, y_min, width, height = calculate_bounding_rectangle(center_x, center_y, a, b, theta)
        print(f"  SnugBox: x_min={x_min:.2f}, y_min={y_min:.2f}, width={width:.2f}, height={height:.2f}")
        
        # Calculate DualBox Left & Right boxes
        left_box, right_box = create_dual_boxes(center_x, center_y, x_min, y_min, width, height, theta, e)
        print(f"  Left Box: x={left_box[0]:.2f}, y={left_box[1]:.2f}, w={left_box[2]:.2f}, h={left_box[3]:.2f}")
        print(f"  Right Box: x={right_box[0]:.2f}, y={right_box[1]:.2f}, w={right_box[2]:.2f}, h={right_box[3]:.2f}")
        
        # Calculate areas
        snugbox_area = width * height
        left_box_area = left_box[2] * left_box[3]
        right_box_area = right_box[2] * right_box[3]
        dualbox_total_area = left_box_area + right_box_area
        
        print(f"  SnugBox area: {snugbox_area:.2f}")
        print(f"  Left Box area: {left_box_area:.2f}")
        print(f"  Right Box area: {right_box_area:.2f}")
        print(f"  DualBox total area: {dualbox_total_area:.2f}")
        print(f"  Area ratio (DualBox/SnugBox): {dualbox_total_area/snugbox_area:.3f}")
        
        # Visualize current test case
        ax = axes[i]
        
        # Draw ellipse
        ellipse = Ellipse((center_x, center_y), 2*a, 2*b, angle=theta, 
                         facecolor='lightblue', edgecolor='blue', alpha=0.6, linewidth=2)
        ax.add_patch(ellipse)
        
        # Draw SnugBox (bounding rectangle)
        snugbox_rect = patches.Rectangle((x_min, y_min), width, height, 
                                       linewidth=3, edgecolor='green', facecolor='none', linestyle='-')
        ax.add_patch(snugbox_rect)
        
        # Draw Left Box
        left_rect = patches.Rectangle((left_box[0], left_box[1]), left_box[2], left_box[3], 
                                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.4)
        ax.add_patch(left_rect)
        
        # Draw Right Box
        right_rect = patches.Rectangle((right_box[0], right_box[1]), right_box[2], right_box[3], 
                                     linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.4)
        ax.add_patch(right_rect)
        
        # Draw center point
        ax.plot(center_x, center_y, 'ko', markersize=8, label='Ellipse Center')
        
        # Set chart properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Test Case {i+1}\nCenter({center_x:.1f}, {center_y:.1f}), a={a:.1f}, b={b:.1f}, θ={theta:.1f}°')
        ax.grid(True, alpha=0.3)
        ax.legend(['Ellipse Center', 'Ellipse', 'SnugBox', 'Left Box', 'Right Box'], 
                 loc='upper right', fontsize=8)
        
        # Set axis limits with padding
        all_x = [x_min, x_min + width, left_box[0], left_box[0] + left_box[2], 
                right_box[0], right_box[0] + right_box[2]]
        all_y = [y_min, y_min + height, left_box[1], left_box[1] + left_box[3], 
                right_box[1], right_box[1] + right_box[3]]
        
        x_min_plot = min(all_x) - 1
        x_max_plot = max(all_x) + 1
        y_min_plot = min(all_y) - 1
        y_max_plot = max(all_y) + 1
        
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('snugbox_vs_dualbox.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 80)
    print("Visualization saved as 'snugbox_vs_dualbox.png'")
    print("=" * 80)
    print("Legend:")
    print("  Blue ellipse: Original ellipse")
    print("  Green rectangle: SnugBox (bounding rectangle)")
    print("  Red rectangle: DualBox Left Box")
    print("  Orange rectangle: DualBox Right Box")
    print("=" * 80)

if __name__ == "__main__":
    compare_boxes() 