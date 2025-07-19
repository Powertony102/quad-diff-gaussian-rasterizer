import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import random
import math

def generate_ellipse_in_first_quadrant(grid_size=20):
    """Generate an ellipse within the first quadrant"""
    while True:
        # Randomly generate ellipse parameters with larger sizes
        center_x = random.uniform(5, grid_size - 5)
        center_y = random.uniform(5, grid_size - 5)
        a = random.uniform(2, 5)  # Increased semi-major axis
        b = random.uniform(1, a)  # Increased semi-minor axis
        theta = random.uniform(0, 180)  # Inclination angle (degrees)
        
        # Convert to radians
        theta_rad = math.radians(theta)
        
        # Calculate ellipse boundary points to check if it's in the first quadrant
        # Parametric equation extremal points
        t_values = np.linspace(0, 2*np.pi, 1000)
        x_points = center_x + a * np.cos(t_values) * np.cos(theta_rad) - b * np.sin(t_values) * np.sin(theta_rad)
        y_points = center_y + a * np.cos(t_values) * np.sin(theta_rad) + b * np.sin(t_values) * np.cos(theta_rad)
        
        # Check if ellipse is completely in the first quadrant
        if np.min(x_points) > 0 and np.min(y_points) > 0:
            return center_x, center_y, a, b, theta

def calculate_bounding_rectangle(center_x, center_y, a, b, theta):
    """Calculate the bounding rectangle of the ellipse"""
    theta_rad = math.radians(theta)
    
    # Calculate ellipse boundaries after rotation
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # Ellipse boundary calculation
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

def create_coverage_rectangles(center_x, center_y, x_min, y_min, width, height, theta, e):
    """Create coverage rectangles"""
    theta_rad = math.radians(theta)
    
    # Four corner points of the large rectangle
    x_max = x_min + width
    y_max = y_min + height
    
    if 0 <= theta <= 90:
        # Left rectangle: bottom-left point is bottom-left of large rectangle, top-right point is ellipse center
        left_rect_x = x_min
        left_rect_y = y_min
        left_rect_width = center_x - x_min
        left_rect_height = center_y - y_min
        
        # Right rectangle: bottom-left point is ellipse center, top-right point is top-right of large rectangle
        right_rect_x = center_x
        right_rect_y = center_y
        right_rect_width = x_max - center_x
        right_rect_height = y_max - center_y
        
    else:  # theta > 90
        # Left rectangle: top-left point is top-left of large rectangle, bottom-right point is ellipse center
        left_rect_x = x_min
        left_rect_y = center_y
        left_rect_width = center_x - x_min
        left_rect_height = y_max - center_y
        
        # Right rectangle: top-left point is ellipse center, bottom-right point is bottom-right of large rectangle
        right_rect_x = center_x
        right_rect_y = y_min
        right_rect_width = x_max - center_x
        right_rect_height = center_y - y_min
    
    # Calculate extension coefficient
    f_val = f_function(e, theta_rad)
    
    # Extend rectangles
    # Left rectangle extends to the right
    left_extension = left_rect_width * f_val
    left_rect_width += left_extension
    
    # Right rectangle extends to the left
    right_extension = right_rect_width * f_val
    right_rect_x -= right_extension
    right_rect_width += right_extension
    
    return (left_rect_x, left_rect_y, left_rect_width, left_rect_height), \
           (right_rect_x, right_rect_y, right_rect_width, right_rect_height)

def calculate_display_bounds(left_rect_params, right_rect_params, x_min, y_min, width, height):
    """Calculate the optimal display bounds for the figure"""
    # Get all rectangle bounds
    left_x_min = left_rect_params[0]
    left_x_max = left_rect_params[0] + left_rect_params[2]
    left_y_min = left_rect_params[1]
    left_y_max = left_rect_params[1] + left_rect_params[3]
    
    right_x_min = right_rect_params[0]
    right_x_max = right_rect_params[0] + right_rect_params[2]
    right_y_min = right_rect_params[1]
    right_y_max = right_rect_params[1] + right_rect_params[3]
    
    # Bounding rectangle bounds
    bound_x_min = x_min
    bound_x_max = x_min + width
    bound_y_min = y_min
    bound_y_max = y_min + height
    
    # Find overall bounds
    overall_x_min = min(left_x_min, right_x_min, bound_x_min)
    overall_x_max = max(left_x_max, right_x_max, bound_x_max)
    overall_y_min = min(left_y_min, right_y_min, bound_y_min)
    overall_y_max = max(left_y_max, right_y_max, bound_y_max)
    
    # Add some padding
    padding_x = (overall_x_max - overall_x_min) * 0.1
    padding_y = (overall_y_max - overall_y_min) * 0.1
    
    return (overall_x_min - padding_x, overall_x_max + padding_x,
            overall_y_min - padding_y, overall_y_max + padding_y)

def main():
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Generate ellipse with larger size
    center_x, center_y, a, b, theta = generate_ellipse_in_first_quadrant(30)  # Increased grid size
    
    # Calculate eccentricity
    e = calculate_eccentricity(a, b)
    
    # Calculate bounding rectangle
    x_min, y_min, width, height = calculate_bounding_rectangle(center_x, center_y, a, b, theta)
    
    # Create coverage rectangles
    left_rect_params, right_rect_params = create_coverage_rectangles(
        center_x, center_y, x_min, y_min, width, height, theta, e)
    
    # Calculate optimal display bounds
    x_min_display, x_max_display, y_min_display, y_max_display = calculate_display_bounds(
        left_rect_params, right_rect_params, x_min, y_min, width, height)
    
    # Set up adaptive grid and limits
    ax.set_xlim(x_min_display, x_max_display)
    ax.set_ylim(y_min_display, y_max_display)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Draw large rectangle (light blue)
    bounding_rect = patches.Rectangle((x_min, y_min), width, height, 
                                    linewidth=2, edgecolor='blue', 
                                    facecolor='lightblue', alpha=0.5)
    ax.add_patch(bounding_rect)
    
    # Draw left rectangle (light yellow)
    left_rect = patches.Rectangle((left_rect_params[0], left_rect_params[1]), 
                                left_rect_params[2], left_rect_params[3],
                                linewidth=2, edgecolor='orange', 
                                facecolor='lightyellow', alpha=0.7)
    ax.add_patch(left_rect)
    
    # Draw right rectangle (light green)
    right_rect = patches.Rectangle((right_rect_params[0], right_rect_params[1]), 
                                 right_rect_params[2], right_rect_params[3],
                                 linewidth=2, edgecolor='green', 
                                 facecolor='lightgreen', alpha=0.7)
    ax.add_patch(right_rect)
    
    # Draw ellipse
    ellipse = Ellipse((center_x, center_y), 2*a, 2*b, angle=theta,
                     linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(ellipse)
    
    # Mark ellipse center
    ax.plot(center_x, center_y, 'ro', markersize=8, label='Ellipse Center')
    
    # Add title and labels
    plt.title(f'Ellipse Coverage Diagram\nEllipse Parameters: a={a:.2f}, b={b:.2f}, θ={theta:.1f}°, e={e:.3f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('X Axis', fontsize=12)
    plt.ylabel('Y Axis', fontsize=12)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', alpha=0.5, label='Bounding Rectangle'),
        patches.Patch(color='lightyellow', alpha=0.7, label='Left Coverage Rectangle'),
        patches.Patch(color='lightgreen', alpha=0.7, label='Right Coverage Rectangle'),
        patches.Patch(color='red', alpha=0, label='Ellipse')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add parameter information
    info_text = f'Eccentricity e = {e:.3f}\nInclination θ = {theta:.1f}°\nf(e,θ) = {f_function(e, math.radians(theta)):.3f}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save image
    plt.tight_layout()
    plt.savefig('ellipse_coverage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Image saved as 'ellipse_coverage.png'")
    print(f"Ellipse parameters: Center({center_x:.2f}, {center_y:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.1f}°")
    print(f"Eccentricity: e = {e:.3f}")
    print(f"Extension coefficient: f(e,θ) = {f_function(e, math.radians(theta)):.3f}")

if __name__ == "__main__":
    main()