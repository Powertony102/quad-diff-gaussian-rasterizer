import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import math

def compute_tilt_angle(cov2d):
    """Calculate tilt angle using CUDA formula"""
    numerator = 2.0 * cov2d[1]  # 2 * sigma_xy
    denominator = cov2d[0] - cov2d[2]  # sigma_xx - sigma_yy
    
    angle = 0.5 * np.arctan2(numerator, denominator)
    if angle < 0.0:
        angle += np.pi
    
    return angle

def compute_eccentricity(con_o):
    """Calculate eccentricity using CUDA formula"""
    A, B, C = con_o[0], con_o[1], con_o[2]
    
    diff_AC = A - C
    term_under_sqrt = diff_AC * diff_AC + 4.0 * B * B
    term_sqrt = np.sqrt(term_under_sqrt)
    
    sum_AC = A + C
    lambda_max = (sum_AC + term_sqrt) / 2.0
    lambda_min = (sum_AC - term_sqrt) / 2.0
    
    if lambda_max <= 0:
        return 0
        
    ratio = lambda_min / lambda_max
    if ratio < 0:
        ratio = 0
        
    return np.sqrt(1.0 - ratio)

def construct_snugbox(con_o, disc, t, center):
    """Construct snugbox following CUDA logic exactly"""
    # CUDA code: x_term = sqrt(-(con_o.y^2 * t) / (disc * con_o.x))
    x_term = np.sqrt(-(con_o[1]**2 * t) / (disc * con_o[0]))
    if con_o[1] < 0:
        x_term = -x_term
        
    # CUDA code: y_term = sqrt(-(con_o.y^2 * t) / (disc * con_o.z))
    y_term = np.sqrt(-(con_o[1]**2 * t) / (disc * con_o[2]))
    if con_o[1] < 0:
        y_term = -y_term
    
    # CUDA coordinate mapping - Note the coordinate swap!
    # bbox_argmin = { center.y - y_term, center.x - x_term }
    # bbox_argmax = { center.y + y_term, center.x + x_term }
    snug_min_x = center[1] - y_term  # center.y - y_term
    snug_max_x = center[1] + y_term  # center.y + y_term
    snug_min_y = center[0] - x_term  # center.x - x_term
    snug_max_y = center[0] + x_term  # center.x + x_term
    
    return [snug_min_x, snug_min_y, snug_max_x, snug_max_y]

def construct_quadboxes(con_o, disc, t, center, theta, eccentricity):
    """Construct quadboxes following exact CUDA logic"""
    snug_bbox = construct_snugbox(con_o, disc, t, center)
    snug_min_x, snug_min_y, snug_max_x, snug_max_y = snug_bbox
    
    # Calculate stretch factor
    e_sq = eccentricity * eccentricity
    if e_sq >= 1.0:
        e_sq = 0.999
        
    sin_2theta = np.sin(2.0 * theta)
    sin_2theta_sq = sin_2theta * sin_2theta
    stretch_factor = 1.0 / np.sqrt(1.0 + (e_sq * e_sq / (4.0 * (1.0 - e_sq))) * sin_2theta_sq)
    
    if theta >= 0 and theta <= np.pi/2:  # 0° to 90°
        # Left rectangle
        left_rect_x = snug_min_x
        left_rect_y = snug_min_y
        left_rect_width = center[0] - snug_min_x
        left_rect_height = center[1] - snug_min_y
        
        # Right rectangle
        right_rect_x = center[0]
        right_rect_y = center[1]
        right_rect_width = snug_max_x - center[0]
        right_rect_height = snug_max_y - center[1]
        
        # Small rectangles
        left_small_width = left_rect_width * stretch_factor
        left_small_height = left_rect_height * stretch_factor
        left_small_rect_x = center[0] - left_small_width
        left_small_rect_y = center[1]
        
        right_small_width = left_small_width
        right_small_height = left_small_height
        right_small_rect_x = center[0]
        right_small_rect_y = center[1] - right_small_height
        
    else:  # θ > 90°
        # Left rectangle
        left_rect_x = snug_min_x
        left_rect_y = center[1]
        left_rect_width = center[0] - snug_min_x
        left_rect_height = snug_max_y - center[1]
        
        # Right rectangle
        right_rect_x = center[0]
        right_rect_y = snug_min_y
        right_rect_width = snug_max_x - center[0]
        right_rect_height = center[1] - snug_min_y
        
        # Small rectangles
        left_small_width = left_rect_width * stretch_factor
        left_small_height = left_rect_height * stretch_factor
        left_small_rect_x = center[0] - left_small_width
        left_small_rect_y = center[1] - left_small_height
        
        right_small_width = left_small_width
        right_small_height = left_small_height
        right_small_rect_x = center[0]
        right_small_rect_y = center[1]
    
    # Return boxes as [min_x, min_y, max_x, max_y]
    left_box = [left_rect_x, left_rect_y, 
                left_rect_x + left_rect_width, 
                left_rect_y + left_rect_height]
    
    right_box = [right_rect_x, right_rect_y, 
                 right_rect_x + right_rect_width, 
                 right_rect_y + right_rect_height]
    
    left_small_box = [left_small_rect_x, left_small_rect_y,
                      left_small_rect_x + left_small_width,
                      left_small_rect_y + left_small_height]
    
    right_small_box = [right_small_rect_x, right_small_rect_y,
                       right_small_rect_x + right_small_width,
                       right_small_rect_y + right_small_height]
    
    return left_box, right_box, left_small_box, right_small_box

def create_visible_ellipse_from_covariance(center, sigma_xx, sigma_xy, sigma_yy, scale=3.0):
    """Create a clearly visible ellipse from covariance matrix parameters"""
    # Build covariance matrix
    cov_matrix = np.array([[sigma_xx, sigma_xy], 
                          [sigma_xy, sigma_yy]])
    
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # Calculate ellipse parameters (using scale for visibility)
    width = 2 * np.sqrt(eigenvals[0] * scale)
    height = 2 * np.sqrt(eigenvals[1] * scale)
    
    # Calculate rotation angle in degrees
    angle = np.degrees(np.arctan2(eigenvecs[0, 1], eigenvecs[0, 0]))
    
    return Ellipse(center, width, height, angle=angle)

def visualize_gaussian_shapes_fixed():
    """Create visualization with guaranteed visible ellipses"""
    
    # Test cases with properly scaled parameters
    test_cases = [
        {
            'name': 'Tilted Ellipse (45°)',
            'center': [5, 5],
            'sigma': [1.0, 0.7, 1.5],  # sigma_xx, sigma_xy, sigma_yy
            'con_o': [0.8, 0.5, 1.2, 0.8],  # Corresponding con_o parameters
        },
        {
            'name': 'Horizontal Ellipse',
            'center': [5, 5],
            'sigma': [2.0, 0.1, 0.8],
            'con_o': [0.4, 0.05, 1.0, 0.7],
        },
        {
            'name': 'Vertical Ellipse',
            'center': [5, 5],
            'sigma': [0.8, 0.1, 2.0],
            'con_o': [1.0, 0.05, 0.4, 0.6],
        },
        {
            'name': 'High Eccentricity',
            'center': [5, 5],
            'sigma': [3.0, 0.8, 0.5],
            'con_o': [0.2, 0.3, 1.5, 0.5],
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, case in enumerate(test_cases):
        ax = axes[i]
        center = case['center']
        sigma = case['sigma']
        con_o = case['con_o']
        
        # Create a clearly visible ellipse
        ellipse = create_visible_ellipse_from_covariance(
            center, sigma[0], sigma[1], sigma[2], scale=2.0
        )
        ellipse.set_facecolor('lightblue')
        ellipse.set_edgecolor('blue')
        ellipse.set_alpha(0.6)
        ellipse.set_linewidth(3)
        ax.add_patch(ellipse)
        
        # Calculate parameters
        theta = compute_tilt_angle(sigma)
        eccentricity = compute_eccentricity(con_o)
        
        # Check if parameters are valid
        disc = con_o[1]**2 - con_o[0] * con_o[2]
        if con_o[0] <= 0 or con_o[2] <= 0 or disc >= 0:
            print(f"Warning: Invalid parameters for {case['name']}")
            continue
            
        t = 2.0 * np.log(con_o[3] * 255.0)
        
        # Draw snugbox
        try:
            snug_bbox = construct_snugbox(con_o, disc, t, center)
            snug_rect = patches.Rectangle(
                (snug_bbox[0], snug_bbox[1]), 
                snug_bbox[2] - snug_bbox[0], 
                snug_bbox[3] - snug_bbox[1],
                linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(snug_rect)
        except Exception as e:
            print(f"Error drawing snugbox for {case['name']}: {e}")
        
        # Draw quadboxes
        try:
            left_box, right_box, left_small_box, right_small_box = construct_quadboxes(
                con_o, disc, t, center, theta, eccentricity
            )
            
            boxes = [left_box, right_box, left_small_box, right_small_box]
            colors = ['green', 'orange', 'purple', 'brown']
            labels = ['Left Large', 'Right Large', 'Left Small', 'Right Small']
            alphas = [0.3, 0.3, 0.5, 0.5]
            
            for box, color, label, alpha in zip(boxes, colors, labels, alphas):
                if all(np.isfinite(box)):  # Check for valid coordinates
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=alpha,
                        label=label
                    )
                    ax.add_patch(rect)
        except Exception as e:
            print(f"Error drawing quadboxes for {case['name']}: {e}")
        
        # Mark center
        ax.plot(center[0], center[1], 'ko', markersize=10, label='Center', zorder=10)
        
        # Set plot properties
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Add parameter info
        info_text = f'{case["name"]}\n'
        info_text += f'θ = {np.degrees(theta):.1f}°\n'
        info_text += f'Eccentricity = {eccentricity:.3f}\n'
        info_text += f'Discriminant = {disc:.3f}'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(case['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
    
    plt.tight_layout()
    plt.suptitle('Gaussian Ellipse Visualization: Ellipse, Snugbox and Quadboxes', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.show()

def create_simple_demo():
    """Create a simple demonstration with guaranteed visible components"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Simple, guaranteed visible case
    center = [5, 5]
    
    # Create a clear ellipse manually
    ellipse = Ellipse(center, 4, 2.5, angle=30, 
                     facecolor='lightblue', edgecolor='blue', 
                     alpha=0.7, linewidth=4)
    ax.add_patch(ellipse)
    
    # Corresponding parameters for this ellipse
    sigma = [1.2, 0.6, 0.8]  # sigma_xx, sigma_xy, sigma_yy
    con_o = [0.7, 0.4, 1.0, 0.6]  # A, B, C, opacity
    
    theta = compute_tilt_angle(sigma)
    eccentricity = compute_eccentricity(con_o)
    disc = con_o[1]**2 - con_o[0] * con_o[2]
    t = 2.0 * np.log(con_o[3] * 255.0)
    
    print(f"Parameters: theta={np.degrees(theta):.1f}°, ecc={eccentricity:.3f}, disc={disc:.3f}, t={t:.3f}")
    
    # Draw snugbox if valid
    if disc < 0:
        snug_bbox = construct_snugbox(con_o, disc, t, center)
        snug_rect = patches.Rectangle(
            (snug_bbox[0], snug_bbox[1]), 
            snug_bbox[2] - snug_bbox[0], 
            snug_bbox[3] - snug_bbox[1],
            linewidth=4, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(snug_rect)
        
        # Draw quadboxes
        left_box, right_box, left_small_box, right_small_box = construct_quadboxes(
            con_o, disc, t, center, theta, eccentricity
        )
        
        boxes = [left_box, right_box, left_small_box, right_small_box]
        colors = ['green', 'orange', 'purple', 'brown']
        labels = ['Left Large', 'Right Large', 'Left Small', 'Right Small']
        alphas = [0.4, 0.4, 0.6, 0.6]
        
        for box, color, label, alpha in zip(boxes, colors, labels, alphas):
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=3,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                label=label
            )
            ax.add_patch(rect)
    
    # Mark center
    ax.plot(center[0], center[1], 'ko', markersize=12, label='Center', zorder=10)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_title('Simple Gaussian Ellipse Demo with Snugbox and Quadboxes', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Creating simple demo with guaranteed visible ellipse...")
    create_simple_demo()
    
    print("Creating detailed visualization...")
    visualize_gaussian_shapes_fixed()
    
    print("Visualization complete!")
