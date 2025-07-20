import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import math

def compute_ellipse_parameters(con_o, t):
    """
    计算椭圆参数，模拟CUDA代码中的计算
    con_o: [A, B, C, opacity] - 二次型系数
    t: 阈值参数
    """
    A, B, C = con_o[0], con_o[1], con_o[2]
    disc = B * B - A * C
    
    # 检查椭圆的有效性
    if A <= 0 or C <= 0 or disc >= 0:
        return None, None, None
    
    # 检查数值稳定性
    if abs(disc) < 1e-10:
        return None, None, None
    
    # 计算半轴长度（模拟CUDA代码）
    try:
        x_term = math.sqrt(-(B * B * t) / (disc * A))
        x_term = x_term if B < 0 else -x_term
        
        y_term = math.sqrt(-(B * B * t) / (disc * C))
        y_term = y_term if B < 0 else -y_term
        
        # 检查数值有效性
        if math.isnan(x_term) or math.isnan(y_term) or math.isinf(x_term) or math.isinf(y_term):
            return None, None, None
            
        return x_term, y_term, disc
    except (ValueError, ZeroDivisionError):
        return None, None, None

def compute_ellipse_intersection(con_o, disc, t, center, is_y, coord):
    """
    计算椭圆与坐标线的交点（模拟CUDA代码）
    """
    A, B, C = con_o[0], con_o[1], con_o[2]
    
    if is_y:
        p_u, p_v = center[1], center[0]  # y, x
        coeff = A
    else:
        p_u, p_v = center[0], center[1]  # x, y
        coeff = C
    
    h = coord - p_u
    sqrt_term = math.sqrt(disc * h * h + t * coeff)
    
    v1 = (-B * h - sqrt_term) / coeff + p_v
    v2 = (-B * h + sqrt_term) / coeff + p_v
    
    return v1, v2

def compute_tilt_angle(cov2d):
    """
    计算倾斜角度
    """
    numerator = 2.0 * cov2d[1]
    denominator = cov2d[0] - cov2d[2]
    
    # 处理分母为零的情况
    if abs(denominator) < 1e-10:
        return 0.0
    
    angle = 0.5 * math.atan2(numerator, denominator)
    
    if angle < 0:
        angle += math.pi
    
    return angle

def compute_eccentricity(con_o):
    """
    计算椭圆离心率
    """
    A, B, C = con_o[0], con_o[1], con_o[2]
    
    diff_AC = A - C
    term_under_sqrt = diff_AC * diff_AC + 4.0 * B * B
    term_sqrt = math.sqrt(term_under_sqrt)
    
    sum_AC = A + C
    lambda_max = (sum_AC + term_sqrt) / 2.0
    lambda_min = (sum_AC - term_sqrt) / 2.0
    
    # 处理数值稳定性
    if lambda_max < 1e-10:
        return 1.0
    
    ratio = lambda_min / lambda_max
    ratio = max(0.0, min(1.0, ratio))  # 限制在[0,1]范围内
    
    return math.sqrt(1.0 - ratio)

def visualize_ellipse_boundaries(con_o, center, t, title="Ellipse Boundary Visualization"):
    """
    可视化椭圆边界计算
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 计算椭圆参数
    x_term, y_term, disc = compute_ellipse_parameters(con_o, t)
    if x_term is None:
        print("Invalid ellipse parameters")
        return None
    
    # 计算倾斜角度和离心率
    cov2d = [con_o[0], con_o[1], con_o[2]]
    theta = compute_tilt_angle(cov2d)
    eccentricity = compute_eccentricity(con_o)
    
    print(f"Ellipse parameters:")
    print(f"  Center: {center}")
    print(f"  Semi-axes: x_term={x_term:.4f}, y_term={y_term:.4f}")
    print(f"  Tilt angle: {math.degrees(theta):.2f}°")
    print(f"  Eccentricity: {eccentricity:.4f}")
    
    # 1. 绘制椭圆
    # 计算椭圆的主轴长度和角度
    A, B, C = con_o[0], con_o[1], con_o[2]
    
    # 计算特征值和特征向量
    trace = A + C
    det = A * C - B * B
    
    # 主轴长度
    try:
        major_axis = math.sqrt(-t / (det / trace - math.sqrt((trace/2)**2 - det)))
        minor_axis = math.sqrt(-t / (det / trace + math.sqrt((trace/2)**2 - det)))
    except (ValueError, ZeroDivisionError):
        # 如果计算失败，使用简化的近似
        major_axis = abs(x_term)
        minor_axis = abs(y_term)
    
    # 绘制椭圆
    ellipse = Ellipse(center, 2*major_axis, 2*minor_axis, 
                     angle=math.degrees(theta), 
                     fill=False, color='blue', linewidth=2, label='Ellipse')
    ax.add_patch(ellipse)
    
    # 2. 绘制轴对齐边界框 (AABB)
    bbox_argmin = [center[1] - y_term, center[0] - x_term]
    bbox_argmax = [center[1] + y_term, center[0] + x_term]
    
    aabb_width = bbox_argmax[0] - bbox_argmin[0]
    aabb_height = bbox_argmax[1] - bbox_argmin[1]
    
    aabb = patches.Rectangle(bbox_argmin, aabb_width, aabb_height,
                           fill=False, color='red', linewidth=2, 
                           linestyle='--', label='Axis-Aligned Bounding Box (AABB)')
    ax.add_patch(aabb)
    
    # 3. 计算并绘制精确椭圆边界
    try:
        # 计算椭圆与边界框边的交点
        bbox_min_x = compute_ellipse_intersection(con_o, disc, t, center, False, bbox_argmin[0])
        bbox_max_x = compute_ellipse_intersection(con_o, disc, t, center, False, bbox_argmax[0])
        bbox_min_y = compute_ellipse_intersection(con_o, disc, t, center, True, bbox_argmin[1])
        bbox_max_y = compute_ellipse_intersection(con_o, disc, t, center, True, bbox_argmax[1])
        
        # 找到精确边界
        exact_min_x = min(bbox_min_x[0], bbox_min_x[1])
        exact_max_x = max(bbox_max_x[0], bbox_max_x[1])
        exact_min_y = min(bbox_min_y[0], bbox_min_y[1])
        exact_max_y = max(bbox_max_y[0], bbox_max_y[1])
        
        exact_width = exact_max_x - exact_min_x
        exact_height = exact_max_y - exact_min_y
        
        exact_bbox = patches.Rectangle([exact_min_x, exact_min_y], exact_width, exact_height,
                                     fill=False, color='green', linewidth=2,
                                     linestyle=':', label='Exact Ellipse Boundary')
        ax.add_patch(exact_bbox)
        
        print(f"Bounding box area comparison:")
        print(f"  AABB area: {aabb_width * aabb_height:.4f}")
        print(f"  Exact boundary area: {exact_width * exact_height:.4f}")
        print(f"  Area ratio (AABB/Exact): {(aabb_width * aabb_height) / (exact_width * exact_height):.2f}")
        
    except Exception as e:
        print(f"Exact boundary calculation failed: {e}")
    
    # 4. 绘制中心点和坐标轴
    ax.plot(center[0], center[1], 'ko', markersize=8, label='Ellipse Center')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # 5. 绘制主轴方向
    try:
        major_end_x = center[0] + major_axis * math.cos(theta)
        major_end_y = center[1] + major_axis * math.sin(theta)
        minor_end_x = center[0] + minor_axis * math.cos(theta + math.pi/2)
        minor_end_y = center[1] + minor_axis * math.sin(theta + math.pi/2)
        
        ax.plot([center[0], major_end_x], [center[1], major_end_y], 
               'g-', linewidth=3, alpha=0.7, label='Major Axis')
        ax.plot([center[0], minor_end_x], [center[1], minor_end_y], 
               'g-', linewidth=3, alpha=0.7, label='Minor Axis')
    except:
        pass
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 设置坐标轴范围
    margin = max(abs(x_term), abs(y_term)) * 0.2
    if margin < 0.1:
        margin = 0.5  # 最小边距
    
    ax.set_xlim(center[0] - abs(x_term) - margin, center[0] + abs(x_term) + margin)
    ax.set_ylim(center[1] - abs(y_term) - margin, center[1] + abs(y_term) + margin)
    
    plt.tight_layout()
    return fig

def compare_different_ellipses():
    """
    比较不同椭圆的边界计算
    """
    # 测试用例1: 圆形
    print("=== Test Case 1: Circle ===")
    con_o1 = [1.0, 0.0, 1.0, 0.5]  # A=1, B=0, C=1 (circle)
    center1 = [0.0, 0.0]
    t1 = -2.0
    
    fig1 = visualize_ellipse_boundaries(con_o1, center1, t1, "Circle Boundary")
    if fig1:
        plt.savefig('ellipse_circle.png', dpi=300, bbox_inches='tight')
    
    # 测试用例2: 倾斜椭圆
    print("\n=== Test Case 2: Tilted Ellipse ===")
    con_o2 = [2.0, 0.5, 1.0, 0.5]  # A=2, B=0.5, C=1 (tilted ellipse)
    center2 = [0.0, 0.0]
    t2 = -2.0
    
    fig2 = visualize_ellipse_boundaries(con_o2, center2, t2, "Tilted Ellipse Boundary")
    if fig2:
        plt.savefig('ellipse_tilted.png', dpi=300, bbox_inches='tight')
    
    # 测试用例3: 高离心率椭圆
    print("\n=== Test Case 3: High Eccentricity Ellipse ===")
    con_o3 = [4.0, 0.0, 1.0, 0.5]  # A=4, B=0, C=1 (flat ellipse)
    center3 = [0.0, 0.0]
    t3 = -2.0
    
    fig3 = visualize_ellipse_boundaries(con_o3, center3, t3, "High Eccentricity Ellipse Boundary")
    if fig3:
        plt.savefig('ellipse_eccentric.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def interactive_ellipse_explorer():
    """
    交互式椭圆探索器
    """
    from matplotlib.widgets import Slider, Button
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.3)
    
    # 初始参数
    A_init, B_init, C_init = 2.0, 0.5, 1.0
    t_init = -2.0
    center = [0.0, 0.0]
    
    # 创建滑块
    ax_A = plt.axes([0.1, 0.2, 0.65, 0.03])
    ax_B = plt.axes([0.1, 0.15, 0.65, 0.03])
    ax_C = plt.axes([0.1, 0.1, 0.65, 0.03])
    ax_t = plt.axes([0.1, 0.05, 0.65, 0.03])
    
    s_A = Slider(ax_A, 'A', 0.1, 5.0, valinit=A_init)
    s_B = Slider(ax_B, 'B', -2.0, 2.0, valinit=B_init)
    s_C = Slider(ax_C, 'C', 0.1, 5.0, valinit=C_init)
    s_t = Slider(ax_t, 't', -5.0, -0.1, valinit=t_init)
    
    def update(val):
        ax.clear()
        
        A, B, C = s_A.val, s_B.val, s_C.val
        t = s_t.val
        
        con_o = [A, B, C, 0.5]
        
        try:
            # 计算椭圆参数
            x_term, y_term, disc = compute_ellipse_parameters(con_o, t)
            if x_term is not None:
                # 绘制椭圆
                cov2d = [A, B, C]
                theta = compute_tilt_angle(cov2d)
                
                # 计算主轴长度
                trace = A + C
                det = A * C - B * B
                try:
                    major_axis = math.sqrt(-t / (det / trace - math.sqrt((trace/2)**2 - det)))
                    minor_axis = math.sqrt(-t / (det / trace + math.sqrt((trace/2)**2 - det)))
                except:
                    major_axis = abs(x_term)
                    minor_axis = abs(y_term)
                
                ellipse = Ellipse(center, 2*major_axis, 2*minor_axis, 
                                angle=math.degrees(theta), 
                                fill=False, color='blue', linewidth=2)
                ax.add_patch(ellipse)
                
                # 绘制AABB
                bbox_argmin = [center[1] - y_term, center[0] - x_term]
                bbox_argmax = [center[1] + y_term, center[0] + x_term]
                aabb_width = bbox_argmax[0] - bbox_argmin[0]
                aabb_height = bbox_argmax[1] - bbox_argmin[1]
                
                aabb = patches.Rectangle(bbox_argmin, aabb_width, aabb_height,
                                       fill=False, color='red', linewidth=2, 
                                       linestyle='--')
                ax.add_patch(aabb)
                
                # 设置坐标轴范围
                margin = max(abs(x_term), abs(y_term)) * 0.2
                if margin < 0.1:
                    margin = 0.5
                    
                ax.set_xlim(center[0] - abs(x_term) - margin, center[0] + abs(x_term) + margin)
                ax.set_ylim(center[1] - abs(y_term) - margin, center[1] + abs(y_term) + margin)
                
                ax.set_title(f'Ellipse Boundary (A={A:.2f}, B={B:.2f}, C={C:.2f}, t={t:.2f})')
            else:
                ax.set_title('Invalid ellipse parameters')
            
        except Exception as e:
            ax.set_title(f'Error: {e}')
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        fig.canvas.draw_idle()
    
    s_A.on_changed(update)
    s_B.on_changed(update)
    s_C.on_changed(update)
    s_t.on_changed(update)
    
    # 初始绘制
    update(None)
    
    plt.show()

if __name__ == "__main__":
    print("Ellipse Boundary Visualization Tool")
    print("1. Compare different ellipses")
    print("2. Interactive explorer")
    
    choice = input("Please choose (1/2): ").strip()
    
    if choice == "1":
        compare_different_ellipses()
    elif choice == "2":
        interactive_ellipse_explorer()
    else:
        print("Invalid choice, running default comparison...")
        compare_different_ellipses()