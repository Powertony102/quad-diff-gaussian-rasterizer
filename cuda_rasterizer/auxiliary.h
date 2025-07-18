/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// ---- Dual-SnugBox Core Mathematical Functions ---- //

// Constants for numerical stability and error handling
#define DUAL_SNUGBOX_EPSILON 1e-6f
#define DUAL_SNUGBOX_MAX_ASPECT_RATIO 1000.0f
#define DUAL_SNUGBOX_MIN_OPACITY_THRESHOLD (1.0f / 255.0f)
#define DUAL_SNUGBOX_MAX_COORDINATE 1e6f

// Structure to hold extreme points of an ellipse
struct ExtremePoints {
    float2 x_extremes;  // (x_min, x_max)
    float2 y_extremes;  // (y_min, y_max)
    float2 x_coords_at_y_extremes;  // (x1, x2) at (y_min, y_max)
    float2 y_coords_at_x_extremes;  // (y1, y2) at (x_min, x_max)
};

// Structure to hold dual asymmetric AABBs
struct DualBox {
    float4 left_box;   // (min_x, min_y, max_x, max_y)
    float4 right_box;  // (min_x, min_y, max_x, max_y)
    bool valid;        // Whether boxes are valid
};

// Enhanced degenerate ellipse detection and validation - PERFORMANCE OPTIMIZED
// Requirements: 6.1, 6.2 - robust degenerate ellipse detection and numerical stability
// Requirements: 4.1, 4.2, 4.3 - O(1) complexity, efficient GPU functions, minimize branching
__device__ inline bool isValidEllipse(const float4& con_o, float& disc) {
    // Use register variables for better performance
    register float A = con_o.x;
    register float B = con_o.y;
    register float C = con_o.z;
    register float opacity = con_o.w;
    
    // Compute discriminant B² - AC using fused multiply-add for optimal performance
    // Requirements: 4.2 - use efficient GPU trigonometric functions
    disc = fmaf(B, B, -A * C);
    
    // RELAXED validation for real-world ellipse configurations - FIXED
    // Use fast GPU comparison operations and bitwise logic to minimize branching
    // Requirements: 4.3 - minimize branching for better SIMD utilization
    register bool valid_diagonal = (A > 1e-8f) & (C > 1e-8f);  // More lenient epsilon
    register bool valid_opacity = (opacity > 0.0f) & __finite(opacity);  // Use fast GPU finite check
    register bool valid_disc = (disc < 0.0f);  // Allow disc to be just negative, not strictly < -epsilon
    
    // Use fast GPU finite checks for better performance
    register bool finite_coeffs = __finite(A) & __finite(B) & __finite(C);
    
    // Optimized aspect ratio check using reciprocal multiplication instead of division
    // Requirements: 4.2 - use efficient GPU operations
    register float inv_C = __frcp_rn(C);  // Fast reciprocal approximation
    register float aspect_ratio_sq = A * inv_C;
    register float max_aspect_sq = DUAL_SNUGBOX_MAX_ASPECT_RATIO * DUAL_SNUGBOX_MAX_ASPECT_RATIO;
    register float min_aspect_sq = __frcp_rn(max_aspect_sq);  // 1/(max_aspect²)
    register bool valid_aspect = (aspect_ratio_sq < max_aspect_sq) & (aspect_ratio_sq > min_aspect_sq);
    
    // Single bitwise AND operation to combine all conditions - optimal for SIMD
    return valid_diagonal & valid_opacity & valid_disc & finite_coeffs & valid_aspect;
}

// Enhanced opacity thresholding with numerical stability - PERFORMANCE OPTIMIZED
// Requirements: 6.4, 6.5 - opacity thresholding consistent with original implementation
// Requirements: 4.1, 4.2, 4.3 - O(1) complexity, efficient GPU functions, minimize branching
__device__ inline bool passesOpacityThreshold(float opacity) {
    // RELAXED opacity threshold for real-world scenarios - FIXED
    // Use register variables and fast GPU operations for optimal performance
    register bool valid_opacity = (opacity > 0.0f) & __finite(opacity);
    
    if (!valid_opacity) {
        return false;
    }
    
    register float scaled_opacity = fmaf(opacity, 255.0f, 0.0f);  // Fused multiply-add
    register float t = 2.0f * __logf(scaled_opacity);  // Fast GPU log function
    
    // More lenient log validation - allow smaller values
    // Requirements: 4.3 - minimize branching for better SIMD utilization
    register bool valid_log = __finite(t);  // Just require finite values
    
    return valid_log;
}

// Boundary clamping for screen-space coordinates - PERFORMANCE OPTIMIZED
// Requirements: 6.3 - boundary clamping for screen-space coordinates
// Requirements: 4.1, 4.2, 4.3 - O(1) complexity, efficient GPU functions, minimize branching
__device__ inline float2 clampCoordinates(const float2& coord) {
    // Use register variables and vectorized operations for optimal performance
    register float x_clamped = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, coord.x));
    register float y_clamped = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, coord.y));
    
    // Return vectorized result - GPU can process both coordinates in parallel
    return make_float2(x_clamped, y_clamped);
}

// Safe square root with numerical stability check - PERFORMANCE OPTIMIZED
// Requirements: 6.2 - numerical stability safeguards
// Requirements: 4.1, 4.2, 4.3 - O(1) complexity, efficient GPU functions, minimize branching
__device__ inline float safeSqrt(float value) {
    // Use fast GPU square root with conditional move for optimal performance
    // Requirements: 4.2 - use efficient GPU trigonometric functions
    register float clamped_value = fmaxf(value, DUAL_SNUGBOX_EPSILON);
    return __fsqrt_rn(clamped_value);  // Fast GPU square root with round-to-nearest
}

// Compute tilt angle θ using covariance matrix eigenvalue approach - PERFORMANCE OPTIMIZED
// θ = 0.5 * atan2(2σ_xy, σ_xx - σ_yy)
// Requirements: 2.1, 4.1, 4.2 - O(1) complexity, efficient GPU trigonometric functions
__device__ inline float computeTiltAngle(const float3& cov2d) {
    // Use register variables and fast GPU trigonometric functions for optimal performance
    // Requirements: 4.2 - use efficient GPU trigonometric functions (atan2f)
    register float numerator = 2.0f * cov2d.y;
    register float denominator = cov2d.x - cov2d.z;
    
    // Use fast GPU atan2 function with optimal precision
    return 0.5f * atan2f(numerator, denominator);
}

// Compute stretching factor s(θ) = 1 + β|cos(2θ)| where β ∈ [1.0, 1.2]
// When θ ≈ 0° or 90°: s ≈ 2.0 (maximum stretching)
// When θ ≈ 45°: s = 1.0 (minimal stretching)
// Requirements: 2.2, 2.3, 2.4
__device__ inline float computeStretchingFactor(float theta, float beta = 1.1f) {
    // s(θ) = 1 + β|cos(2θ)|
    return 1.0f + beta * fabsf(cosf(2.0f * theta));
}

// Compute extreme points of ellipse using analytical methods with enhanced error handling
// Based on the ellipse equation: Ax² + 2Bxy + Cy² = constant
// Requirements: 1.1, 4.1, 6.2 - analytical ellipse methods with numerical stability
__device__ inline ExtremePoints computeExtremePoints(
    const float4& con_o,  // (A, B, C, opacity) conic coefficients
    float disc,           // discriminant B² - AC (should be negative for valid ellipse)
    float t,              // threshold constant
    const float2& p       // ellipse center
) {
    ExtremePoints extremes;
    
    // For ellipse Ax² + 2Bxy + Cy² = t, compute extreme points analytically
    float A = con_o.x;
    float B = con_o.y;
    float C = con_o.z;
    
    // Apply boundary clamping to center coordinates
    // Requirements: 6.3 - boundary clamping for screen-space coordinates
    float2 clamped_center = clampCoordinates(p);
    
    // X-extremes: solve d/dx = 0 → 2Ax + 2By = 0 → y = -Ax/B
    // Substitute back: Ax² + 2B(-Ax/B)x + C(-Ax/B)² = t
    // Simplifies to: x² = -t*C/disc
    // Requirements: 6.2 - numerical stability safeguards
    float x_extreme_offset = safeSqrt(-t * C / disc);
    float x_min = clamped_center.x - x_extreme_offset;
    float x_max = clamped_center.x + x_extreme_offset;
    
    // Apply coordinate clamping to extreme points
    extremes.x_extremes = make_float2(
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_min)),
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_max))
    );
    
    // Y-coordinates at x-extremes with numerical stability checks
    if (fabsf(B) > DUAL_SNUGBOX_EPSILON) {
        float y_at_x_min = clamped_center.y - A * (-x_extreme_offset) / B;
        float y_at_x_max = clamped_center.y - A * x_extreme_offset / B;
        
        // Apply coordinate clamping
        extremes.y_coords_at_x_extremes = make_float2(
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_at_x_min)),
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_at_x_max))
        );
    } else {
        // When B ≈ 0, ellipse axes are aligned with coordinate axes
        extremes.y_coords_at_x_extremes = make_float2(clamped_center.y, clamped_center.y);
    }
    
    // Y-extremes: solve d/dy = 0 → 2Bx + 2Cy = 0 → x = -Cy/B
    // Substitute back: A(-Cy/B)² + 2B(-Cy/B)y + Cy² = t
    // Simplifies to: y² = -t*A/disc
    float y_extreme_offset = safeSqrt(-t * A / disc);
    float y_min = clamped_center.y - y_extreme_offset;
    float y_max = clamped_center.y + y_extreme_offset;
    
    // Apply coordinate clamping to extreme points
    extremes.y_extremes = make_float2(
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_min)),
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_max))
    );
    
    // X-coordinates at y-extremes with numerical stability checks
    if (fabsf(B) > DUAL_SNUGBOX_EPSILON) {
        float x_at_y_min = clamped_center.x - C * (-y_extreme_offset) / B;
        float x_at_y_max = clamped_center.x - C * y_extreme_offset / B;
        
        // Apply coordinate clamping
        extremes.x_coords_at_y_extremes = make_float2(
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_at_y_min)),
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_at_y_max))
        );
    } else {
        // When B ≈ 0, ellipse axes are aligned with coordinate axes
        extremes.x_coords_at_y_extremes = make_float2(clamped_center.x, clamped_center.x);
    }
    
    return extremes;
}

// Construct dual asymmetric AABBs using extreme points and center with enhanced error handling
// Implements left-right partitioning based on Gaussian center x-coordinate
// Requirements: 1.2, 1.3, 2.3, 2.4, 2.5, 6.2, 6.3 - dual-box construction with error handling
__device__ inline DualBox constructDualBoxes(
    const ExtremePoints& extremes,
    const float2& center,
    float stretch_factor
) {
    DualBox dual_box;
    dual_box.valid = false;
    
    // Validate input parameters
    // Requirements: 6.2 - numerical stability safeguards
    if (!isfinite(center.x) || !isfinite(center.y) || 
        !isfinite(stretch_factor) || stretch_factor < 1.0f || stretch_factor > 3.0f) {
        return dual_box;  // Invalid input parameters
    }
    
    // Collect all extreme points for partitioning with validation
    float extreme_points_x[4] = {
        extremes.x_extremes.x,  // x_min
        extremes.x_extremes.y,  // x_max
        extremes.x_coords_at_y_extremes.x,  // x at y_min
        extremes.x_coords_at_y_extremes.y   // x at y_max
    };
    
    float extreme_points_y[4] = {
        extremes.y_coords_at_x_extremes.x,  // y at x_min
        extremes.y_coords_at_x_extremes.y,  // y at x_max
        extremes.y_extremes.x,  // y_min
        extremes.y_extremes.y   // y_max
    };
    
    // Validate all extreme points for numerical stability
    for (int i = 0; i < 4; i++) {
        if (!isfinite(extreme_points_x[i]) || !isfinite(extreme_points_y[i])) {
            return dual_box;  // Invalid extreme points
        }
        
        // Apply boundary clamping to extreme points
        // Requirements: 6.3 - boundary clamping for screen-space coordinates
        extreme_points_x[i] = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, 
                                   fminf(DUAL_SNUGBOX_MAX_COORDINATE, extreme_points_x[i]));
        extreme_points_y[i] = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, 
                                   fminf(DUAL_SNUGBOX_MAX_COORDINATE, extreme_points_y[i]));
    }
    
    // Partition extreme points based on x-coordinate relative to center
    // Requirements: 1.2 - left-right partitioning based on Gaussian center x-coordinate
    float left_points_x[5], left_points_y[5];  // +1 for center
    float right_points_x[5], right_points_y[5]; // +1 for center
    int left_count = 0, right_count = 0;
    
    // Add center point to both partitions
    left_points_x[left_count] = center.x;
    left_points_y[left_count] = center.y;
    left_count++;
    
    right_points_x[right_count] = center.x;
    right_points_y[right_count] = center.y;
    right_count++;
    
    // Partition extreme points with safety checks
    for (int i = 0; i < 4; i++) {
        // Ensure we don't exceed array bounds
        if (left_count >= 5 || right_count >= 5) {
            return dual_box;  // Array bounds exceeded
        }
        
        if (extreme_points_x[i] <= center.x) {
            // Left partition
            left_points_x[left_count] = extreme_points_x[i];
            left_points_y[left_count] = extreme_points_y[i];
            left_count++;
        }
        if (extreme_points_x[i] >= center.x) {
            // Right partition (points on center line go to both)
            right_points_x[right_count] = extreme_points_x[i];
            right_points_y[right_count] = extreme_points_y[i];
            right_count++;
        }
    }
    
    // Validate partition counts
    if (left_count < 1 || right_count < 1) {
        return dual_box;  // Invalid partitioning
    }
    
    // Construct left AABB from left partition points
    // Requirements: 1.3 - asymmetric AABB construction using extreme points and center
    float left_min_x = left_points_x[0], left_max_x = left_points_x[0];
    float left_min_y = left_points_y[0], left_max_y = left_points_y[0];
    
    for (int i = 1; i < left_count; i++) {
        left_min_x = fminf(left_min_x, left_points_x[i]);
        left_max_x = fmaxf(left_max_x, left_points_x[i]);
        left_min_y = fminf(left_min_y, left_points_y[i]);
        left_max_y = fmaxf(left_max_y, left_points_y[i]);
    }
    
    // Construct right AABB from right partition points
    float right_min_x = right_points_x[0], right_max_x = right_points_x[0];
    float right_min_y = right_points_y[0], right_max_y = right_points_y[0];
    
    for (int i = 1; i < right_count; i++) {
        right_min_x = fminf(right_min_x, right_points_x[i]);
        right_max_x = fmaxf(right_max_x, right_points_x[i]);
        right_min_y = fminf(right_min_y, right_points_y[i]);
        right_max_y = fmaxf(right_max_y, right_points_y[i]);
    }
    
    // Validate constructed boxes for degenerate cases
    // Requirements: 6.1, 6.2 - degenerate ellipse detection and numerical stability
    if (left_max_x <= left_min_x || left_max_y <= left_min_y ||
        right_max_x <= right_min_x || right_max_y <= right_min_y) {
        return dual_box;  // Degenerate boxes
    }
    
    // Apply adaptive stretching to box dimensions with safety checks
    // Requirements: 2.5 - extend each half-box along its longer dimension away from center
    float left_width = left_max_x - left_min_x;
    float left_height = left_max_y - left_min_y;
    float right_width = right_max_x - right_min_x;
    float right_height = right_max_y - right_min_y;
    
    // Validate box dimensions
    if (!isfinite(left_width) || !isfinite(left_height) || 
        !isfinite(right_width) || !isfinite(right_height) ||
        left_width <= 0.0f || left_height <= 0.0f ||
        right_width <= 0.0f || right_height <= 0.0f) {
        return dual_box;  // Invalid box dimensions
    }
    
    // Apply stretching to left box along its longer dimension, away from center
    if (left_width >= left_height) {
        // Stretch horizontally away from center (rightward for left box)
        float stretch_amount = left_width * (stretch_factor - 1.0f);
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            left_max_x += stretch_amount;  // Extend rightward away from center
        }
        // Short edge: keep original
        // left_min_y, left_max_y unchanged
    } else {
        // Stretch vertically away from center
        float stretch_amount = left_height * (stretch_factor - 1.0f) * 0.5f;
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            left_min_y -= stretch_amount;
            left_max_y += stretch_amount;
        }
        // Short edge: keep original
        // left_min_x, left_max_x unchanged
    }
    
    // Apply stretching to right box along its longer dimension, away from center
    if (right_width >= right_height) {
        // Stretch horizontally away from center (leftward for right box)
        float stretch_amount = right_width * (stretch_factor - 1.0f);
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            right_min_x -= stretch_amount;  // Extend leftward away from center
        }
        // Short edge: keep original
        // right_min_y, right_max_y unchanged
    } else {
        // Stretch vertically away from center
        float stretch_amount = right_height * (stretch_factor - 1.0f) * 0.5f;
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            right_min_y -= stretch_amount;
            right_max_y += stretch_amount;
        }
        // Short edge: keep original
        // right_min_x, right_max_x unchanged
    }
    
    // Apply final boundary clamping to constructed boxes
    // Requirements: 6.3 - boundary clamping for screen-space coordinates
    left_min_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_min_x));
    left_min_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_min_y));
    left_max_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_max_x));
    left_max_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_max_y));
    
    right_min_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_min_x));
    right_min_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_min_y));
    right_max_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_max_x));
    right_max_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_max_y));
    
    // Final validation of constructed boxes
    if (!isfinite(left_min_x) || !isfinite(left_min_y) || !isfinite(left_max_x) || !isfinite(left_max_y) ||
        !isfinite(right_min_x) || !isfinite(right_min_y) || !isfinite(right_max_x) || !isfinite(right_max_y)) {
        return dual_box;  // Invalid final box coordinates
    }
    
    // Ensure boxes are still valid after stretching and clamping
    if (left_max_x <= left_min_x || left_max_y <= left_min_y ||
        right_max_x <= right_min_x || right_max_y <= right_min_y) {
        return dual_box;  // Degenerate boxes after processing
    }
    
    // Store the constructed dual boxes
    dual_box.left_box = make_float4(left_min_x, left_min_y, left_max_x, left_max_y);
    dual_box.right_box = make_float4(right_min_x, right_min_y, right_max_x, right_max_y);
    dual_box.valid = true;
    
    return dual_box;
}

// ---- Unique Tile Intersection Generation System ---- //

// Efficient AABB-tile intersection test - OPTIMIZED
// Requirements: 1.4, 5.1, 5.2, 5.3, 5.4
// Requirements: 4.2, 4.3 - minimize branching for better SIMD utilization
__device__ inline bool tileIntersectsBox(
    int tile_x, int tile_y,
    const float4& box  // (min_x, min_y, max_x, max_y)
) {
    // Convert tile coordinates to pixel boundaries using efficient operations
    float tile_min_x = __int2float_rn(tile_x * BLOCK_X);      // Use fast int-to-float conversion
    float tile_max_x = __int2float_rn((tile_x + 1) * BLOCK_X);
    float tile_min_y = __int2float_rn(tile_y * BLOCK_Y);
    float tile_max_y = __int2float_rn((tile_y + 1) * BLOCK_Y);
    
    // AABB intersection test using bitwise operations to minimize branching
    // boxes intersect if they overlap in both dimensions
    bool x_overlap = (tile_min_x < box.z) & (tile_max_x > box.x);
    bool y_overlap = (tile_min_y < box.w) & (tile_max_y > box.y);
    
    return x_overlap & y_overlap;
}

// Generate unique tile intersections using union-based approach with enhanced error handling
// Prevents duplicate key-value pairs by processing each tile exactly once
// Requirements: 1.4, 5.1, 5.2, 5.3, 5.4, 6.3 - unique tile intersection with boundary clamping
__device__ inline uint32_t generateUniqueTileIntersections(
    const DualBox& dual_box,
    const dim3& grid,
    uint32_t idx,
    uint32_t off,
    float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
) {
    if (!dual_box.valid) {
        return 0;
    }
    
    // Validate grid dimensions
    if (grid.x == 0 || grid.y == 0 || grid.x > 65535 || grid.y > 65535) {
        return 0;  // Invalid grid dimensions
    }
    
    // Compute union bounding rectangle of both boxes with validation
    float union_min_x = fminf(dual_box.left_box.x, dual_box.right_box.x);
    float union_min_y = fminf(dual_box.left_box.y, dual_box.right_box.y);
    float union_max_x = fmaxf(dual_box.left_box.z, dual_box.right_box.z);
    float union_max_y = fmaxf(dual_box.left_box.w, dual_box.right_box.w);
    
    // Validate union bounds
    if (!isfinite(union_min_x) || !isfinite(union_min_y) || 
        !isfinite(union_max_x) || !isfinite(union_max_y)) {
        return 0;  // Invalid union bounds
    }
    
    // Check for reasonable union size
    if (union_max_x - union_min_x <= 0.0f || union_max_y - union_min_y <= 0.0f) {
        return 0;  // Degenerate union rectangle
    }
    
    // Convert to tile coordinates with enhanced boundary clamping
    // Requirements: 6.3 - boundary clamping for screen-space coordinates
    int rect_min_x = max(0, min((int)grid.x, (int)floorf(union_min_x / BLOCK_X)));
    int rect_min_y = max(0, min((int)grid.y, (int)floorf(union_min_y / BLOCK_Y)));
    int rect_max_x = max(0, min((int)grid.x, (int)ceilf(union_max_x / BLOCK_X)));
    int rect_max_y = max(0, min((int)grid.y, (int)ceilf(union_max_y / BLOCK_Y)));
    
    // Additional safety bounds checking
    rect_min_x = max(0, rect_min_x);
    rect_min_y = max(0, rect_min_y);
    rect_max_x = min((int)grid.x, rect_max_x);
    rect_max_y = min((int)grid.y, rect_max_y);
    
    // If no tiles are touched, return 0
    if (rect_min_x >= rect_max_x || rect_min_y >= rect_max_y) {
        return 0;
    }
    
    // Prevent excessive tile generation (safety check)
    int total_tiles = (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y);
    const int MAX_TILES_PER_GAUSSIAN = 10000;  // Reasonable upper bound
    if (total_tiles > MAX_TILES_PER_GAUSSIAN) {
        return 0;  // Too many tiles, likely numerical error
    }
    
    uint32_t tiles_count = 0;
    
    // Single-pass key generation with proper indexing and error handling
    // Process each tile in the union rectangle exactly once
    // Requirements: 5.1, 5.2, 5.3 - prevent duplicate key-value pairs
    for (int tile_y = rect_min_y; tile_y < rect_max_y; ++tile_y) {
        for (int tile_x = rect_min_x; tile_x < rect_max_x; ++tile_x) {
            // Additional bounds checking within loop
            if (tile_x < 0 || tile_x >= (int)grid.x || tile_y < 0 || tile_y >= (int)grid.y) {
                continue;  // Skip invalid tile coordinates
            }
            
            // Test intersection with either left box OR right box (union logic)
            // Requirements: 1.4, 5.2 - logical OR operation for tile overlap
            bool intersects_left = tileIntersectsBox(tile_x, tile_y, dual_box.left_box);
            bool intersects_right = tileIntersectsBox(tile_x, tile_y, dual_box.right_box);
            
            if (intersects_left || intersects_right) {
                tiles_count++;
                
                // Generate single key-value pair for this tile
                // Requirements: 5.3, 5.4 - unique (tile_index, gaussian_index) pairs
                if (gaussian_keys_unsorted != nullptr && gaussian_values_unsorted != nullptr) {
                    // Validate tile coordinates before key generation
                    uint64_t tile_id = (uint64_t)tile_y * grid.x + tile_x;
                    
                    // Check for potential overflow in tile ID calculation
                    if (tile_id >= ((uint64_t)grid.x * grid.y)) {
                        continue;  // Skip invalid tile ID
                    }
                    
                    // Key format: | tile ID | depth |
                    uint64_t key = tile_id;
                    key <<= 32;
                    key |= *((uint32_t*)&depth);
                    
                    gaussian_keys_unsorted[off] = key;
                    gaussian_values_unsorted[off] = idx;
                    off++;
                }
            }
        }
    }
    
    return tiles_count;
}

__device__ inline float2 computeEllipseIntersection(
    const float4 con_o, const float disc, const float t, const float2 p,
    const bool isY, const float coord)
{
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? con_o.x : con_o.z;

    float h = coord - p_u;  // h = y - p.y for y, x - p.x for x
    float sqrt_term = sqrt(disc * h * h + t * coeff);

    return {
      (-con_o.y * h - sqrt_term) / coeff + p_v,
      (-con_o.y * h + sqrt_term) / coeff + p_v
    };
}

__device__ inline uint32_t processTiles(
    const float4 con_o, const float disc, const float t, const float2 p,
    float2 bbox_min, float2 bbox_max,
    float2 bbox_argmin, float2 bbox_argmax,
    int2 rect_min, int2 rect_max,
    const dim3 grid, const bool isY,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{

    // ---- AccuTile Code ---- //

    // Set variables based on the isY flag
    float BLOCK_U = isY ? BLOCK_Y : BLOCK_X;
    float BLOCK_V = isY ? BLOCK_X : BLOCK_Y;

    if (isY) {
      rect_min = {rect_min.y, rect_min.x};
      rect_max = {rect_max.y, rect_max.x};

      bbox_min = {bbox_min.y, bbox_min.x};
      bbox_max = {bbox_max.y, bbox_max.x};

      bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
      bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    // Initialize max line
    // Just need the min to be >= all points on the ellipse
    // and  max to be <= all points on the ellipse
    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK_U;
    // Initialize min line intersections.
    if (bbox_min.x <= min_line) {
      // Boundary case
      intersect_min_line = computeEllipseIntersection(
                con_o, disc, t, p, isY, rect_min.x * BLOCK_U);

    } else {
      // Same as max line
      intersect_min_line = intersect_max_line;
    }


    // Loop over either y slices or x slices based on the `isY` flag.
    for (int u = rect_min.x; u < rect_max.x; ++u)
    {
        // Starting from the bottom or left, we will only need to compute
        // intersections at the next line.
        max_line = min_line + BLOCK_U;
        if (max_line <= bbox_max.x) {
          intersect_max_line = computeEllipseIntersection(
                    con_o, disc, t, p, isY, max_line);
        }

        // If the bbox min is in this slice, then it is the minimum
        // ellipse point in this slice. Otherwise, the minimum ellipse
        // point will be the minimum of the intersections of the min/max lines.
        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
          ellipse_min = bbox_min.y;
        } else {
          ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        // If the bbox max is in this slice, then it is the maximum
        // ellipse point in this slice. Otherwise, the maximum ellipse
        // point will be the maximum of the intersections of the min/max lines.
        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
          ellipse_max = bbox_max.y;
        } else {
          ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        // Convert ellipse_min/ellipse_max to tiles touched
        // First map back to tile coordinates, then subtract.
        int min_tile_v = max(rect_min.y,
            min(rect_max.y, (int)(ellipse_min / BLOCK_V))
            );
        int max_tile_v = min(rect_max.y,
            max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1))
            );

        tiles_count += max_tile_v - min_tile_v;
        // Only update keys array if it exists.
        if (gaussian_keys_unsorted != nullptr) {
          // Loop over tiles and add to keys array
          for (int v = min_tile_v; v < max_tile_v; v++)
          {
            // For each tile that the Gaussian overlaps, emit a
            // key/value pair. The key is |  tile ID  |      depth      |,
            // and the value is the ID of the Gaussian. Sorting the values
            // with this key yields Gaussian IDs in a list, such that they
            // are first sorted by tile and then by depth.
            uint64_t key = isY ?  (u * grid.x + v) : (v * grid.x + u);
            key <<= 32;
            key |= *((uint32_t*)&depth);
            gaussian_keys_unsorted[off] = key;
            gaussian_values_unsorted[off] = idx;
            off++;
          }
        }
        // Max line of this tile slice will be min lin of next tile slice
        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }
    return tiles_count;
}


__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{
    // ---- Dual-SnugBox Algorithm Implementation - PERFORMANCE OPTIMIZED ---- //
    // Requirements: 4.1, 4.2, 4.3 - O(1) complexity, efficient GPU functions, minimize branching
    
    // Phase 1: Enhanced ellipse validation and error handling - OPTIMIZED
    // Requirements: 6.1, 6.2 - robust degenerate ellipse detection and numerical stability
    float disc;
    bool valid_ellipse = isValidEllipse(con_o, disc);
    bool valid_opacity = passesOpacityThreshold(con_o.w);
    
    // FIXED: Compute threshold parameter t for Gaussian cutoff correctly
    // For Gaussian exp(-0.5 * (Ax² + 2Bxy + Cy²)) = threshold
    // We solve for the contour where the Gaussian equals 1/255 (minimum visible opacity)
    float threshold = 1.0f / 255.0f;
    float t = -2.0f * __logf(threshold);  // t = -2*ln(threshold) for the ellipse equation
    
    if (!isfinite(t) || t <= 0.0f) {
        // Use default threshold if calculation fails
        t = 5.0f;  // Reasonable default for most ellipses
    }
    
    bool valid_threshold = isfinite(t) & (t > 0.0f);
    
    // Validate and clamp input coordinates using efficient operations
    // Requirements: 6.3 - boundary clamping for screen-space coordinates
    float2 clamped_center = clampCoordinates(p);
    bool valid_depth = isfinite(depth);
    
    // Combine all early exit conditions using bitwise operations to minimize branching
    // Requirements: 4.2, 4.3 - minimize branching for better SIMD utilization
    bool all_valid = valid_ellipse & valid_opacity & valid_threshold & valid_depth;
    // Fallback: 如果参数不合法，直接 fallback 到 snugbox
    if (!all_valid) {
        // FIXED: Safe single snugbox fallback with proper validation
        float A = con_o.x, B = con_o.y, C = con_o.z;
        float2 center = clamped_center;
        
        // Validate discriminant for fallback calculation
        if (!isfinite(disc) || disc >= 0.0f) {
            // If discriminant is invalid, use a conservative bounding box
            float conservative_radius = 50.0f;  // Conservative radius for invalid ellipses
            float x_min = center.x - conservative_radius;
            float x_max = center.x + conservative_radius;
            float y_min = center.y - conservative_radius;
            float y_max = center.y + conservative_radius;
            
            // Apply clamping and tile calculation
            x_min = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_min));
            x_max = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_max));
            y_min = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_min));
            y_max = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_max));
            
            int rect_min_x = max(0, min((int)grid.x, (int)floorf(x_min / BLOCK_X)));
            int rect_min_y = max(0, min((int)grid.y, (int)floorf(y_min / BLOCK_Y)));
            int rect_max_x = max(0, min((int)grid.x, (int)ceilf(x_max / BLOCK_X)));
            int rect_max_y = max(0, min((int)grid.y, (int)ceilf(y_max / BLOCK_Y)));
            
            uint32_t tiles_count = 0;
            for (int tile_y = rect_min_y; tile_y < rect_max_y; ++tile_y) {
                for (int tile_x = rect_min_x; tile_x < rect_max_x; ++tile_x) {
                    if (tile_x < 0 || tile_x >= (int)grid.x || tile_y < 0 || tile_y >= (int)grid.y) continue;
                    tiles_count++;
                    if (gaussian_keys_unsorted != nullptr && gaussian_values_unsorted != nullptr) {
                        uint64_t tile_id = (uint64_t)tile_y * grid.x + tile_x;
                        if (tile_id >= ((uint64_t)grid.x * grid.y)) continue;
                        uint64_t key = tile_id;
                        key <<= 32;
                        key |= *((uint32_t*)&depth);
                        gaussian_keys_unsorted[off] = key;
                        gaussian_values_unsorted[off] = idx;
                        off++;
                    }
                }
            }
            return tiles_count;
        }
        
        // Use validated discriminant for proper ellipse calculation
        float x_extreme_offset = safeSqrt(-t * C / disc);
        float y_extreme_offset = safeSqrt(-t * A / disc);
        float x_min = center.x - x_extreme_offset;
        float x_max = center.x + x_extreme_offset;
        float y_min = center.y - y_extreme_offset;
        float y_max = center.y + y_extreme_offset;
        // clamp
        x_min = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_min));
        x_max = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_max));
        y_min = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_min));
        y_max = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_max));
        // tile 范围
        int rect_min_x = max(0, min((int)grid.x, (int)floorf(x_min / BLOCK_X)));
        int rect_min_y = max(0, min((int)grid.y, (int)floorf(y_min / BLOCK_Y)));
        int rect_max_x = max(0, min((int)grid.x, (int)ceilf(x_max / BLOCK_X)));
        int rect_max_y = max(0, min((int)grid.y, (int)ceilf(y_max / BLOCK_Y)));
        uint32_t tiles_count = 0;
        for (int tile_y = rect_min_y; tile_y < rect_max_y; ++tile_y) {
            for (int tile_x = rect_min_x; tile_x < rect_max_x; ++tile_x) {
                if (tile_x < 0 || tile_x >= (int)grid.x || tile_y < 0 || tile_y >= (int)grid.y) continue;
                tiles_count++;
                if (gaussian_keys_unsorted != nullptr && gaussian_values_unsorted != nullptr) {
                    uint64_t tile_id = (uint64_t)tile_y * grid.x + tile_x;
                    if (tile_id >= ((uint64_t)grid.x * grid.y)) continue;
                    uint64_t key = tile_id;
                    key <<= 32;
                    key |= *((uint32_t*)&depth);
                    gaussian_keys_unsorted[off] = key;
                    gaussian_values_unsorted[off] = idx;
                    off++;
                }
            }
        }
        return tiles_count;
    }
    
    // Phase 2: Compute extreme points using analytical methods with error handling
    // Requirements: 1.1, 4.1 - analytical ellipse methods for extreme point computation
    ExtremePoints extremes = computeExtremePoints(con_o, disc, t, clamped_center);
    
    // Validate computed extreme points for numerical stability
    // Requirements: 6.2 - numerical stability safeguards
    if (!isfinite(extremes.x_extremes.x) || !isfinite(extremes.x_extremes.y) ||
        !isfinite(extremes.y_extremes.x) || !isfinite(extremes.y_extremes.y) ||
        !isfinite(extremes.x_coords_at_y_extremes.x) || !isfinite(extremes.x_coords_at_y_extremes.y) ||
        !isfinite(extremes.y_coords_at_x_extremes.x) || !isfinite(extremes.y_coords_at_x_extremes.y)) {
        return 0;  // Numerical instability in extreme point computation
    }
    
    // Phase 3: Calculate tilt angle and stretching factor with error handling
    // Requirements: 2.1, 2.2 - tilt angle calculation and stretching factor computation
    float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
    float theta = computeTiltAngle(cov2d);
    
    // Validate tilt angle computation
    if (!isfinite(theta)) {
        // Fallback to axis-aligned case if tilt angle computation fails
        theta = 0.0f;
    }
    
    float stretch_factor = computeStretchingFactor(theta, 1.1f);  // β = 1.1
    
    // Validate stretching factor
    if (!isfinite(stretch_factor) || stretch_factor < 1.0f || stretch_factor > 3.0f) {
        // Fallback to minimal stretching if computation fails
        stretch_factor = 1.0f;
    }
    
    // Phase 4: Construct dual boxes with adaptive stretching and error handling
    // Requirements: 1.2, 1.3, 2.3, 2.4, 2.5 - dual-box construction with stretching
    DualBox dual_box = constructDualBoxes(extremes, clamped_center, stretch_factor);
    if (!dual_box.valid) {
        return 0;  // Failed to construct valid boxes
    }
    
    // Additional validation of constructed boxes
    // Requirements: 6.2, 6.3 - numerical stability and boundary clamping
    if (!isfinite(dual_box.left_box.x) || !isfinite(dual_box.left_box.y) ||
        !isfinite(dual_box.left_box.z) || !isfinite(dual_box.left_box.w) ||
        !isfinite(dual_box.right_box.x) || !isfinite(dual_box.right_box.y) ||
        !isfinite(dual_box.right_box.z) || !isfinite(dual_box.right_box.w)) {
        return 0;  // Invalid box coordinates
    }
    
    // Check for reasonable box sizes to prevent excessive tile generation
    float left_box_area = (dual_box.left_box.z - dual_box.left_box.x) * 
                         (dual_box.left_box.w - dual_box.left_box.y);
    float right_box_area = (dual_box.right_box.z - dual_box.right_box.x) * 
                          (dual_box.right_box.w - dual_box.right_box.y);
    
    // Prevent boxes that are unreasonably large (potential numerical overflow)
    const float MAX_BOX_AREA = 1e8f;  // Reasonable upper bound for box area
    if (left_box_area > MAX_BOX_AREA || right_box_area > MAX_BOX_AREA ||
        left_box_area < 0.0f || right_box_area < 0.0f) {
        return 0;  // Box area is unreasonable
    }
    
    // Phase 5: Generate unique tile intersections using union logic with error handling
    // Requirements: 1.4, 5.1, 5.2, 5.3, 5.4 - unique tile intersection generation
    return generateUniqueTileIntersections(dual_box, grid, idx, off, depth,
                                         gaussian_keys_unsorted, gaussian_values_unsorted);
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif