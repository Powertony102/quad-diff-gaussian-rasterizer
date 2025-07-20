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
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#define M_PI_2 1.5707963267948966f
#endif

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
	// float p_w = 1.0f / (p_hom.w + 0.0000001f); // Unused variable
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // Commented out unused variable
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

// Structure to hold dual asymmetric AABBs
struct DualBox {
    float4 left_box;   // (min_x, min_y, max_x, max_y)
    float4 right_box;  // (min_x, min_y, max_x, max_y)
    bool valid;        // Whether boxes are valid
};

// Structure to hold quad boxes
struct QuadBox {
    float4 left_box; 
    float4 left_small_box;  
    float4 right_box;  
    float4 right_small_box;
    bool valid;        
};

// Compute tilt angle θ using covariance matrix eigenvalue approach - PERFORMANCE OPTIMIZED
// θ = 0.5 * atan2(2σ_xy, σ_xx - σ_yy)
// Requirements: 2.1, 4.1, 4.2 - O(1) complexity, efficient GPU trigonometric functions
__device__ inline float computeTiltAngle(const float3& cov2d) {
    // Use register variables and fast GPU trigonometric functions for optimal performance
    // Requirements: 4.2 - use efficient GPU trigonometric functions (atan2f)
    register float numerator = 2.0f * cov2d.y;
    register float denominator = cov2d.x - cov2d.z;
    
    // Use fast GPU atan2 function with optimal precision
    float angle = 0.5f * atan2f(numerator, denominator);

    if (angle < 0.0f) {
        angle += M_PI;
    }

    return angle;
}

// Compute eccentricity of a 2D Gaussian ellipse
// The ellipse is defined by Ax² + 2Bxy + Cy² = const
// Requirements: A, B, C are conic coefficients from con_o
__device__ inline float computeEccentricity(const float4& con_o) {
    float A = con_o.x;
    float B = con_o.y;
    float C = con_o.z;

    // The matrix M = [[A, B], [B, C]] is related to the inverse of the 2D Gaussian's covariance matrix.
    // The eccentricity 'e' can be derived from the eigenvalues of M.
    // e = sqrt(1 - (lambda_min / lambda_max)), where lambda_min and lambda_max
    // are the smaller and larger eigenvalues of M.

    float diff_AC = A - C;
    // Using fmaf for potentially better performance on some hardware
    float term_under_sqrt = fmaf(diff_AC, diff_AC, 4.0f * B * B);
    float term_sqrt = sqrtf(term_under_sqrt);
    
    float sum_AC = A + C;

    // Eigenvalues of M
    float lambda_max = (sum_AC + term_sqrt) / 2.0f;
    float lambda_min = (sum_AC - term_sqrt) / 2.0f;

    float ratio = lambda_min / lambda_max;

    return sqrtf(1.0f - ratio);
}

__device__ inline QuadBox constructQuadBoxes(
    const float4& con_o,
    const float disc,
    const float t,
    const float2& center,
    float theta, // tilt angle in radians
    float eccentricity
) {
    QuadBox quad_box;
    quad_box.valid = false;

    // 使用 computeEllipseIntersection 计算精确的椭圆边界
    float x_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.x));
    x_term = (con_o.y < 0) ? x_term : -x_term;
    float y_term = sqrt(-(con_o.y * con_o.y * t) / (disc * con_o.z));
    y_term = (con_o.y < 0) ? y_term : -y_term;

    float2 bbox_argmin = { center.y - y_term, center.x - x_term };
    float2 bbox_argmax = { center.y + y_term, center.x + x_term };

    float2 bbox_min = {
      computeEllipseIntersection(con_o, disc, t, center, true, bbox_argmin.x).x,
      computeEllipseIntersection(con_o, disc, t, center, false, bbox_argmin.y).x
    };
    float2 bbox_max = {
      computeEllipseIntersection(con_o, disc, t, center, true, bbox_argmax.x).y,
      computeEllipseIntersection(con_o, disc, t, center, false, bbox_argmax.y).y
    };

    // 使用精确计算的边界构造 snugbox
    const float snug_min_x = bbox_min.x;
    const float snug_max_x = bbox_max.x;
    const float snug_min_y = bbox_min.y;
    const float snug_max_y = bbox_max.y;

    // Calculate extension coefficient f(e,theta)
    // f(e,theta) = 1 / sqrt(1 + (e^4 / (4*(1-e^2))) * sin^2(2*theta))
    float e_sq = eccentricity * eccentricity;
    if (e_sq >= 1.0f) e_sq = 0.999f; // prevent division by zero
    float sin_2theta = sinf(2.0f * theta);
    float sin_2theta_sq = sin_2theta * sin_2theta;
    float stretch_factor = 1.0f / sqrtf(1.0f + (e_sq * e_sq / (4.0f * (1.0f - e_sq))) * sin_2theta_sq);

    float left_rect_x, left_rect_y, left_rect_width, left_rect_height;
    float right_rect_x, right_rect_y, right_rect_width, right_rect_height;
    float left_small_rect_x, left_small_rect_y, left_small_width, left_small_height;
    float right_small_rect_x, right_small_rect_y, right_small_width, right_small_height;

    if (theta >= 0 && theta <= M_PI_2) // 0° to 90° (0 to π/2)
    {
        // 左矩形: 左下点为大矩形左下点，右上点为椭圆中心
        left_rect_x = snug_min_x;
        left_rect_y = snug_min_y;
        left_rect_width = center.x - snug_min_x;
        left_rect_height = center.y - snug_min_y;

        // 右矩形: 右上点为大矩形右上点，左下点为椭圆中心
        right_rect_x = center.x;
        right_rect_y = center.y;
        right_rect_width = snug_max_x - center.x;
        right_rect_height = snug_max_y - center.y;

        // 左小矩形: 右下点为椭圆中心
        left_small_width = left_rect_width * stretch_factor;
        left_small_height = left_rect_height * stretch_factor;
        left_small_rect_x = center.x - left_small_width;
        left_small_rect_y = center.y;

        // 右小矩形: 左上点为椭圆中心
        right_small_width = left_small_width;  // 与左小矩形相同
        right_small_height = left_small_height;
        right_small_rect_x = center.x;
        right_small_rect_y = center.y - right_small_height;
    }
    else // θ > 90° (θ > π/2)
    {
        // 左矩形: 左上点为大矩形左上点，右下点为椭圆中心
        left_rect_x = snug_min_x;
        left_rect_y = center.y;
        left_rect_width = center.x - snug_min_x;
        left_rect_height = snug_max_y - center.y;

        // 右矩形: 右下点为大矩形右下点，左上点为椭圆中心
        right_rect_x = center.x;
        right_rect_y = snug_min_y;
        right_rect_width = snug_max_x - center.x;
        right_rect_height = center.y - snug_min_y;

        // 左小矩形和右小矩形
        left_small_width = left_rect_width * stretch_factor;
        left_small_height = left_rect_height * stretch_factor;
        left_small_rect_x = center.x - left_small_width;
        left_small_rect_y = center.y - left_small_height;

        right_small_width = left_small_width;  // 与左小矩形相同
        right_small_height = left_small_height;
        right_small_rect_x = center.x;
        right_small_rect_y = center.y;
    }
    
    // Store the constructed quad boxes (min_x, min_y, max_x, max_y)
    quad_box.left_box = make_float4(left_rect_x, left_rect_y, 
                                   left_rect_x + left_rect_width, 
                                   left_rect_y + left_rect_height);
    
    quad_box.right_box = make_float4(right_rect_x, right_rect_y, 
                                    right_rect_x + right_rect_width, 
                                    right_rect_y + right_rect_height);
    
    quad_box.left_small_box = make_float4(left_small_rect_x, left_small_rect_y,
                                         left_small_rect_x + left_small_width,
                                         left_small_rect_y + left_small_height);
    
    quad_box.right_small_box = make_float4(right_small_rect_x, right_small_rect_y,
                                          right_small_rect_x + right_small_width,
                                          right_small_rect_y + right_small_height);
    
    quad_box.valid = true;
    
    return quad_box;
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

__device__ inline uint32_t generateUniqueTileIntersectionsQuad(
    const QuadBox& quad_box,
    const dim3& grid,
    uint32_t idx,
    uint32_t off,
    float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
) {
    // Compute union bounding rectangle of all four boxes with validation
    float union_min_x = fminf(fminf(quad_box.left_box.x, quad_box.right_box.x),
                             fminf(quad_box.left_small_box.x, quad_box.right_small_box.x));
    float union_min_y = fminf(fminf(quad_box.left_box.y, quad_box.right_box.y),
                             fminf(quad_box.left_small_box.y, quad_box.right_small_box.y));
    float union_max_x = fmaxf(fmaxf(quad_box.left_box.z, quad_box.right_box.z),
                             fmaxf(quad_box.left_small_box.z, quad_box.right_small_box.z));
    float union_max_y = fmaxf(fmaxf(quad_box.left_box.w, quad_box.right_box.w),
                             fmaxf(quad_box.left_small_box.w, quad_box.right_small_box.w));
    
    // Convert to tile coordinates with enhanced boundary clamping
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
    for (int tile_y = rect_min_y; tile_y < rect_max_y; ++tile_y) {
        for (int tile_x = rect_min_x; tile_x < rect_max_x; ++tile_x) {
            // Additional bounds checking within loop
            if (tile_x < 0 || tile_x >= (int)grid.x || tile_y < 0 || tile_y >= (int)grid.y) {
                continue;  // Skip invalid tile coordinates
            }
            
            // Test intersection with any of the four boxes (union logic)
            bool intersects_left = tileIntersectsBox(tile_x, tile_y, quad_box.left_box);
            bool intersects_right = tileIntersectsBox(tile_x, tile_y, quad_box.right_box);
            bool intersects_left_small = tileIntersectsBox(tile_x, tile_y, quad_box.left_small_box);
            bool intersects_right_small = tileIntersectsBox(tile_x, tile_y, quad_box.right_small_box);
            
            if (intersects_left || intersects_right || intersects_left_small || intersects_right_small) {
                tiles_count++;
                
                // Generate single key-value pair for this tile
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

__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{
    // 保留：计算 theta 和 eccentricity（用于 quadbox 构造）
    float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
    float theta = computeTiltAngle(cov2d);
    float eccentricity = computeEccentricity(con_o);

    // 保持现有逻辑不变
    QuadBox quad_box = constructQuadBoxes(con_o, disc, t, p, theta, eccentricity);

    return generateUniqueTileIntersectionsQuad(
        quad_box, grid, idx, off, depth,
        gaussian_keys_unsorted, gaussian_values_unsorted
    );
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