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
	float invsum32 = 1.0f / sqrtf(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrtf(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrtf(sum2 * sum2 * sum2);

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

__device__ inline float2 computeEllipseIntersection(
    const float4 con_o, const float disc, const float t, const float2 p,
    const bool isY, const float coord)
{
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? con_o.x : con_o.z;

    float h = coord - p_u;  // h = y - p.y for y, x - p.x for x
    float sqrt_term = sqrtf(disc * h * h + t * coeff);

    return {
      (-con_o.y * h - sqrt_term) / coeff + p_v,
      (-con_o.y * h + sqrt_term) / coeff + p_v
    };
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

// Structure to hold quad boxes
struct QuadBox {
    float4 left_box; 
    float4 left_small_box;  
    float4 right_box;  
    float4 right_small_box;
    float2 center; // ellipse center
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

    float x_extent = sqrtf(-t * con_o.z / disc);
    float y_extent = sqrtf(-t * con_o.x / disc);
    
    // 使用精确计算的边界构造 snugbox
    const float snug_min_x = center.x - x_extent;
    const float snug_max_x = center.x + x_extent;
    const float snug_min_y = center.y - y_extent;
    const float snug_max_y = center.y + y_extent;

    quad_box.center = center;

    // Calculate extension coefficient f(e,theta)
    // f(e,theta) = 1 / sqrt(1 + (e^4 / (4*(1-e^2))) * sin^2(2*theta))
    float e2 = fminf(eccentricity * eccentricity, 0.999f);
    float e4 = e2 * e2;

    float s, c;
    __sincosf_fast(theta, &s, &c);            // 1×trig_fast
    float sin_2theta = 2.0f * s * c;
    float sin_2theta_sq = sin_2theta * sin_2theta;
    
    // float sin_2theta = sinf(2.0f * theta);
    // float sin_2theta_sq = sin_2theta * sin_2theta;
    float stretch_factor = 1.0f / sqrtf(1.0f + (e4 / (4.0f * (1.0f - e2))) * sin_2theta_sq);

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

//--- Unique Tile Intersection Generation System ---- //
__device__ inline uint32_t generateUniqueTileIntersectionsQuad(
    const QuadBox& quad_box,
    const dim3& grid,
    uint32_t idx,
    uint32_t off,
    float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
) {
    uint32_t tiles_count = 0;
    
    // 1. 计算SnugBox的边界
    float snug_min_x = fminf(fminf(quad_box.left_box.x, quad_box.left_small_box.x), 
                            fminf(quad_box.right_box.x, quad_box.right_small_box.x));
    float snug_max_x = fmaxf(fmaxf(quad_box.left_box.z, quad_box.left_small_box.z), 
                            fmaxf(quad_box.right_box.z, quad_box.right_small_box.z));
    float snug_min_y = fminf(fminf(quad_box.left_box.y, quad_box.left_small_box.y), 
                            fminf(quad_box.right_box.y, quad_box.right_small_box.y));
    float snug_max_y = fmaxf(fmaxf(quad_box.left_box.w, quad_box.left_small_box.w), 
                            fmaxf(quad_box.right_box.w, quad_box.right_small_box.w));
    
    // 转换为tile坐标
    int snug_min_tile_x = max(0, min((int)grid.x, (int)floorf(snug_min_x / BLOCK_X)));
    int snug_max_tile_x = max(0, min((int)grid.x, (int)ceilf(snug_max_x / BLOCK_X)));
    int snug_min_tile_y = max(0, min((int)grid.y, (int)floorf(snug_min_y / BLOCK_Y)));
    int snug_max_tile_y = max(0, min((int)grid.y, (int)ceilf(snug_max_y / BLOCK_Y)));
    
    // 2. 确定短边和长边
    int rect_a = snug_max_tile_x - snug_min_tile_x;  // x方向tile数量
    int rect_b = snug_max_tile_y - snug_min_tile_y;  // y方向tile数量
    
    bool x_is_short = (rect_a <= rect_b);  // x方向是短边

    bool null_array = (gaussian_keys_unsorted == nullptr && gaussian_values_unsorted == nullptr);
    
    // 3. 沿短边遍历
    if (x_is_short) {
        // x是短边，外层循环遍历x，内层循环遍历y
        for (int tile_x = snug_min_tile_x; tile_x < snug_max_tile_x; ++tile_x) {
            
            // 3.1 计算当前列(tile_x)上，四个box的y范围并集
            float tile_x_min = tile_x * BLOCK_X;
            float tile_x_max = (tile_x + 1) * BLOCK_X;
            
            // 找出在当前列上有覆盖的box，并求y范围的并集
            float col_min_y = 1e9f, col_max_y = -1e9f;
            bool has_coverage = false;
            
            // 检查四个box在当前列是否有覆盖
            float4 boxes[4] = {quad_box.left_box, quad_box.left_small_box, 
                              quad_box.right_box, quad_box.right_small_box};
            
            for (int b = 0; b < 4; ++b) {
                // 修复：检查tile区间与box区间是否有交集
                if (!(tile_x_max <= boxes[b].x || tile_x_min >= boxes[b].z)) {
                    // 当前box在此列有覆盖
                    col_min_y = fminf(col_min_y, boxes[b].y);
                    col_max_y = fmaxf(col_max_y, boxes[b].w);
                    has_coverage = true;
                }
            }
            
            if (!has_coverage) continue;  // 当前列没有任何box覆盖，跳过
            
            // 3.2 转换为tile坐标
            int start_tile_y = max(snug_min_tile_y, max(0, min((int)grid.y, (int)floorf(col_min_y / BLOCK_Y))));
            int end_tile_y = min(snug_max_tile_y, max(0, min((int)grid.y, (int)ceilf(col_max_y / BLOCK_Y))));
            
            tiles_count += (end_tile_y - start_tile_y);
            if (null_array) continue;

            // 3.3 内层循环遍历y
            for (int tile_y = start_tile_y; tile_y < end_tile_y; ++tile_y) {
                uint64_t key = ((uint64_t)tile_y * grid.x + tile_x) << 32 | *((uint32_t*)&depth);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    } else {
        // y是短边，外层循环遍历y，内层循环遍历x
        for (int tile_y = snug_min_tile_y; tile_y < snug_max_tile_y; ++tile_y) {
            
            // 3.1 计算当前行(tile_y)上，四个box的x范围并集
            float tile_y_min = tile_y * BLOCK_Y;
            float tile_y_max = (tile_y + 1) * BLOCK_Y;
            
            // 找出在当前行上有覆盖的box，并求x范围的并集
            float row_min_x = 1e9f, row_max_x = -1e9f;
            bool has_coverage = false;
            
            // 检查四个box在当前行是否有覆盖
            float4 boxes[4] = {quad_box.left_box, quad_box.left_small_box, 
                              quad_box.right_box, quad_box.right_small_box};
            
            for (int b = 0; b < 4; ++b) {
                // 修复：检查tile区间与box区间是否有交集
                if (!(tile_y_max <= boxes[b].y || tile_y_min >= boxes[b].w)) {
                    // 当前box在此行有覆盖
                    row_min_x = fminf(row_min_x, boxes[b].x);
                    row_max_x = fmaxf(row_max_x, boxes[b].z);
                    has_coverage = true;
                }
            }
            
            if (!has_coverage) continue;  // 当前行没有任何box覆盖，跳过
            
            // 3.2 转换为tile坐标
            int start_tile_x = max(snug_min_tile_x, max(0, min((int)grid.x, (int)floorf(row_min_x / BLOCK_X))));
            int end_tile_x = min(snug_max_tile_x, max(0, min((int)grid.x, (int)ceilf(row_max_x / BLOCK_X))));

            tiles_count += (end_tile_x - start_tile_x);
            if (null_array) continue;

            // 3.3 内层循环遍历x
            for (int tile_x = start_tile_x; tile_x < end_tile_x; ++tile_x) {
                uint64_t key = ((uint64_t)tile_y * grid.x + tile_x) << 32 | *((uint32_t*)&depth);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
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
    // 保留：计算 theta 和 eccentricity（用于 quadbox 构造）
    float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
    float theta = computeTiltAngle(cov2d);
    float eccentricity = computeEccentricity(con_o);

    // 计算判别式和阈值
    float disc = con_o.y * con_o.y - con_o.x * con_o.z;
    if (con_o.x <= 0.0f || con_o.z <= 0.0f || disc >= 0.0f) {
        return 0;
    }

    // // 检查 opacity 是否太小，如果 opacity < 1/255，直接跳过
    // if (con_o.w < 1.0f / 255.0f) {
    //     return 0;
    // }

    float t = 2.0f * logf(con_o.w * 255.0f);

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