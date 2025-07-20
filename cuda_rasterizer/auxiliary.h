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

// Compute bounding rectangle of ellipse using geometric approach
// This matches the original test.py method for consistency
__device__ inline ExtremePoints computeBoundingRectangle(
    const float2& center,
    float a, float b, float theta_rad
) {
    ExtremePoints ext;
    
    // Calculate ellipse boundaries after rotation
    float cos_theta = cosf(theta_rad);
    float sin_theta = sinf(theta_rad);
    
    // Ellipse boundary calculation
    float dx = sqrtf((a * cos_theta) * (a * cos_theta) + (b * sin_theta) * (b * sin_theta));
    float dy = sqrtf((a * sin_theta) * (a * sin_theta) + (b * cos_theta) * (b * cos_theta));
    
    float x_min = center.x - dx;
    float x_max = center.x + dx;
    float y_min = center.y - dy;
    float y_max = center.y + dy;
    
    ext.x_extremes = make_float2(x_min, x_max);
    ext.y_extremes = make_float2(y_min, y_max);
    ext.x_coords_at_y_extremes = make_float2(x_min, x_max);
    ext.y_coords_at_x_extremes = make_float2(y_min, y_max);
    
    return ext;
}

// ---------------------------------------------------------------------------
// Compute extreme points of ellipse using analytical, numerically-stable math
// Ellipse:  A*(x-p.x)^2 + 2B*(x-p.x)*(y-p.y) + C*(y-p.y)^2 = t  (t>0)
// ---------------------------------------------------------------------------
__device__ inline ExtremePoints computeExtremePoints(
    const float4& con_o,   // (A, B, C, opacity)
    float disc,            // discriminant (B^2 - A*C)  < 0 for ellipse
    float t,               // threshold on quadratic form (>0)
    const float2& p)       // ellipse centre in image space
{
    ExtremePoints ext;

    // -----------------------------------------------------------------------
    // 1. Basic validation & safe defaults
    // -----------------------------------------------------------------------
    const float A = con_o.x;
    const float B = con_o.y;
    const float C = con_o.z;

    // Initialise with centre to avoid uninitialised reads if ellipse invalid
    ext.x_extremes              = make_float2(p.x, p.x);
    ext.y_extremes              = make_float2(p.y, p.y);
    ext.x_coords_at_y_extremes  = make_float2(p.x, p.x);
    ext.y_coords_at_x_extremes  = make_float2(p.y, p.y);

    // Quick reject: non-positive A/C, non-elliptic discriminant, non-positive t
    if (A <= 0.0f || C <= 0.0f || disc >= 0.0f || t <= 0.0f)
        return ext;

    // -----------------------------------------------------------------------
    // 2. Pre-compute safe denominators
    //     denomX = A − B²/C , denomY = C − B²/A   (both must be > 0 for ellipse)
    // -----------------------------------------------------------------------
    const float denomX = A - (B * B) / C;
    const float denomY = C - (B * B) / A;

    if (denomX <= 0.0f || denomY <= 0.0f)   // degeneration guard
        return ext;

    // -----------------------------------------------------------------------
    // 3. Compute u_max (Δx)  &  v_max (Δy) in local (u,v) space
    // -----------------------------------------------------------------------
    const float u_max = sqrtf(t / denomX);        // ≥ 0
    const float v_max = sqrtf(t / denomY);        // ≥ 0

    // Helper ratios (may be zero if B == 0, that is fine)
    const float B_over_C = B / C;
    const float B_over_A = B / A;

    // -----------------------------------------------------------------------
    // 4. Assemble extreme coordinates (global space)
    // -----------------------------------------------------------------------
    // -- x-direction extremes --
    const float y_at_xmin = p.y +  B_over_C * u_max;   // corresponds to u = -u_max
    const float y_at_xmax = p.y -  B_over_C * u_max;   // corresponds to u = +u_max

    ext.x_extremes             = make_float2(p.x - u_max, p.x + u_max);
    ext.y_coords_at_x_extremes = make_float2(y_at_xmin,   y_at_xmax);

    // -- y-direction extremes --
    const float x_at_ymin = p.x +  B_over_A * v_max;   // corresponds to v = -v_max
    const float x_at_ymax = p.x -  B_over_A * v_max;   // corresponds to v = +v_max

    ext.y_extremes             = make_float2(p.y - v_max, p.y + v_max);
    ext.x_coords_at_y_extremes = make_float2(x_at_ymin,   x_at_ymax);

    return ext;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#define M_PI_2 1.5707963267948966f
#endif

// Construct dual asymmetric AABBs using extreme points and center with enhanced error handling
// Implements left-right partitioning based on Gaussian center x-coordinate
// Requirements: 1.2, 1.3, 2.3, 2.4, 2.5, 6.2, 6.3 - dual-box construction with error handling
__device__ inline DualBox constructDualBoxes(
    const ExtremePoints& extremes,
    const float2& center,
    float theta, // tilt angle in radians
    float eccentricity
) {
    DualBox dual_box;
    dual_box.valid = false;
    
    // Original snugbox boundaries for clamping
    const float snug_min_x = extremes.x_extremes.x;
    const float snug_max_x = extremes.x_extremes.y;
    const float snug_min_y = extremes.y_extremes.x;
    const float snug_max_y = extremes.y_extremes.y;

    // Calculate extension coefficient f(e,theta)
    float e_sq = eccentricity * eccentricity;
    if (e_sq >= 1.0f) e_sq = 0.999f; // prevent division by zero
    float sin_2theta = sinf(2.0f * theta);
    float sin_2theta_sq = sin_2theta * sin_2theta;
    float stretch_factor = 1.0f / sqrtf(1.0f + (e_sq * e_sq / (4.0f * (1.0f - e_sq))) * sin_2theta_sq);

    float left_rect_x, left_rect_y, left_rect_width, left_rect_height;
    float right_rect_x, right_rect_y, right_rect_width, right_rect_height;

    if (theta >= 0 && theta <= M_PI_2) // 0到90度对应0到π/2弧度
    {
        left_rect_x = snug_min_x;
        left_rect_y = snug_min_y;
        left_rect_width = center.x - snug_min_x;
        left_rect_height = center.y - snug_min_y;

        right_rect_x = center.x;
        right_rect_y = center.y;
        right_rect_width = snug_max_x - center.x;
        right_rect_height = snug_max_y - center.y;
    }
    else // Corresponds to Python's theta > 90
    {
        left_rect_x = snug_min_x;
        left_rect_y = center.y;
        left_rect_width = center.x - snug_min_x;
        left_rect_height = snug_max_y - center.y;

        right_rect_x = center.x;
        right_rect_y = snug_min_y;
        right_rect_width = snug_max_x - center.x;
        right_rect_height = center.y - snug_min_y;
    }
    
    // Extend rectangles
    if (left_rect_width > 0)
    {
        float left_extension = left_rect_width * stretch_factor;
        left_rect_width += left_extension;
    }

    if (right_rect_width > 0)
    {
        float right_extension = right_rect_width * stretch_factor;
        right_rect_x -= right_extension;
        right_rect_width += right_extension;
    }
    
    // Store the constructed dual boxes
    dual_box.left_box = make_float4(left_rect_x, left_rect_y, left_rect_x + left_rect_width, left_rect_y + left_rect_height);
    dual_box.right_box = make_float4(right_rect_x, right_rect_y, right_rect_x + right_rect_width, right_rect_y + right_rect_height);
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

    // Compute union bounding rectangle of both boxes with validation
    float union_min_x = fminf(dual_box.left_box.x, dual_box.right_box.x);
    float union_min_y = fminf(dual_box.left_box.y, dual_box.right_box.y);
    float union_max_x = fmaxf(dual_box.left_box.z, dual_box.right_box.z);
    float union_max_y = fmaxf(dual_box.left_box.w, dual_box.right_box.w);
    
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

__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{
    float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
    float theta = computeTiltAngle(cov2d);
    float eccentricity = computeEccentricity(con_o);
    
    // Extract ellipse parameters a and b from con_o
    // con_o represents the inverse covariance matrix: [[A, B], [B, C]]
    // We need to compute the original covariance matrix and extract a, b
    float A = con_o.x;
    float B = con_o.y;
    float C = con_o.z;
    
    // Compute determinant of the inverse covariance matrix
    float det_inv = A * C - B * B;
    
    // The original covariance matrix is the inverse of [[A, B], [B, C]]
    // For a 2x2 matrix [[a11, a12], [a21, a22]], the inverse is:
    // [[a22, -a12], [-a21, a11]] / det
    float cov_xx = C / det_inv;
    float cov_xy = -B / det_inv;
    float cov_yy = A / det_inv;
    
    // Extract a and b from the covariance matrix
    // The covariance matrix represents the ellipse parameters after rotation
    // We need to compute the principal axes a and b
    float trace = cov_xx + cov_yy;
    float det = cov_xx * cov_yy - cov_xy * cov_xy;
    
    // Eigenvalues are (trace ± sqrt(trace² - 4*det)) / 2
    float discriminant = trace * trace - 4.0f * det;
    float sqrt_disc = sqrtf(discriminant);
    
    float lambda_max = (trace + sqrt_disc) / 2.0f;
    float lambda_min = (trace - sqrt_disc) / 2.0f;
    
    // a and b are the square roots of the eigenvalues
    float a = sqrtf(lambda_max);
    float b = sqrtf(lambda_min);
    
    // Use bounding rectangle instead of ellipse equation extremes
    ExtremePoints extremes = computeBoundingRectangle(p, a, b, theta);
    
    DualBox dual_box = constructDualBoxes(extremes, p, theta, eccentricity);
    
    // Check for reasonable box sizes to prevent excessive tile generation
    // float left_box_area = (dual_box.left_box.z - dual_box.left_box.x) * 
    //                      (dual_box.left_box.w - dual_box.left_box.y);
    // float right_box_area = (dual_box.right_box.z - dual_box.right_box.x) * 
    //                       (dual_box.right_box.w - dual_box.right_box.y);
    
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