
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

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)
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

struct QuadBox {
    float4 left_box; 
    float4 left_small_box;  
    float4 right_box;  
    float4 right_small_box;
    float2 center; 
    bool valid;        
};

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
    
    float snug_min_x = fminf(fminf(quad_box.left_box.x, quad_box.left_small_box.x), 
                            fminf(quad_box.right_box.x, quad_box.right_small_box.x));
    float snug_max_x = fmaxf(fmaxf(quad_box.left_box.z, quad_box.left_small_box.z), 
                            fmaxf(quad_box.right_box.z, quad_box.right_small_box.z));
    float snug_min_y = fminf(fminf(quad_box.left_box.y, quad_box.left_small_box.y), 
                            fminf(quad_box.right_box.y, quad_box.right_small_box.y));
    float snug_max_y = fmaxf(fmaxf(quad_box.left_box.w, quad_box.left_small_box.w), 
                            fmaxf(quad_box.right_box.w, quad_box.right_small_box.w));
    
    int snug_min_tile_x = max(0, min((int)grid.x, (int)floorf(snug_min_x / BLOCK_X)));
    int snug_max_tile_x = max(0, min((int)grid.x, (int)ceilf(snug_max_x / BLOCK_X)));
    int snug_min_tile_y = max(0, min((int)grid.y, (int)floorf(snug_min_y / BLOCK_Y)));
    int snug_max_tile_y = max(0, min((int)grid.y, (int)ceilf(snug_max_y / BLOCK_Y)));
    
    int rect_a = snug_max_tile_x - snug_min_tile_x;  
    int rect_b = snug_max_tile_y - snug_min_tile_y;  
    
    bool x_is_short = (rect_a <= rect_b);  

    bool null_array = (gaussian_keys_unsorted == nullptr && gaussian_values_unsorted == nullptr);
    
    if (x_is_short) {
        for (int tile_x = snug_min_tile_x; tile_x < snug_max_tile_x; ++tile_x) {
            
            float tile_x_min = tile_x * BLOCK_X;
            float tile_x_max = (tile_x + 1) * BLOCK_X;
            
            float col_min_y = 1e9f, col_max_y = -1e9f;
            bool has_coverage = false;
            
            float4 boxes[4] = {quad_box.left_box, quad_box.left_small_box, 
                              quad_box.right_box, quad_box.right_small_box};
            
            for (int b = 0; b < 4; ++b) {
                if (!(tile_x_max <= boxes[b].x || tile_x_min >= boxes[b].z)) {
                    col_min_y = fminf(col_min_y, boxes[b].y);
                    col_max_y = fmaxf(col_max_y, boxes[b].w);
                    has_coverage = true;
                }
            }
            
            if (!has_coverage) continue;  
            
            int start_tile_y = max(snug_min_tile_y, max(0, min((int)grid.y, (int)floorf(col_min_y / BLOCK_Y))));
            int end_tile_y = min(snug_max_tile_y, max(0, min((int)grid.y, (int)ceilf(col_max_y / BLOCK_Y))));
            
            tiles_count += (end_tile_y - start_tile_y);
            if (null_array) continue;

            for (int tile_y = start_tile_y; tile_y < end_tile_y; ++tile_y) {
                uint64_t key = ((uint64_t)tile_y * grid.x + tile_x) << 32 | *((uint32_t*)&depth);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    } else {
        for (int tile_y = snug_min_tile_y; tile_y < snug_max_tile_y; ++tile_y) {
            
            float tile_y_min = tile_y * BLOCK_Y;
            float tile_y_max = (tile_y + 1) * BLOCK_Y;
            
            float row_min_x = 1e9f, row_max_x = -1e9f;
            bool has_coverage = false;
            
            float4 boxes[4] = {quad_box.left_box, quad_box.left_small_box, 
                              quad_box.right_box, quad_box.right_small_box};
            
            for (int b = 0; b < 4; ++b) {
                if (!(tile_y_max <= boxes[b].y || tile_y_min >= boxes[b].w)) {
                    row_min_x = fminf(row_min_x, boxes[b].x);
                    row_max_x = fmaxf(row_max_x, boxes[b].z);
                    has_coverage = true;
                }
            }
            
            if (!has_coverage) continue;  
            
            int start_tile_x = max(snug_min_tile_x, max(0, min((int)grid.x, (int)floorf(row_min_x / BLOCK_X))));
            int end_tile_x = min(snug_max_tile_x, max(0, min((int)grid.x, (int)ceilf(row_max_x / BLOCK_X))));

            tiles_count += (end_tile_x - start_tile_x);
            if (null_array) continue;

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

__device__ inline QuadBox constructQuadBoxes(
    const float4& con_o,
    const float disc,
    const float t,
    const float2& center
) {
    QuadBox quad_box;
    quad_box.valid = false;

    const float a = con_o.x;
    const float b = con_o.y;
    const float c = con_o.z;

    const float det = fmaf(a, c, -b * b);
    
    if (a <= 0.0f || c <= 0.0f || det <= 0.0f || t <= 0.0f) {
        return quad_box; 
    }

    const float x_extent = sqrtf(fmaxf(0.0f, t * c / det));
    const float y_extent = sqrtf(fmaxf(0.0f, t * a / det));

    float denom_ac = fmaxf(1e-30f, a * c); 
    float raw_f = sqrtf(fmaxf(0.0f, det / denom_ac));
    const float stretch_factor = fminf(fmaxf(raw_f, 1e-6f), 1.0f);

    const float snug_min_x = center.x - x_extent;
    const float snug_max_x = center.x + x_extent;
    const float snug_min_y = center.y - y_extent;
    const float snug_max_y = center.y + y_extent;

    float left_rect_x, left_rect_y, left_rect_width, left_rect_height;
    float right_rect_x, right_rect_y, right_rect_width, right_rect_height;
    float left_small_rect_x, left_small_rect_y, left_small_width, left_small_height;
    float right_small_rect_x, right_small_rect_y, right_small_width, right_small_height;

    bool isQ1Q3 = 1;
    if (std::abs(b) < 1e-12) {
        if (a <= c) isQ1Q3 = 1; 
        else if (a > c) isQ1Q3 = 0; 
    }
    else  isQ1Q3 = (b < 0 ? 1 : 0);
    

    if ( isQ1Q3 ) 
    {
        left_rect_x = snug_min_x;
        left_rect_y = snug_min_y;
        left_rect_width = center.x - snug_min_x;
        left_rect_height = center.y - snug_min_y;

        right_rect_x = center.x;
        right_rect_y = center.y;
        right_rect_width = snug_max_x - center.x;
        right_rect_height = snug_max_y - center.y;

        left_small_width = left_rect_width * stretch_factor;
        left_small_height = left_rect_height * stretch_factor;
        left_small_rect_x = center.x - left_small_width;
        left_small_rect_y = center.y;

        right_small_width = left_small_width;  
        right_small_height = left_small_height;
        right_small_rect_x = center.x;
        right_small_rect_y = center.y - right_small_height;
    }
    else 
    {
        left_rect_x = snug_min_x;
        left_rect_y = center.y;
        left_rect_width = center.x - snug_min_x;
        left_rect_height = snug_max_y - center.y;

        right_rect_x = center.x;
        right_rect_y = snug_min_y;
        right_rect_width = snug_max_x - center.x;
        right_rect_height = center.y - snug_min_y;

        left_small_width = left_rect_width * stretch_factor;
        left_small_height = left_rect_height * stretch_factor;
        left_small_rect_x = center.x - left_small_width;
        left_small_rect_y = center.y - left_small_height;

        right_small_width = left_small_width;  
        right_small_height = left_small_height;
        right_small_rect_x = center.x;
        right_small_rect_y = center.y;
    }
    
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

__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted
    )
{
    float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
    
    const float a = con_o.x, b = con_o.y, c = con_o.z;
    float disc = fmaf(b, b, -a * c);
    if (a <= 0.0f || c <= 0.0f || disc >= 0.0f) {
        return 0;
    }

    float t = 2.0f * logf(con_o.w * 255.0f);
    if (!(t > 0.0f)) {
        return 0;
    }

    QuadBox quad_box = constructQuadBoxes(con_o, disc, t, p);

    if (!quad_box.valid) {
        return 0;
    }

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