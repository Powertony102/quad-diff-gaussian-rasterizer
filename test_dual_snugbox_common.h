#ifndef TEST_DUAL_SNUGBOX_COMMON_H
#define TEST_DUAL_SNUGBOX_COMMON_H

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <set>
#include <map>
#include <iomanip>
#include <algorithm>

// CPU-compatible versions of CUDA functions for testing
#define __host__
#define __device__
#define __forceinline__ inline

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// CPU versions of CUDA math functions
using std::atan2f;
using std::cosf;
using std::fabsf;
using std::logf;
using std::sqrtf;
using std::fminf;
using std::fmaxf;
using std::floorf;
using std::ceilf;
using std::isfinite;

// CUDA vector types for CPU
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct dim3 { unsigned int x, y, z; };

inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline dim3 make_dim3(unsigned int x, unsigned int y, unsigned int z = 1) { return {x, y, z}; }

// Constants from auxiliary.h
#define DUAL_SNUGBOX_EPSILON 1e-6f
#define DUAL_SNUGBOX_MAX_ASPECT_RATIO 1000.0f
#define DUAL_SNUGBOX_MIN_OPACITY_THRESHOLD (1.0f / 255.0f)
#define DUAL_SNUGBOX_MAX_COORDINATE 1e6f
#define BLOCK_X 16
#define BLOCK_Y 16

// Structures from auxiliary.h
struct ExtremePoints {
    float2 x_extremes;  // (x_min, x_max)
    float2 y_extremes;  // (y_min, y_max)
    float2 x_coords_at_y_extremes;  // (x1, x2) at (y_min, y_max)
    float2 y_coords_at_x_extremes;  // (y1, y2) at (x_min, x_max)
};

struct DualBox {
    float4 left_box;   // (min_x, min_y, max_x, max_y)
    float4 right_box;  // (min_x, min_y, max_x, max_y)
    bool valid;        // Whether boxes are valid
};

// Test ellipse structure
struct TestEllipse {
    float4 con_o;      // (A, B, C, opacity)
    float2 center;
    float angle_degrees;
    const char* description;
};

// Structure to hold tile intersection results for testing
struct TileIntersectionResult {
    std::vector<std::pair<int, int>> tiles;  // (tile_x, tile_y) pairs
    uint32_t count;
    bool valid;
};

// Mathematical functions (CPU versions from auxiliary.h)
__host__ __device__ inline bool isValidEllipse(const float4& con_o, float& disc);
__host__ __device__ inline bool passesOpacityThreshold(float opacity);
__host__ __device__ inline float2 clampCoordinates(const float2& coord);
__host__ __device__ inline float safeSqrt(float value);
__host__ __device__ inline float computeTiltAngle(const float3& cov2d);
__host__ __device__ inline float computeStretchingFactor(float theta, float beta = 1.1f);
__host__ __device__ inline ExtremePoints computeExtremePoints(
    const float4& con_o, float disc, float t, const float2& p);
__host__ __device__ inline DualBox constructDualBoxes(
    const ExtremePoints& extremes, const float2& center, float stretch_factor);
__host__ __device__ inline bool tileIntersectsBox(int tile_x, int tile_y, const float4& box);
__host__ inline TileIntersectionResult generateTileIntersections(
    const DualBox& dual_box, const dim3& grid);

// Helper function to create ellipse coefficients from angle and aspect ratio
float4 createEllipseCoefficients(float angle_rad, float aspect_ratio, float opacity = 1.0f);

#endif // TEST_DUAL_SNUGBOX_COMMON_H