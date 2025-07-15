/*
 * Dual-SnugBox Debugging and Validation Capabilities
 * Requirements: 3.4, 4.3 - debugging and performance validation
 */

#ifndef DUAL_SNUGBOX_DEBUG_H_INCLUDED
#define DUAL_SNUGBOX_DEBUG_H_INCLUDED

#include "config.h"
#include "stdio.h"

// Debug mode configuration
#ifndef DUAL_SNUGBOX_DEBUG_MODE
#define DUAL_SNUGBOX_DEBUG_MODE 0  // Set to 1 to enable debugging
#endif

// Performance counter structure for tile count comparisons
struct DualSnugBoxPerformanceCounters {
    uint32_t total_gaussians_processed;
    uint32_t valid_gaussians_processed;
    uint32_t degenerate_ellipses_detected;
    uint32_t numerical_instabilities_detected;
    uint32_t total_tiles_generated;
    uint32_t max_tiles_per_gaussian;
    uint32_t min_tiles_per_gaussian;
    float average_tiles_per_gaussian;
    uint32_t stretching_factor_applications;
    uint32_t boundary_clamps_applied;
};

// Global performance counters (device memory)
__device__ DualSnugBoxPerformanceCounters g_dual_snugbox_counters;

// Debug logging structure for box construction details
struct DualSnugBoxDebugInfo {
    uint32_t gaussian_idx;
    float2 center;
    float4 conic_coeffs;
    float discriminant;
    float threshold;
    float tilt_angle;
    float stretching_factor;
    ExtremePoints extreme_points;
    DualBox constructed_boxes;
    uint32_t tiles_touched;
    bool validation_passed;
    char error_message[128];
};

// Visual debugging output structure for tile coverage verification
struct TileCoverageDebugInfo {
    uint32_t tile_x;
    uint32_t tile_y;
    bool intersects_left_box;
    bool intersects_right_box;
    bool intersects_union;
    float4 left_box_bounds;
    float4 right_box_bounds;
    uint32_t gaussian_idx;
};

// Initialize performance counters
__device__ inline void initializePerformanceCounters() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_dual_snugbox_counters.total_gaussians_processed = 0;
        g_dual_snugbox_counters.valid_gaussians_processed = 0;
        g_dual_snugbox_counters.degenerate_ellipses_detected = 0;
        g_dual_snugbox_counters.numerical_instabilities_detected = 0;
        g_dual_snugbox_counters.total_tiles_generated = 0;
        g_dual_snugbox_counters.max_tiles_per_gaussian = 0;
        g_dual_snugbox_counters.min_tiles_per_gaussian = UINT32_MAX;
        g_dual_snugbox_counters.average_tiles_per_gaussian = 0.0f;
        g_dual_snugbox_counters.stretching_factor_applications = 0;
        g_dual_snugbox_counters.boundary_clamps_applied = 0;
    }
}

// Update performance counters atomically
__device__ inline void updatePerformanceCounters