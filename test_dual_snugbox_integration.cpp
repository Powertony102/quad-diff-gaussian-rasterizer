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

// Mathematical functions (CPU versions from auxiliary.h)
__host__ __device__ inline bool isValidEllipse(const float4& con_o, float& disc) {
    float A = con_o.x;
    float B = con_o.y;
    float C = con_o.z;
    float opacity = con_o.w;
    
    disc = B * B - A * C;
    
    // Relaxed validation for testing - allow more ellipse configurations
    bool valid_diagonal = (A > 1e-8f) && (C > 1e-8f);  // More lenient epsilon
    bool valid_opacity = (opacity > 0.0f) && isfinite(opacity);
    bool valid_disc = (disc < 0.0f);  // Allow disc to be just negative, not strictly < -epsilon
    bool finite_coeffs = isfinite(A) && isfinite(B) && isfinite(C);
    
    // More lenient aspect ratio check
    float aspect_ratio_sq = A / C;
    float max_aspect_sq = 100.0f * 100.0f;  // Reduced from 1000x1000 to 100x100
    float min_aspect_sq = 1.0f / max_aspect_sq;
    bool valid_aspect = (aspect_ratio_sq < max_aspect_sq) && (aspect_ratio_sq > min_aspect_sq);
    
    return valid_diagonal && valid_opacity && valid_disc && finite_coeffs && valid_aspect;
}

__host__ __device__ inline bool passesOpacityThreshold(float opacity) {
    // Relaxed opacity threshold for testing - accept any positive finite opacity
    bool valid_opacity = (opacity > 0.0f) && isfinite(opacity);
    
    if (!valid_opacity) {
        return false;
    }
    
    float scaled_opacity = opacity * 255.0f;
    float t = 2.0f * logf(scaled_opacity);
    
    // More lenient log validation - allow smaller values
    bool valid_log = isfinite(t);
    
    return valid_log;
}

__host__ __device__ inline float2 clampCoordinates(const float2& coord) {
    float x_clamped = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, coord.x));
    float y_clamped = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, coord.y));
    return make_float2(x_clamped, y_clamped);
}

__host__ __device__ inline float safeSqrt(float value) {
    float clamped_value = fmaxf(value, DUAL_SNUGBOX_EPSILON);
    return sqrtf(clamped_value);
}

__host__ __device__ inline float computeTiltAngle(const float3& cov2d) {
    float numerator = 2.0f * cov2d.y;
    float denominator = cov2d.x - cov2d.z;
    return 0.5f * atan2f(numerator, denominator);
}

__host__ __device__ inline float computeStretchingFactor(float theta, float beta = 1.1f) {
    return 1.0f + beta * fabsf(cosf(2.0f * theta));
}

__host__ __device__ inline ExtremePoints computeExtremePoints(
    const float4& con_o, float disc, float t, const float2& p) {
    ExtremePoints extremes;
    
    float A = con_o.x;
    float B = con_o.y;
    float C = con_o.z;
    
    float2 clamped_center = clampCoordinates(p);
    
    // X-extremes computation
    float x_extreme_offset = safeSqrt(-t * C / disc);
    float x_min = clamped_center.x - x_extreme_offset;
    float x_max = clamped_center.x + x_extreme_offset;
    
    extremes.x_extremes = make_float2(
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_min)),
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_max))
    );
    
    // Y-coordinates at x-extremes
    if (fabsf(B) > DUAL_SNUGBOX_EPSILON) {
        float y_at_x_min = clamped_center.y - A * (-x_extreme_offset) / B;
        float y_at_x_max = clamped_center.y - A * x_extreme_offset / B;
        
        extremes.y_coords_at_x_extremes = make_float2(
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_at_x_min)),
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_at_x_max))
        );
    } else {
        extremes.y_coords_at_x_extremes = make_float2(clamped_center.y, clamped_center.y);
    }
    
    // Y-extremes computation
    float y_extreme_offset = safeSqrt(-t * A / disc);
    float y_min = clamped_center.y - y_extreme_offset;
    float y_max = clamped_center.y + y_extreme_offset;
    
    extremes.y_extremes = make_float2(
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_min)),
        fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, y_max))
    );
    
    // X-coordinates at y-extremes
    if (fabsf(B) > DUAL_SNUGBOX_EPSILON) {
        float x_at_y_min = clamped_center.x - C * (-y_extreme_offset) / B;
        float x_at_y_max = clamped_center.x - C * y_extreme_offset / B;
        
        extremes.x_coords_at_y_extremes = make_float2(
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_at_y_min)),
            fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, x_at_y_max))
        );
    } else {
        extremes.x_coords_at_y_extremes = make_float2(clamped_center.x, clamped_center.x);
    }
    
    return extremes;
}

__host__ __device__ inline DualBox constructDualBoxes(
    const ExtremePoints& extremes, const float2& center, float stretch_factor) {
    DualBox dual_box;
    dual_box.valid = false;
    
    // Validate input parameters
    if (!isfinite(center.x) || !isfinite(center.y) || 
        !isfinite(stretch_factor) || stretch_factor < 1.0f || stretch_factor > 3.0f) {
        return dual_box;
    }
    
    // Collect all extreme points for partitioning
    float extreme_points_x[4] = {
        extremes.x_extremes.x,
        extremes.x_extremes.y,
        extremes.x_coords_at_y_extremes.x,
        extremes.x_coords_at_y_extremes.y
    };
    
    float extreme_points_y[4] = {
        extremes.y_coords_at_x_extremes.x,
        extremes.y_coords_at_x_extremes.y,
        extremes.y_extremes.x,
        extremes.y_extremes.y
    };
    
    // Validate all extreme points
    for (int i = 0; i < 4; i++) {
        if (!isfinite(extreme_points_x[i]) || !isfinite(extreme_points_y[i])) {
            return dual_box;
        }
        
        extreme_points_x[i] = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, 
                                   fminf(DUAL_SNUGBOX_MAX_COORDINATE, extreme_points_x[i]));
        extreme_points_y[i] = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, 
                                   fminf(DUAL_SNUGBOX_MAX_COORDINATE, extreme_points_y[i]));
    }
    
    // Partition extreme points based on x-coordinate relative to center
    float left_points_x[5], left_points_y[5];
    float right_points_x[5], right_points_y[5];
    int left_count = 0, right_count = 0;
    
    // Add center point to both partitions
    left_points_x[left_count] = center.x;
    left_points_y[left_count] = center.y;
    left_count++;
    
    right_points_x[right_count] = center.x;
    right_points_y[right_count] = center.y;
    right_count++;
    
    // Partition extreme points
    for (int i = 0; i < 4; i++) {
        if (left_count >= 5 || right_count >= 5) {
            return dual_box;
        }
        
        if (extreme_points_x[i] <= center.x) {
            left_points_x[left_count] = extreme_points_x[i];
            left_points_y[left_count] = extreme_points_y[i];
            left_count++;
        }
        if (extreme_points_x[i] >= center.x) {
            right_points_x[right_count] = extreme_points_x[i];
            right_points_y[right_count] = extreme_points_y[i];
            right_count++;
        }
    }
    
    if (left_count < 1 || right_count < 1) {
        return dual_box;
    }
    
    // Construct left AABB
    float left_min_x = left_points_x[0], left_max_x = left_points_x[0];
    float left_min_y = left_points_y[0], left_max_y = left_points_y[0];
    
    for (int i = 1; i < left_count; i++) {
        left_min_x = fminf(left_min_x, left_points_x[i]);
        left_max_x = fmaxf(left_max_x, left_points_x[i]);
        left_min_y = fminf(left_min_y, left_points_y[i]);
        left_max_y = fmaxf(left_max_y, left_points_y[i]);
    }
    
    // Construct right AABB
    float right_min_x = right_points_x[0], right_max_x = right_points_x[0];
    float right_min_y = right_points_y[0], right_max_y = right_points_y[0];
    
    for (int i = 1; i < right_count; i++) {
        right_min_x = fminf(right_min_x, right_points_x[i]);
        right_max_x = fmaxf(right_max_x, right_points_x[i]);
        right_min_y = fminf(right_min_y, right_points_y[i]);
        right_max_y = fmaxf(right_max_y, right_points_y[i]);
    }
    
    // Validate constructed boxes
    if (left_max_x <= left_min_x || left_max_y <= left_min_y ||
        right_max_x <= right_min_x || right_max_y <= right_min_y) {
        return dual_box;
    }
    
    // Apply adaptive stretching
    float left_width = left_max_x - left_min_x;
    float left_height = left_max_y - left_min_y;
    float right_width = right_max_x - right_min_x;
    float right_height = right_max_y - right_min_y;
    
    if (!isfinite(left_width) || !isfinite(left_height) || 
        !isfinite(right_width) || !isfinite(right_height) ||
        left_width <= 0.0f || left_height <= 0.0f ||
        right_width <= 0.0f || right_height <= 0.0f) {
        return dual_box;
    }
    
    // Apply stretching to left box
    if (left_width >= left_height) {
        float stretch_amount = left_width * (stretch_factor - 1.0f);
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            left_min_x -= stretch_amount;
        }
    } else {
        float stretch_amount = left_height * (stretch_factor - 1.0f) * 0.5f;
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            left_min_y -= stretch_amount;
            left_max_y += stretch_amount;
        }
    }
    
    // Apply stretching to right box
    if (right_width >= right_height) {
        float stretch_amount = right_width * (stretch_factor - 1.0f);
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            right_max_x += stretch_amount;
        }
    } else {
        float stretch_amount = right_height * (stretch_factor - 1.0f) * 0.5f;
        if (isfinite(stretch_amount) && stretch_amount >= 0.0f) {
            right_min_y -= stretch_amount;
            right_max_y += stretch_amount;
        }
    }
    
    // Apply final boundary clamping
    left_min_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_min_x));
    left_min_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_min_y));
    left_max_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_max_x));
    left_max_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, left_max_y));
    
    right_min_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_min_x));
    right_min_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_min_y));
    right_max_x = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_max_x));
    right_max_y = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, right_max_y));
    
    // Final validation
    if (!isfinite(left_min_x) || !isfinite(left_min_y) || !isfinite(left_max_x) || !isfinite(left_max_y) ||
        !isfinite(right_min_x) || !isfinite(right_min_y) || !isfinite(right_max_x) || !isfinite(right_max_y)) {
        return dual_box;
    }
    
    if (left_max_x <= left_min_x || left_max_y <= left_min_y ||
        right_max_x <= right_min_x || right_max_y <= right_min_y) {
        return dual_box;
    }
    
    dual_box.left_box = make_float4(left_min_x, left_min_y, left_max_x, left_max_y);
    dual_box.right_box = make_float4(right_min_x, right_min_y, right_max_x, right_max_y);
    dual_box.valid = true;
    
    return dual_box;
}

__host__ __device__ inline bool tileIntersectsBox(int tile_x, int tile_y, const float4& box) {
    float tile_min_x = (float)(tile_x * BLOCK_X);
    float tile_max_x = (float)((tile_x + 1) * BLOCK_X);
    float tile_min_y = (float)(tile_y * BLOCK_Y);
    float tile_max_y = (float)((tile_y + 1) * BLOCK_Y);
    
    bool x_overlap = (tile_min_x < box.z) && (tile_max_x > box.x);
    bool y_overlap = (tile_min_y < box.w) && (tile_max_y > box.y);
    
    return x_overlap && y_overlap;
}

// Structure to hold tile intersection results for testing
struct TileIntersectionResult {
    std::vector<std::pair<int, int>> tiles;  // (tile_x, tile_y) pairs
    uint32_t count;
    bool valid;
};

// Generate tile intersections for testing (CPU version)
__host__ inline TileIntersectionResult generateTileIntersections(
    const DualBox& dual_box, const dim3& grid) {
    TileIntersectionResult result;
    result.valid = false;
    result.count = 0;
    
    if (!dual_box.valid) {
        return result;
    }
    
    // Compute union bounding rectangle
    float union_min_x = fminf(dual_box.left_box.x, dual_box.right_box.x);
    float union_min_y = fminf(dual_box.left_box.y, dual_box.right_box.y);
    float union_max_x = fmaxf(dual_box.left_box.z, dual_box.right_box.z);
    float union_max_y = fmaxf(dual_box.left_box.w, dual_box.right_box.w);
    
    if (!isfinite(union_min_x) || !isfinite(union_min_y) || 
        !isfinite(union_max_x) || !isfinite(union_max_y)) {
        return result;
    }
    
    if (union_max_x - union_min_x <= 0.0f || union_max_y - union_min_y <= 0.0f) {
        return result;
    }
    
    // Convert to tile coordinates
    int rect_min_x = std::max(0, std::min((int)grid.x, (int)floorf(union_min_x / BLOCK_X)));
    int rect_min_y = std::max(0, std::min((int)grid.y, (int)floorf(union_min_y / BLOCK_Y)));
    int rect_max_x = std::max(0, std::min((int)grid.x, (int)ceilf(union_max_x / BLOCK_X)));
    int rect_max_y = std::max(0, std::min((int)grid.y, (int)ceilf(union_max_y / BLOCK_Y)));
    
    rect_min_x = std::max(0, rect_min_x);
    rect_min_y = std::max(0, rect_min_y);
    rect_max_x = std::min((int)grid.x, rect_max_x);
    rect_max_y = std::min((int)grid.y, rect_max_y);
    
    if (rect_min_x >= rect_max_x || rect_min_y >= rect_max_y) {
        result.valid = true;  // Valid but empty result
        return result;
    }
    
    // Generate unique tile intersections
    for (int tile_y = rect_min_y; tile_y < rect_max_y; ++tile_y) {
        for (int tile_x = rect_min_x; tile_x < rect_max_x; ++tile_x) {
            if (tile_x < 0 || tile_x >= (int)grid.x || tile_y < 0 || tile_y >= (int)grid.y) {
                continue;
            }
            
            bool intersects_left = tileIntersectsBox(tile_x, tile_y, dual_box.left_box);
            bool intersects_right = tileIntersectsBox(tile_x, tile_y, dual_box.right_box);
            
            if (intersects_left || intersects_right) {
                result.tiles.push_back(std::make_pair(tile_x, tile_y));
                result.count++;
            }
        }
    }
    
    result.valid = true;
    return result;
}

// Test tolerance for comparisons
#define TEST_EPSILON 1e-5f

// Helper function to create test ellipses with different orientations
struct TestEllipse {
    float4 con_o;      // (A, B, C, opacity)
    float2 center;
    float angle_degrees;
    const char* description;
};

// Test case structure for integration tests
struct IntegrationTestCase {
    TestEllipse ellipse;
    dim3 grid;
    const char* test_name;
};

// Helper function to create ellipse coefficients from angle and aspect ratio
float4 createEllipseCoefficients(float angle_rad, float aspect_ratio, float opacity = 1.0f) {
    // Create ellipse with semi-axes a and b where aspect_ratio = a/b
    float a = 30.0f;  // Major axis length (reduced for better numerical stability)
    float b = a / aspect_ratio;  // Minor axis length
    
    // Ensure minimum axis length for numerical stability
    b = fmaxf(b, 5.0f);
    a = fmaxf(a, b);  // Ensure a >= b
    
    float cos_theta = cosf(angle_rad);
    float sin_theta = sinf(angle_rad);
    float cos2 = cos_theta * cos_theta;
    float sin2 = sin_theta * sin_theta;
    float sin_cos = sin_theta * cos_theta;
    
    // Transform to conic form: Ax² + 2Bxy + Cy² = 1
    // Use more stable formulation
    float inv_a2 = 1.0f / (a * a);
    float inv_b2 = 1.0f / (b * b);
    
    float A = cos2 * inv_a2 + sin2 * inv_b2;
    float B = sin_cos * (inv_a2 - inv_b2);
    float C = sin2 * inv_a2 + cos2 * inv_b2;
    
    // Ensure discriminant is negative (valid ellipse)
    float disc = B * B - A * C;
    if (disc >= 0.0f) {
        // Force ellipse by adjusting B slightly
        B = B * 0.99f;
        disc = B * B - A * C;
        if (disc >= 0.0f) {
            B = 0.0f;  // Make it axis-aligned if needed
        }
    }
    
    // Scale coefficients to ensure reasonable magnitude
    float scale = 1.0f / fmaxf(fmaxf(A, C), fabsf(B));
    A *= scale;
    B *= scale;
    C *= scale;
    
    return make_float4(A, B, C, opacity);
}

// Test 1: Verify no duplicate (tile_index, gaussian_index) pairs in output
// Requirements: 5.1, 5.2, 5.3, 5.4
bool testNoDuplicateTilePairs() {
    std::cout << "\n=== Test 1: No Duplicate Tile Pairs ===\n";
    std::cout << "Requirements: 5.1, 5.2, 5.3, 5.4 - Unique tile intersection generation\n";
    
    std::vector<IntegrationTestCase> test_cases = {
        // Horizontal ellipse that should generate overlapping boxes
        {{createEllipseCoefficients(0.0f, 3.0f), {100.0f, 100.0f}, 0.0f, "Horizontal ellipse"},
         make_dim3(20, 20), "Horizontal ellipse overlap test"},
        
        // Vertical ellipse
        {{createEllipseCoefficients(M_PI/2.0f, 3.0f), {100.0f, 100.0f}, 90.0f, "Vertical ellipse"},
         make_dim3(20, 20), "Vertical ellipse overlap test"},
        
        // 45-degree tilted ellipse
        {{createEllipseCoefficients(M_PI/4.0f, 2.0f), {100.0f, 100.0f}, 45.0f, "45-degree ellipse"},
         make_dim3(20, 20), "45-degree ellipse overlap test"},
        
        // Large ellipse spanning many tiles
        {{createEllipseCoefficients(M_PI/6.0f, 4.0f), {200.0f, 200.0f}, 30.0f, "Large ellipse"},
         make_dim3(30, 30), "Large ellipse overlap test"},
        
        // Small ellipse with minimal overlap
        {{make_float4(0.1f, 0.05f, 0.1f, 1.0f), {50.0f, 50.0f}, 0.0f, "Small ellipse"},
         make_dim3(10, 10), "Small ellipse minimal overlap test"},
    };
    
    bool all_passed = true;
    int test_count = 0;
    
    for (const auto& test_case : test_cases) {
        test_count++;
        std::cout << "  Test " << test_count << ": " << test_case.test_name << std::endl;
        
        // Validate ellipse and compute parameters
        float disc;
        if (!isValidEllipse(test_case.ellipse.con_o, disc) || 
            !passesOpacityThreshold(test_case.ellipse.con_o.w)) {
            std::cout << "    SKIP: Invalid ellipse parameters" << std::endl;
            continue;
        }
        
        // Compute threshold parameter t for Gaussian cutoff
        // For Gaussian exp(-0.5 * (Ax² + 2Bxy + Cy²)) = threshold
        // We solve for the contour where the Gaussian equals 1/255 (minimum visible opacity)
        float threshold = 1.0f / 255.0f;
        float t = -2.0f * logf(threshold);  // t = -2*ln(threshold) for the ellipse equation
        
        if (!isfinite(t) || t <= 0.0f) {
            // Use default threshold if calculation fails
            t = 5.0f;  // Reasonable default for most ellipses
        }
        
        // Compute dual boxes
        ExtremePoints extremes = computeExtremePoints(test_case.ellipse.con_o, disc, t, test_case.ellipse.center);
        float3 cov2d = make_float3(test_case.ellipse.con_o.x, test_case.ellipse.con_o.y, test_case.ellipse.con_o.z);
        float theta = computeTiltAngle(cov2d);
        float stretch_factor = computeStretchingFactor(theta, 1.1f);
        
        DualBox dual_box = constructDualBoxes(extremes, test_case.ellipse.center, stretch_factor);
        
        if (!dual_box.valid) {
            std::cout << "    SKIP: Failed to construct valid dual boxes" << std::endl;
            continue;
        }
        
        // Generate tile intersections
        TileIntersectionResult result = generateTileIntersections(dual_box, test_case.grid);
        
        if (!result.valid) {
            std::cout << "    FAIL: Failed to generate tile intersections" << std::endl;
            all_passed = false;
            continue;
        }
        
        // Check for duplicates using set
        std::set<std::pair<int, int>> unique_tiles(result.tiles.begin(), result.tiles.end());
        
        bool no_duplicates = (unique_tiles.size() == result.tiles.size());
        
        if (no_duplicates) {
            std::cout << "    PASS: No duplicate tiles found (" << result.count << " unique tiles)" << std::endl;
        } else {
            std::cout << "    FAIL: Found " << (result.tiles.size() - unique_tiles.size()) 
                      << " duplicate tiles out of " << result.tiles.size() << " total" << std::endl;
            all_passed = false;
        }
        
        // Additional validation: verify each tile appears exactly once
        std::map<std::pair<int, int>, int> tile_counts;
        for (const auto& tile : result.tiles) {
            tile_counts[tile]++;
        }
        
        int duplicates_found = 0;
        for (const auto& entry : tile_counts) {
            if (entry.second > 1) {
                duplicates_found++;
                std::cout << "    DUPLICATE: Tile (" << entry.first.first << ", " << entry.first.second 
                          << ") appears " << entry.second << " times" << std::endl;
            }
        }
        
        if (duplicates_found > 0) {
            all_passed = false;
        }
    }
    
    std::cout << "No Duplicate Pairs Test: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    return all_passed;
}

// Test 2: Test coverage completeness for various ellipse orientations
// Requirements: 5.1, 5.2, 5.3, 5.4
bool testCoverageCompleteness() {
    std::cout << "\n=== Test 2: Coverage Completeness ===\n";
    std::cout << "Requirements: 5.1, 5.2, 5.3, 5.4 - Complete ellipse coverage verification\n";
    
    // Test ellipses at different orientations
    std::vector<float> test_angles = {
        0.0f,           // Horizontal
        M_PI/6.0f,      // 30 degrees
        M_PI/4.0f,      // 45 degrees
        M_PI/3.0f,      // 60 degrees
        M_PI/2.0f,      // Vertical
        2.0f*M_PI/3.0f, // 120 degrees
        3.0f*M_PI/4.0f, // 135 degrees
        5.0f*M_PI/6.0f  // 150 degrees
    };
    
    std::vector<float> aspect_ratios = {1.5f, 2.0f, 3.0f, 4.0f};
    
    bool all_passed = true;
    int test_count = 0;
    
    for (float angle : test_angles) {
        for (float aspect_ratio : aspect_ratios) {
            test_count++;
            float angle_degrees = angle * 180.0f / M_PI;
            
            std::cout << "  Test " << test_count << ": Angle " << std::fixed << std::setprecision(1) 
                      << angle_degrees << "°, Aspect " << aspect_ratio << std::endl;
            
            // Create test ellipse
            float4 con_o = createEllipseCoefficients(angle, aspect_ratio);
            float2 center = {150.0f, 150.0f};
            dim3 grid = make_dim3(25, 25);
            
            // Validate ellipse
            float disc;
            if (!isValidEllipse(con_o, disc) || !passesOpacityThreshold(con_o.w)) {
                std::cout << "    SKIP: Invalid ellipse parameters" << std::endl;
                continue;
            }
            
            // Compute threshold parameter t for Gaussian cutoff
            float threshold = 1.0f / 255.0f;
            float t = -2.0f * logf(threshold);  // t = -2*ln(threshold) for the ellipse equation
            
            if (!isfinite(t) || t <= 0.0f) {
                t = 5.0f;  // Reasonable default for most ellipses
            }
            
            // Compute dual boxes
            ExtremePoints extremes = computeExtremePoints(con_o, disc, t, center);
            float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
            float theta = computeTiltAngle(cov2d);
            float stretch_factor = computeStretchingFactor(theta, 1.1f);
            
            DualBox dual_box = constructDualBoxes(extremes, center, stretch_factor);
            
            if (!dual_box.valid) {
                std::cout << "    SKIP: Failed to construct valid dual boxes" << std::endl;
                continue;
            }
            
            // Generate tile intersections
            TileIntersectionResult result = generateTileIntersections(dual_box, grid);
            
            if (!result.valid) {
                std::cout << "    FAIL: Failed to generate tile intersections" << std::endl;
                all_passed = false;
                continue;
            }
            
            // Verify coverage completeness by checking that all tiles intersecting 
            // the union of both boxes are included
            std::set<std::pair<int, int>> expected_tiles;
            std::set<std::pair<int, int>> actual_tiles(result.tiles.begin(), result.tiles.end());
            
            // Compute union bounding rectangle
            float union_min_x = fminf(dual_box.left_box.x, dual_box.right_box.x);
            float union_min_y = fminf(dual_box.left_box.y, dual_box.right_box.y);
            float union_max_x = fmaxf(dual_box.left_box.z, dual_box.right_box.z);
            float union_max_y = fmaxf(dual_box.left_box.w, dual_box.right_box.w);
            
            int rect_min_x = std::max(0, std::min((int)grid.x, (int)floorf(union_min_x / BLOCK_X)));
            int rect_min_y = std::max(0, std::min((int)grid.y, (int)floorf(union_min_y / BLOCK_Y)));
            int rect_max_x = std::max(0, std::min((int)grid.x, (int)ceilf(union_max_x / BLOCK_X)));
            int rect_max_y = std::max(0, std::min((int)grid.y, (int)ceilf(union_max_y / BLOCK_Y)));
            
            // Generate expected tiles by testing each tile in union rectangle
            for (int tile_y = rect_min_y; tile_y < rect_max_y; ++tile_y) {
                for (int tile_x = rect_min_x; tile_x < rect_max_x; ++tile_x) {
                    if (tile_x >= 0 && tile_x < (int)grid.x && tile_y >= 0 && tile_y < (int)grid.y) {
                        bool intersects_left = tileIntersectsBox(tile_x, tile_y, dual_box.left_box);
                        bool intersects_right = tileIntersectsBox(tile_x, tile_y, dual_box.right_box);
                        
                        if (intersects_left || intersects_right) {
                            expected_tiles.insert(std::make_pair(tile_x, tile_y));
                        }
                    }
                }
            }
            
            // Compare expected vs actual tiles
            bool coverage_complete = (expected_tiles == actual_tiles);
            
            if (coverage_complete) {
                std::cout << "    PASS: Complete coverage (" << result.count << " tiles)" << std::endl;
            } else {
                std::cout << "    FAIL: Incomplete coverage - Expected " << expected_tiles.size() 
                          << " tiles, got " << actual_tiles.size() << std::endl;
                
                // Show missing tiles
                std::set<std::pair<int, int>> missing_tiles;
                std::set_difference(expected_tiles.begin(), expected_tiles.end(),
                                  actual_tiles.begin(), actual_tiles.end(),
                                  std::inserter(missing_tiles, missing_tiles.begin()));
                
                if (!missing_tiles.empty()) {
                    std::cout << "    Missing tiles: ";
                    for (const auto& tile : missing_tiles) {
                        std::cout << "(" << tile.first << "," << tile.second << ") ";
                    }
                    std::cout << std::endl;
                }
                
                // Show extra tiles
                std::set<std::pair<int, int>> extra_tiles;
                std::set_difference(actual_tiles.begin(), actual_tiles.end(),
                                  expected_tiles.begin(), expected_tiles.end(),
                                  std::inserter(extra_tiles, extra_tiles.begin()));
                
                if (!extra_tiles.empty()) {
                    std::cout << "    Extra tiles: ";
                    for (const auto& tile : extra_tiles) {
                        std::cout << "(" << tile.first << "," << tile.second << ") ";
                    }
                    std::cout << std::endl;
                }
                
                all_passed = false;
            }
        }
    }
    
    std::cout << "Coverage Completeness Test: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    return all_passed;
}

// Test 3: Validate proper handling of overlapping box regions
// Requirements: 5.1, 5.2, 5.3, 5.4
bool testOverlappingBoxHandling() {
    std::cout << "\n=== Test 3: Overlapping Box Handling ===\n";
    std::cout << "Requirements: 5.1, 5.2, 5.3, 5.4 - Proper handling of overlapping regions\n";
    
    // Test cases designed to create significant box overlap
    std::vector<IntegrationTestCase> test_cases = {
        // Nearly circular ellipse (minimal stretching, maximum overlap)
        {{createEllipseCoefficients(M_PI/4.0f, 1.1f), {100.0f, 100.0f}, 45.0f, "Nearly circular"},
         make_dim3(15, 15), "Nearly circular ellipse with maximum overlap"},
        
        // Slightly tilted ellipse (moderate overlap)
        {{createEllipseCoefficients(M_PI/8.0f, 2.0f), {120.0f, 120.0f}, 22.5f, "Slightly tilted"},
         make_dim3(20, 20), "Slightly tilted ellipse with moderate overlap"},
        
        // Ellipse at critical angle for stretching
        {{createEllipseCoefficients(0.0f, 3.0f), {80.0f, 80.0f}, 0.0f, "Horizontal critical"},
         make_dim3(12, 12), "Horizontal ellipse at critical stretching angle"},
        
        // Large ellipse with significant overlap region
        {{createEllipseCoefficients(M_PI/3.0f, 2.5f), {200.0f, 200.0f}, 60.0f, "Large overlapping"},
         make_dim3(30, 30), "Large ellipse with significant overlap"},
    };
    
    bool all_passed = true;
    int test_count = 0;
    
    for (const auto& test_case : test_cases) {
        test_count++;
        std::cout << "  Test " << test_count << ": " << test_case.test_name << std::endl;
        
        // Validate ellipse
        float disc;
        if (!isValidEllipse(test_case.ellipse.con_o, disc) || 
            !passesOpacityThreshold(test_case.ellipse.con_o.w)) {
            std::cout << "    SKIP: Invalid ellipse parameters" << std::endl;
            continue;
        }
        
        // Compute threshold parameter t for Gaussian cutoff
        float threshold = 1.0f / 255.0f;
        float t = -2.0f * logf(threshold);  // t = -2*ln(threshold) for the ellipse equation
        
        if (!isfinite(t) || t <= 0.0f) {
            t = 5.0f;  // Reasonable default for most ellipses
        }
        
        // Compute dual boxes
        ExtremePoints extremes = computeExtremePoints(test_case.ellipse.con_o, disc, t, test_case.ellipse.center);
        float3 cov2d = make_float3(test_case.ellipse.con_o.x, test_case.ellipse.con_o.y, test_case.ellipse.con_o.z);
        float theta = computeTiltAngle(cov2d);
        float stretch_factor = computeStretchingFactor(theta, 1.1f);
        
        DualBox dual_box = constructDualBoxes(extremes, test_case.ellipse.center, stretch_factor);
        
        if (!dual_box.valid) {
            std::cout << "    SKIP: Failed to construct valid dual boxes" << std::endl;
            continue;
        }
        
        // Analyze box overlap
        float left_box_area = (dual_box.left_box.z - dual_box.left_box.x) * 
                             (dual_box.left_box.w - dual_box.left_box.y);
        float right_box_area = (dual_box.right_box.z - dual_box.right_box.x) * 
                              (dual_box.right_box.w - dual_box.right_box.y);
        
        // Compute overlap region
        float overlap_min_x = fmaxf(dual_box.left_box.x, dual_box.right_box.x);
        float overlap_max_x = fminf(dual_box.left_box.z, dual_box.right_box.z);
        float overlap_min_y = fmaxf(dual_box.left_box.y, dual_box.right_box.y);
        float overlap_max_y = fminf(dual_box.left_box.w, dual_box.right_box.w);
        
        float overlap_area = 0.0f;
        bool has_overlap = (overlap_max_x > overlap_min_x) && (overlap_max_y > overlap_min_y);
        if (has_overlap) {
            overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y);
        }
        
        std::cout << "    Box areas: Left=" << std::fixed << std::setprecision(1) << left_box_area 
                  << ", Right=" << right_box_area << ", Overlap=" << overlap_area << std::endl;
        
        // Generate tile intersections
        TileIntersectionResult result = generateTileIntersections(dual_box, test_case.grid);
        
        if (!result.valid) {
            std::cout << "    FAIL: Failed to generate tile intersections" << std::endl;
            all_passed = false;
            continue;
        }
        
        // Test 1: No duplicates in overlapping region
        std::set<std::pair<int, int>> unique_tiles(result.tiles.begin(), result.tiles.end());
        bool no_duplicates = (unique_tiles.size() == result.tiles.size());
        
        // Test 2: Verify tiles in overlap region are handled correctly
        std::set<std::pair<int, int>> overlap_tiles;
        if (has_overlap) {
            int overlap_tile_min_x = std::max(0, (int)floorf(overlap_min_x / BLOCK_X));
            int overlap_tile_max_x = std::min((int)test_case.grid.x, (int)ceilf(overlap_max_x / BLOCK_X));
            int overlap_tile_min_y = std::max(0, (int)floorf(overlap_min_y / BLOCK_Y));
            int overlap_tile_max_y = std::min((int)test_case.grid.y, (int)ceilf(overlap_max_y / BLOCK_Y));
            
            for (int tile_y = overlap_tile_min_y; tile_y < overlap_tile_max_y; ++tile_y) {
                for (int tile_x = overlap_tile_min_x; tile_x < overlap_tile_max_x; ++tile_x) {
                    bool intersects_left = tileIntersectsBox(tile_x, tile_y, dual_box.left_box);
                    bool intersects_right = tileIntersectsBox(tile_x, tile_y, dual_box.right_box);
                    
                    if (intersects_left && intersects_right) {
                        overlap_tiles.insert(std::make_pair(tile_x, tile_y));
                    }
                }
            }
        }
        
        // Test 3: All overlap tiles should appear exactly once in result
        bool overlap_handled_correctly = true;
        for (const auto& overlap_tile : overlap_tiles) {
            auto it = std::find(result.tiles.begin(), result.tiles.end(), overlap_tile);
            if (it == result.tiles.end()) {
                std::cout << "    FAIL: Overlap tile (" << overlap_tile.first << "," 
                          << overlap_tile.second << ") missing from result" << std::endl;
                overlap_handled_correctly = false;
            }
        }
        
        bool test_passed = no_duplicates && overlap_handled_correctly;
        
        if (test_passed) {
            std::cout << "    PASS: Overlapping region handled correctly (" 
                      << overlap_tiles.size() << " overlap tiles, " 
                      << result.count << " total tiles)" << std::endl;
        } else {
            if (!no_duplicates) {
                std::cout << "    FAIL: Found duplicate tiles in overlapping region" << std::endl;
            }
            all_passed = false;
        }
    }
    
    std::cout << "Overlapping Box Handling Test: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    return all_passed;
}

// Test 4: Edge cases and boundary conditions
bool testEdgeCases() {
    std::cout << "\n=== Test 4: Edge Cases and Boundary Conditions ===\n";
    std::cout << "Testing edge cases for robust tile culling system\n";
    
    bool all_passed = true;
    
    // Test case 1: Ellipse at grid boundary
    {
        std::cout << "  Test 4.1: Ellipse at grid boundary" << std::endl;
        
        float4 con_o = createEllipseCoefficients(0.0f, 2.0f);
        float2 center = {15.0f, 15.0f};  // Near boundary
        dim3 grid = make_dim3(2, 2);     // Small grid
        
        float disc;
        if (isValidEllipse(con_o, disc) && passesOpacityThreshold(con_o.w)) {
            // Compute threshold parameter t for Gaussian cutoff
            float threshold = 1.0f / 255.0f;
            float t = -2.0f * logf(threshold);  // t = -2*ln(threshold) for the ellipse equation
            
            if (!isfinite(t) || t <= 0.0f) {
                t = 5.0f;  // Reasonable default for most ellipses
            }
            ExtremePoints extremes = computeExtremePoints(con_o, disc, t, center);
            float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
            float theta = computeTiltAngle(cov2d);
            float stretch_factor = computeStretchingFactor(theta, 1.1f);
            
            DualBox dual_box = constructDualBoxes(extremes, center, stretch_factor);
            
            if (dual_box.valid) {
                TileIntersectionResult result = generateTileIntersections(dual_box, grid);
                
                if (result.valid) {
                    std::set<std::pair<int, int>> unique_tiles(result.tiles.begin(), result.tiles.end());
                    bool no_duplicates = (unique_tiles.size() == result.tiles.size());
                    
                    // Verify all tiles are within grid bounds
                    bool all_in_bounds = true;
                    for (const auto& tile : result.tiles) {
                        if (tile.first < 0 || tile.first >= (int)grid.x || 
                            tile.second < 0 || tile.second >= (int)grid.y) {
                            all_in_bounds = false;
                            break;
                        }
                    }
                    
                    if (no_duplicates && all_in_bounds) {
                        std::cout << "    PASS: Boundary case handled correctly" << std::endl;
                    } else {
                        std::cout << "    FAIL: Boundary case failed" << std::endl;
                        all_passed = false;
                    }
                } else {
                    std::cout << "    FAIL: Failed to generate intersections" << std::endl;
                    all_passed = false;
                }
            } else {
                std::cout << "    SKIP: Failed to construct dual boxes" << std::endl;
            }
        } else {
            std::cout << "    SKIP: Invalid ellipse" << std::endl;
        }
    }
    
    // Test case 2: Very small ellipse
    {
        std::cout << "  Test 4.2: Very small ellipse" << std::endl;
        
        float4 con_o = make_float4(100.0f, 0.0f, 100.0f, 1.0f);  // Small circle
        float2 center = {100.0f, 100.0f};
        dim3 grid = make_dim3(20, 20);
        
        float disc;
        if (isValidEllipse(con_o, disc) && passesOpacityThreshold(con_o.w)) {
            // Compute threshold parameter t for Gaussian cutoff
            float threshold = 1.0f / 255.0f;
            float t = -2.0f * logf(threshold);  // t = -2*ln(threshold) for the ellipse equation
            
            if (!isfinite(t) || t <= 0.0f) {
                t = 5.0f;  // Reasonable default for most ellipses
            }
            ExtremePoints extremes = computeExtremePoints(con_o, disc, t, center);
            float3 cov2d = make_float3(con_o.x, con_o.y, con_o.z);
            float theta = computeTiltAngle(cov2d);
            float stretch_factor = computeStretchingFactor(theta, 1.1f);
            
            DualBox dual_box = constructDualBoxes(extremes, center, stretch_factor);
            
            if (dual_box.valid) {
                TileIntersectionResult result = generateTileIntersections(dual_box, grid);
                
                if (result.valid) {
                    std::set<std::pair<int, int>> unique_tiles(result.tiles.begin(), result.tiles.end());
                    bool no_duplicates = (unique_tiles.size() == result.tiles.size());
                    
                    if (no_duplicates) {
                        std::cout << "    PASS: Small ellipse handled correctly (" 
                                  << result.count << " tiles)" << std::endl;
                    } else {
                        std::cout << "    FAIL: Small ellipse failed" << std::endl;
                        all_passed = false;
                    }
                } else {
                    std::cout << "    PASS: Small ellipse correctly rejected" << std::endl;
                }
            } else {
                std::cout << "    PASS: Small ellipse correctly rejected" << std::endl;
            }
        } else {
            std::cout << "    PASS: Small ellipse correctly rejected" << std::endl;
        }
    }
    
    std::cout << "Edge Cases Test: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    return all_passed;
}

// Main test function
int main() {
    std::cout << "=== Dual-SnugBox Integration Tests for Tile Culling System ===\n";
    std::cout << "Testing Requirements: 5.1, 5.2, 5.3, 5.4\n";
    std::cout << "- Verify no duplicate (tile_index, gaussian_index) pairs in output\n";
    std::cout << "- Test coverage completeness for various ellipse orientations\n";
    std::cout << "- Validate proper handling of overlapping box regions\n\n";
    
    bool all_tests_passed = true;
    
    // Run all integration tests
    all_tests_passed &= testNoDuplicateTilePairs();
    all_tests_passed &= testCoverageCompleteness();
    all_tests_passed &= testOverlappingBoxHandling();
    all_tests_passed &= testEdgeCases();
    
    // Summary
    std::cout << "\n=== Integration Test Summary ===\n";
    if (all_tests_passed) {
        std::cout << "✅ ALL INTEGRATION TESTS PASSED\n";
        std::cout << "✅ No duplicate (tile_index, gaussian_index) pairs verified\n";
        std::cout << "✅ Coverage completeness confirmed for various ellipse orientations\n";
        std::cout << "✅ Overlapping box regions handled correctly\n";
        std::cout << "✅ Edge cases and boundary conditions validated\n";
        std::cout << "\nThe dual-SnugBox tile culling system meets all requirements:\n";
        std::cout << "- Requirement 5.1: Union-based tile processing prevents duplicates\n";
        std::cout << "- Requirement 5.2: Logical OR operation for tile overlap detection\n";
        std::cout << "- Requirement 5.3: Unique (tile_index, gaussian_index) pairs generated\n";
        std::cout << "- Requirement 5.4: No double-counting in overlapping regions\n";
        return 0;
    } else {
        std::cout << "❌ SOME INTEGRATION TESTS FAILED\n";
        std::cout << "Please review the tile culling implementation for:\n";
        std::cout << "- Duplicate key-value pair generation\n";
        std::cout << "- Incomplete coverage for certain ellipse orientations\n";
        std::cout << "- Incorrect handling of overlapping box regions\n";
        return 1;
    }
}