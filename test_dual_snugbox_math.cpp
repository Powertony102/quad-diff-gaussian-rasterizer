#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <iomanip>

// CPU-compatible versions of CUDA functions for testing
#define __host__
#define __device__
#define __forceinline__ inline

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// CPU versions of CUDA math functions (use std:: versions directly)
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

inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

// Constants and structures from auxiliary.h
#define DUAL_SNUGBOX_EPSILON 1e-6f
#define DUAL_SNUGBOX_MAX_ASPECT_RATIO 1000.0f
#define DUAL_SNUGBOX_MIN_OPACITY_THRESHOLD (1.0f / 255.0f)
#define DUAL_SNUGBOX_MAX_COORDINATE 1e6f

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

// Mathematical functions to test (CPU versions)
__host__ __device__ inline float computeTiltAngle(const float3& cov2d) {
    float numerator = 2.0f * cov2d.y;
    float denominator = cov2d.x - cov2d.z;
    return 0.5f * atan2f(numerator, denominator);
}

__host__ __device__ inline float computeStretchingFactor(float theta, float beta = 1.1f) {
    return 1.0f + beta * fabsf(cosf(2.0f * theta));
}

__host__ __device__ inline float safeSqrt(float value) {
    float clamped_value = fmaxf(value, DUAL_SNUGBOX_EPSILON);
    return sqrtf(clamped_value);
}

__host__ __device__ inline float2 clampCoordinates(const float2& coord) {
    float x_clamped = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, coord.x));
    float y_clamped = fmaxf(-DUAL_SNUGBOX_MAX_COORDINATE, fminf(DUAL_SNUGBOX_MAX_COORDINATE, coord.y));
    return make_float2(x_clamped, y_clamped);
}

__host__ __device__ inline ExtremePoints computeExtremePoints(
    const float4& con_o,  // (A, B, C, opacity) conic coefficients
    float disc,           // discriminant B² - AC (should be negative for valid ellipse)
    float t,              // threshold constant
    const float2& p       // ellipse center
) {
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

// Test tolerance for floating point comparisons
#define TEST_EPSILON 1e-5f
#define TEST_ANGLE_EPSILON 1e-4f

// Helper function to compare floating point values
__host__ __device__ bool isClose(float a, float b, float epsilon = TEST_EPSILON) {
    return fabsf(a - b) < epsilon;
}

// Helper function to compare float2 values
__host__ __device__ bool isClose(const float2& a, const float2& b, float epsilon = TEST_EPSILON) {
    return isClose(a.x, b.x, epsilon) && isClose(a.y, b.y, epsilon);
}

// Test structure to hold test cases
struct TiltAngleTestCase {
    float3 cov2d;      // (σ_xx, σ_xy, σ_yy)
    float expected_angle;
    const char* description;
};

struct StretchingFactorTestCase {
    float theta;
    float beta;
    float expected_factor;
    const char* description;
};

struct ExtremePointsTestCase {
    float4 con_o;      // (A, B, C, opacity)
    float2 center;
    float expected_x_min, expected_x_max;
    float expected_y_min, expected_y_max;
    const char* description;
};

// CPU function to test tilt angle computation
void testTiltAngleFunction(TiltAngleTestCase* test_cases, int num_cases, bool* results) {
    for (int idx = 0; idx < num_cases; idx++) {
        TiltAngleTestCase& test = test_cases[idx];
        float computed_angle = computeTiltAngle(test.cov2d);
        
        // Handle angle wrapping - angles can differ by multiples of π
        float diff = fabsf(computed_angle - test.expected_angle);
        float wrapped_diff = fabsf(diff - M_PI);
        float min_diff = fminf(diff, wrapped_diff);
        
        results[idx] = (min_diff < TEST_ANGLE_EPSILON);
        
        if (!results[idx]) {
            std::cout << "FAIL: " << test.description << " - Expected: " << test.expected_angle 
                      << ", Got: " << computed_angle << ", Diff: " << min_diff << std::endl;
        }
    }
}

// CPU function to test stretching factor computation
void testStretchingFactorFunction(StretchingFactorTestCase* test_cases, int num_cases, bool* results) {
    for (int idx = 0; idx < num_cases; idx++) {
        StretchingFactorTestCase& test = test_cases[idx];
        float computed_factor = computeStretchingFactor(test.theta, test.beta);
        
        results[idx] = isClose(computed_factor, test.expected_factor);
        
        if (!results[idx]) {
            std::cout << "FAIL: " << test.description << " - Expected: " << test.expected_factor 
                      << ", Got: " << computed_factor << ", Diff: " 
                      << fabsf(computed_factor - test.expected_factor) << std::endl;
        }
    }
}

// CPU function to test extreme points computation
void testExtremePointsFunction(ExtremePointsTestCase* test_cases, int num_cases, bool* results) {
    for (int idx = 0; idx < num_cases; idx++) {
        ExtremePointsTestCase& test = test_cases[idx];
        
        // Compute discriminant and threshold
        float A = test.con_o.x;
        float B = test.con_o.y;
        float C = test.con_o.z;
        float opacity = test.con_o.w;
        
        float disc = B * B - A * C;
        float t = 2.0f * logf(opacity * 255.0f);
        
        // Only test if ellipse is valid
        if (A > 0 && C > 0 && disc < 0 && t > 0) {
            ExtremePoints extremes = computeExtremePoints(test.con_o, disc, t, test.center);
            
            bool x_extremes_ok = isClose(extremes.x_extremes.x, test.expected_x_min) && 
                                isClose(extremes.x_extremes.y, test.expected_x_max);
            bool y_extremes_ok = isClose(extremes.y_extremes.x, test.expected_y_min) && 
                                isClose(extremes.y_extremes.y, test.expected_y_max);
            
            results[idx] = x_extremes_ok && y_extremes_ok;
            
            if (!results[idx]) {
                std::cout << "FAIL: " << test.description << std::endl;
                std::cout << "  X extremes - Expected: (" << test.expected_x_min << ", " << test.expected_x_max 
                          << "), Got: (" << extremes.x_extremes.x << ", " << extremes.x_extremes.y << ")" << std::endl;
                std::cout << "  Y extremes - Expected: (" << test.expected_y_min << ", " << test.expected_y_max 
                          << "), Got: (" << extremes.y_extremes.x << ", " << extremes.y_extremes.y << ")" << std::endl;
            }
        } else {
            results[idx] = true; // Skip invalid ellipses
        }
    }
}

// Host function to run tilt angle tests
bool testTiltAngleComputation() {
    std::cout << "\n=== Testing Tilt Angle Computation ===\n";
    
    // Test cases covering full angular range
    // θ = 0.5 * atan2(2σ_xy, σ_xx - σ_yy)
    std::vector<TiltAngleTestCase> test_cases = {
        // Axis-aligned ellipses 
        // For (1,0,2): θ = 0.5 * atan2(0, 1-2) = 0.5 * atan2(0, -1) = 0.5 * π = π/2
        {{1.0f, 0.0f, 2.0f}, M_PI/2.0f, "Axis-aligned horizontal ellipse"},
        {{2.0f, 0.0f, 1.0f}, 0.0f, "Axis-aligned vertical ellipse"},
        {{1.0f, 0.0f, 1.0f}, 0.0f, "Perfect circle"},
        
        // 45-degree tilted ellipses
        // For σ_xx = σ_yy, θ = 0.5 * atan2(2σ_xy, 0) = π/4 when σ_xy > 0
        {{1.0f, 1.0f, 1.0f}, M_PI/4.0f, "45-degree tilted ellipse"},
        {{2.0f, 2.0f, 2.0f}, M_PI/4.0f, "45-degree tilted ellipse (scaled)"},
        
        // Negative 45-degree tilt
        {{1.0f, -1.0f, 1.0f}, -M_PI/4.0f, "Negative 45-degree tilt"},
        
        // Edge cases with small values
        {{1.0f, 1e-6f, 1.0f}, 0.5f * atan2f(2e-6f, 0.0f), "Nearly axis-aligned (small B)"},
        {{1.0f + 1e-6f, 0.0f, 1.0f}, 0.0f, "Nearly circular (small difference)"},
        
        // Various specific angles
        // θ = 0.5 * atan2(2*1, 3-1) = 0.5 * atan2(2, 2) = 0.5 * π/4 = π/8
        {{3.0f, 1.0f, 1.0f}, 0.5f * atan2f(2.0f, 2.0f), "22.5-degree case"},
        
        // θ = 0.5 * atan2(4, -2) = 0.5 * atan2(4, -2)
        {{1.0f, 2.0f, 3.0f}, 0.5f * atan2f(4.0f, -2.0f), "Steep angle case"},
        
        // More test cases for different scenarios
        {{2.0f, 1.0f, 1.0f}, 0.5f * atan2f(2.0f, 1.0f), "Another angle case"},
        {{1.0f, -2.0f, 3.0f}, 0.5f * atan2f(-4.0f, -2.0f), "Negative B case"},
    };
    
    int num_cases = test_cases.size();
    
    // Allocate CPU memory
    bool* results = new bool[num_cases];
    
    // Run CPU tests
    testTiltAngleFunction(test_cases.data(), num_cases, results);
    
    // Check results
    int passed = 0;
    for (int i = 0; i < num_cases; i++) {
        if (results[i]) {
            passed++;
            std::cout << "PASS: " << test_cases[i].description << std::endl;
        }
    }
    
    std::cout << "Tilt Angle Tests: " << passed << "/" << num_cases << " passed\n";
    
    // Cleanup
    delete[] results;
    
    return passed == num_cases;
}

// Host function to run stretching factor tests
bool testStretchingFactorComputation() {
    std::cout << "\n=== Testing Stretching Factor Computation ===\n";
    
    // Test cases for critical angles
    std::vector<StretchingFactorTestCase> test_cases = {
        // Critical angles with β = 1.0
        {0.0f, 1.0f, 2.0f, "θ = 0° with β = 1.0 (maximum stretching)"},
        {M_PI/2.0f, 1.0f, 2.0f, "θ = 90° with β = 1.0 (maximum stretching)"},
        {M_PI/4.0f, 1.0f, 1.0f, "θ = 45° with β = 1.0 (minimal stretching)"},
        {3.0f*M_PI/4.0f, 1.0f, 1.0f, "θ = 135° with β = 1.0 (minimal stretching)"},
        
        // Critical angles with β = 1.1 (default)
        {0.0f, 1.1f, 2.1f, "θ = 0° with β = 1.1 (maximum stretching)"},
        {M_PI/2.0f, 1.1f, 2.1f, "θ = 90° with β = 1.1 (maximum stretching)"},
        {M_PI/4.0f, 1.1f, 1.0f, "θ = 45° with β = 1.1 (minimal stretching)"},
        
        // Critical angles with β = 1.2 (maximum)
        {0.0f, 1.2f, 2.2f, "θ = 0° with β = 1.2 (maximum stretching)"},
        {M_PI/2.0f, 1.2f, 2.2f, "θ = 90° with β = 1.2 (maximum stretching)"},
        {M_PI/4.0f, 1.2f, 1.0f, "θ = 45° with β = 1.2 (minimal stretching)"},
        
        // Intermediate angles
        {M_PI/6.0f, 1.1f, 1.0f + 1.1f * fabsf(cosf(M_PI/3.0f)), "θ = 30° with β = 1.1"},
        {M_PI/3.0f, 1.1f, 1.0f + 1.1f * fabsf(cosf(2.0f*M_PI/3.0f)), "θ = 60° with β = 1.1"},
        
        // Negative angles (should behave the same due to |cos(2θ)|)
        {-M_PI/4.0f, 1.1f, 1.0f, "θ = -45° with β = 1.1 (minimal stretching)"},
        {-M_PI/6.0f, 1.1f, 1.0f + 1.1f * fabsf(cosf(-M_PI/3.0f)), "θ = -30° with β = 1.1"},
        
        // Edge cases
        {0.0f, 1.0f, 2.0f, "Minimum β at critical angle"},
        {M_PI/4.0f, 2.0f, 1.0f, "Large β at minimal stretching angle"},
    };
    
    int num_cases = test_cases.size();
    
    // Allocate CPU memory
    bool* results = new bool[num_cases];
    
    // Run CPU tests
    testStretchingFactorFunction(test_cases.data(), num_cases, results);
    
    // Check results
    int passed = 0;
    for (int i = 0; i < num_cases; i++) {
        if (results[i]) {
            passed++;
            std::cout << "PASS: " << test_cases[i].description << std::endl;
        }
    }
    
    std::cout << "Stretching Factor Tests: " << passed << "/" << num_cases << " passed\n";
    
    // Cleanup
    delete[] results;
    
    return passed == num_cases;
}

// Host function to run extreme points tests
bool testExtremePointsComputation() {
    std::cout << "\n=== Testing Extreme Points Computation ===\n";
    
    // Test cases with analytical solutions
    std::vector<ExtremePointsTestCase> test_cases = {
        // Axis-aligned ellipse: x²/a² + y²/b² = 1 → Ax² + Cy² = t
        // For A=1, C=4, center=(0,0), opacity=1.0: x²/t + y²/(t/4) = 1
        // x_extremes = ±√t, y_extremes = ±√(t/4) = ±√t/2
        {{1.0f, 0.0f, 4.0f, 1.0f}, {0.0f, 0.0f}, 
         -sqrtf(2.0f * logf(255.0f)), sqrtf(2.0f * logf(255.0f)),
         -sqrtf(2.0f * logf(255.0f))/2.0f, sqrtf(2.0f * logf(255.0f))/2.0f,
         "Axis-aligned ellipse at origin"},
        
        // Translated axis-aligned ellipse
        {{1.0f, 0.0f, 4.0f, 1.0f}, {10.0f, 5.0f}, 
         10.0f - sqrtf(2.0f * logf(255.0f)), 10.0f + sqrtf(2.0f * logf(255.0f)),
         5.0f - sqrtf(2.0f * logf(255.0f))/2.0f, 5.0f + sqrtf(2.0f * logf(255.0f))/2.0f,
         "Translated axis-aligned ellipse"},
        
        // Circle: A = C, B = 0
        {{2.0f, 0.0f, 2.0f, 1.0f}, {0.0f, 0.0f}, 
         -sqrtf(logf(255.0f)), sqrtf(logf(255.0f)),
         -sqrtf(logf(255.0f)), sqrtf(logf(255.0f)),
         "Perfect circle at origin"},
        
        // Different opacity levels
        {{1.0f, 0.0f, 1.0f, 0.5f}, {0.0f, 0.0f}, 
         -sqrtf(2.0f * logf(127.5f)), sqrtf(2.0f * logf(127.5f)),
         -sqrtf(2.0f * logf(127.5f)), sqrtf(2.0f * logf(127.5f)),
         "Circle with opacity 0.5"},
        
        // Small ellipse
        {{10.0f, 0.0f, 10.0f, 1.0f}, {0.0f, 0.0f}, 
         -sqrtf(2.0f * logf(255.0f) / 10.0f), sqrtf(2.0f * logf(255.0f) / 10.0f),
         -sqrtf(2.0f * logf(255.0f) / 10.0f), sqrtf(2.0f * logf(255.0f) / 10.0f),
         "Small circle"},
    };
    
    int num_cases = test_cases.size();
    
    // Allocate CPU memory
    bool* results = new bool[num_cases];
    
    // Run CPU tests
    testExtremePointsFunction(test_cases.data(), num_cases, results);
    
    // Check results
    int passed = 0;
    for (int i = 0; i < num_cases; i++) {
        if (results[i]) {
            passed++;
            std::cout << "PASS: " << test_cases[i].description << std::endl;
        }
    }
    
    std::cout << "Extreme Points Tests: " << passed << "/" << num_cases << " passed\n";
    
    // Cleanup
    delete[] results;
    
    return passed == num_cases;
}

// Main test function
int main() {
    std::cout << "=== Dual-SnugBox Mathematical Functions Unit Tests ===\n";
    std::cout << "Testing Requirements: 1.1, 2.1, 2.2, 2.3, 2.4\n";
    std::cout << "Running CPU-based tests for mathematical function validation\n";
    
    bool all_passed = true;
    
    // Run all test suites
    all_passed &= testTiltAngleComputation();
    all_passed &= testStretchingFactorComputation();
    all_passed &= testExtremePointsComputation();
    
    // Summary
    std::cout << "\n=== Test Summary ===\n";
    if (all_passed) {
        std::cout << "✅ ALL TESTS PASSED - Mathematical functions are working correctly!\n";
        std::cout << "✅ Tilt angle calculation verified across full angular range\n";
        std::cout << "✅ Stretching factor behavior validated at critical angles (0°, 45°, 90°)\n";
        std::cout << "✅ Extreme point computation accuracy confirmed against analytical solutions\n";
        return 0;
    } else {
        std::cout << "❌ SOME TESTS FAILED - Please review the implementation\n";
        return 1;
    }
}