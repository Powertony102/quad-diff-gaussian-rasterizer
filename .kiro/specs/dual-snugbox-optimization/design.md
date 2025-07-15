# Design Document

## Overview

The Dual-SnugBox with Skew-Adaptive Stretching optimization replaces the current single AABB tile culling approach with a more sophisticated dual-box system that better approximates tilted ellipses. This design maintains O(1) computational complexity while significantly reducing overdraw for anisotropic Gaussians.

## Architecture

The optimization integrates into the existing CUDA rasterizer pipeline by modifying the `duplicateToTilesTouched` function in `auxiliary.h`. The core algorithm consists of three main phases:

1. **Ellipse Analysis Phase**: Compute extreme points and tilt angle
2. **Dual-Box Construction Phase**: Create left and right asymmetric AABBs with adaptive stretching
3. **Tile Intersection Phase**: Generate unique tile-Gaussian key-value pairs using union logic

## Components and Interfaces

### Core Algorithm Components

#### 1. Extreme Point Computation
```cuda
struct ExtremePoints {
    float2 x_extremes;  // (x_min, x_max)
    float2 y_extremes;  // (y_min, y_max)
    float2 x_coords_at_y_extremes;  // (x1, x2) at (y_min, y_max)
    float2 y_coords_at_x_extremes;  // (y1, y2) at (x_min, x_max)
};
```

#### 2. Dual Box Structure
```cuda
struct DualBox {
    float4 left_box;   // (min_x, min_y, max_x, max_y)
    float4 right_box;  // (min_x, min_y, max_x, max_y)
    bool valid;        // Whether boxes are valid
};
```

#### 3. Tilt Angle Calculator
```cuda
__device__ inline float computeTiltAngle(const float3& cov2d) {
    // θ = 0.5 * atan2(2σ_xy, σ_xx - σ_yy)
    return 0.5f * atan2f(2.0f * cov2d.y, cov2d.x - cov2d.z);
}
```

#### 4. Stretching Factor Calculator
```cuda
__device__ inline float computeStretchingFactor(float theta, float beta = 1.1f) {
    // s(θ) = 1 + β|cos(2θ)|
    return 1.0f + beta * fabsf(cosf(2.0f * theta));
}
```

### Modified Function Interface

The main modification occurs in the existing `duplicateToTilesTouched` function:

```cuda
__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p,           // Gaussian center in screen space
    const float4 con_o,       // Conic coefficients + opacity
    const dim3 grid,          // Tile grid dimensions
    uint32_t idx,             // Gaussian index
    uint32_t off,             // Offset in output arrays
    float depth,              // Gaussian depth
    uint64_t* gaussian_keys_unsorted,    // Output keys
    uint32_t* gaussian_values_unsorted   // Output values
);
```

## Data Models

### Input Data
- **Conic Matrix**: `float4 con_o` containing (A, B, C, opacity) where the ellipse equation is Ax² + 2Bxy + Cy² = constant
- **Screen Position**: `float2 p` representing the Gaussian center in pixel coordinates
- **Grid Information**: `dim3 grid` specifying tile grid dimensions

### Intermediate Data
- **Extreme Points**: Four boundary points of the ellipse in each cardinal direction
- **Tilt Angle**: Orientation of the ellipse's major axis relative to x-axis
- **Stretching Factor**: Adaptive scaling based on tilt angle
- **Dual Boxes**: Two AABBs representing left and right halves

### Output Data
- **Tile Count**: Number of unique tiles touched by the Gaussian
- **Key-Value Pairs**: Sorted arrays for tile-Gaussian associations

## Error Handling

### Degenerate Cases
1. **Invalid Ellipse**: When discriminant ≥ 0, return 0 tiles
2. **Zero Opacity**: When opacity below threshold (1/255), return 0 tiles
3. **Extreme Aspect Ratios**: Apply numerical safeguards and clamping
4. **Boundary Conditions**: Clamp tile coordinates to valid grid ranges

### Numerical Stability
- Use `atan2f` instead of `atan` for robust angle computation
- Apply epsilon values for floating-point comparisons
- Implement safety factor β ∈ [1.0, 1.2] for coverage buffer

## Testing Strategy

### Unit Tests
1. **Extreme Point Accuracy**: Verify computed extreme points match analytical solutions
2. **Tilt Angle Precision**: Test angle computation across full range [0°, 90°]
3. **Stretching Function**: Validate s(θ) behavior at key angles (0°, 45°, 90°)
4. **Box Construction**: Ensure left/right boxes properly partition ellipse coverage

### Integration Tests
1. **No Duplicate Keys**: Verify unique (tile, Gaussian) pairs in output
2. **Coverage Completeness**: Ensure all ellipse-intersecting tiles are captured
3. **Performance Benchmarks**: Compare tile counts vs. single SnugBox and AccuTile
4. **Edge Case Handling**: Test degenerate ellipses and boundary conditions

### Validation Tests
1. **Visual Verification**: Render tile boundaries to verify coverage accuracy
2. **Statistical Analysis**: Measure overdraw reduction across diverse scenes
3. **Regression Testing**: Ensure no degradation in rendering quality

## Implementation Details

### Algorithm Flow

```cuda
__device__ inline uint32_t duplicateToTilesTouched_DualSnugBox(
    const float2 p, const float4 con_o, const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted) {
    
    // Phase 1: Validate ellipse and compute basic parameters
    float disc = con_o.y * con_o.y - con_o.x * con_o.z;
    if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) return 0;
    
    float t = 2.0f * logf(con_o.w * 255.0f);
    if (t <= 0) return 0;
    
    // Phase 2: Compute extreme points
    ExtremePoints extremes = computeExtremePoints(con_o, disc, t, p);
    
    // Phase 3: Calculate tilt angle and stretching factor
    float theta = computeTiltAngle({con_o.x, con_o.y, con_o.z});
    float stretch_factor = computeStretchingFactor(theta);
    
    // Phase 4: Construct dual boxes with stretching
    DualBox dual_box = constructDualBoxes(extremes, p, stretch_factor);
    if (!dual_box.valid) return 0;
    
    // Phase 5: Generate unique tile intersections
    return generateUniqueTileIntersections(dual_box, grid, idx, off, depth,
                                         gaussian_keys_unsorted, gaussian_values_unsorted);
}
```

### Key Implementation Considerations

1. **Memory Efficiency**: Reuse existing data structures without additional memory allocation
2. **Branch Minimization**: Use conditional moves instead of branches where possible
3. **Vectorization**: Leverage GPU SIMD operations for box intersection tests
4. **Cache Optimization**: Maintain spatial locality in tile processing order

### Duplicate Prevention Strategy

To ensure no duplicate (tile_index, gaussian_index) pairs:

1. **Union-Based Approach**: For each tile in the combined bounding rectangle, test intersection with (left_box OR right_box)
2. **Single Pass Generation**: Generate keys in a single loop over the union of both boxes
3. **Intersection Logic**: Use boolean OR to determine if tile intersects either box

```cuda
// Pseudo-code for duplicate prevention
for (int tile_y = rect_min.y; tile_y < rect_max.y; ++tile_y) {
    for (int tile_x = rect_min.x; tile_x < rect_max.x; ++tile_x) {
        bool intersects_left = tileIntersectsBox(tile_x, tile_y, dual_box.left_box);
        bool intersects_right = tileIntersectsBox(tile_x, tile_y, dual_box.right_box);
        
        if (intersects_left || intersects_right) {
            // Generate single key-value pair for this tile
            generateKeyValuePair(tile_x, tile_y, idx, off++, depth, keys, values);
        }
    }
}
```

This design ensures that each tile is processed exactly once, eliminating the possibility of duplicate entries while maintaining the improved culling efficiency of the dual-box approach.