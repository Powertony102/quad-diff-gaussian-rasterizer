# Requirements Document

## Introduction

This document outlines the requirements for implementing the "Dual-SnugBox with Skew-Adaptive Stretching" optimization for 3D Gaussian Splatting rasterization. This optimization aims to significantly improve tile culling efficiency by replacing the single axis-aligned bounding box (AABB) approach with two asymmetric "half-boxes" that better fit tilted ellipses, combined with an adaptive stretching mechanism based on ellipse orientation.

## Requirements

### Requirement 1

**User Story:** As a 3D Gaussian Splatting developer, I want to implement dual-SnugBox tile culling optimization, so that I can reduce overdraw and improve rendering performance for scenes with anisotropic Gaussians.

#### Acceptance Criteria

1. WHEN the system processes a Gaussian for tile culling THEN it SHALL compute four extreme points of the projected ellipse using analytical methods
2. WHEN the extreme points are computed THEN the system SHALL partition them into left and right groups based on the Gaussian center's x-coordinate
3. WHEN the partitioning is complete THEN the system SHALL construct two asymmetric AABBs (left-box and right-box) using the partitioned points and center point
4. WHEN testing tile intersection THEN the system SHALL consider a tile as overlapping if it intersects with either the left-box OR the right-box

### Requirement 2

**User Story:** As a performance optimization engineer, I want to implement skew-adaptive stretching for the dual boxes, so that the system maintains proper coverage for ellipses that are nearly horizontal or vertical.

#### Acceptance Criteria

1. WHEN processing a Gaussian THEN the system SHALL compute the ellipse tilt angle θ using the covariance matrix method: θ = 0.5 * atan2(2σ_xy, σ_xx - σ_yy)
2. WHEN the tilt angle is computed THEN the system SHALL calculate the stretching factor s(θ) = 1 + β|cos(2θ)| where β ∈ [1.0, 1.2]
3. WHEN θ ≈ 0° or 90° THEN the stretching factor SHALL approach maximum value (s ≈ 2.0 when β=1.0)
4. WHEN θ ≈ 45° THEN the stretching factor SHALL be minimal (s = 1.0)
5. WHEN applying stretching THEN the system SHALL extend each half-box along its longer dimension away from the center point

### Requirement 3

**User Story:** As a system integrator, I want the dual-SnugBox implementation to seamlessly integrate with the existing CUDA rasterizer, so that it maintains compatibility with current 3DGS pipelines.

#### Acceptance Criteria

1. WHEN integrating the optimization THEN the system SHALL modify the existing `duplicateToTilesTouched` function without changing its interface
2. WHEN processing Gaussians THEN the system SHALL maintain the same input/output data structures as the current implementation
3. WHEN the optimization is active THEN the system SHALL preserve all existing functionality including backward pass compatibility
4. WHEN debugging is enabled THEN the system SHALL provide the same error checking and validation as the original implementation

### Requirement 4

**User Story:** As a performance analyst, I want the dual-SnugBox optimization to have minimal computational overhead, so that the performance gains from reduced tile processing outweigh the additional computation costs.

#### Acceptance Criteria

1. WHEN computing the dual boxes THEN the additional computation SHALL be O(1) constant time per Gaussian
2. WHEN calculating the tilt angle THEN the system SHALL use efficient trigonometric functions (atan2, cos) available on GPU
3. WHEN performing tile intersection tests THEN the system SHALL require at most one additional AABB-tile intersection test compared to single SnugBox
4. WHEN processing ill-formed ellipses THEN the system SHALL handle edge cases gracefully without performance degradation

### Requirement 5

**User Story:** As a rendering pipeline engineer, I want to ensure no duplicate key-value pairs are generated during tile processing, so that the sorting and rendering phases work correctly without redundant entries.

#### Acceptance Criteria

1. WHEN a tile intersects with both left-box and right-box THEN the system SHALL generate only ONE (tile_index, gaussian_index) key-value pair for that tile
2. WHEN processing dual boxes for tile intersection THEN the system SHALL use logical OR operation to determine tile overlap (intersects left-box OR right-box)
3. WHEN generating keys for sorting THEN each unique (tile_index, gaussian_index) combination SHALL appear exactly once in the output arrays
4. WHEN a tile is covered by the overlapping region of both boxes THEN the system SHALL avoid double-counting that tile

### Requirement 6

**User Story:** As a quality assurance engineer, I want the dual-SnugBox implementation to handle edge cases robustly, so that the system remains stable across diverse scene configurations.

#### Acceptance Criteria

1. WHEN encountering degenerate ellipses (det ≤ 0) THEN the system SHALL return zero tiles touched and skip processing
2. WHEN processing ellipses with extreme aspect ratios THEN the system SHALL apply appropriate numerical safeguards
3. WHEN the computed boxes extend beyond screen boundaries THEN the system SHALL clamp coordinates to valid tile ranges
4. WHEN floating-point precision issues occur THEN the system SHALL use the safety factor β to provide coverage buffer
5. WHEN the ellipse is very small or has low opacity THEN the system SHALL apply the same opacity thresholding as the original implementation