# Implementation Plan

- [x] 1. Implement core mathematical functions for dual-SnugBox algorithm
  - Create device functions for extreme point computation using analytical ellipse methods
  - Implement tilt angle calculation using covariance matrix eigenvalue approach
  - Add stretching factor computation with configurable beta parameter
  - _Requirements: 1.1, 2.1, 2.2, 4.1_

- [ ] 2. Develop dual-box construction logic
  - Implement left-right partitioning based on Gaussian center x-coordinate
  - Create asymmetric AABB construction using extreme points and center
  - Add adaptive stretching application to box dimensions
  - _Requirements: 1.2, 1.3, 2.3, 2.4, 2.5_

- [ ] 3. Create unique tile intersection generation system
  - Implement union-based tile processing to prevent duplicate key-value pairs
  - Add efficient AABB-tile intersection testing for dual boxes
  - Ensure single-pass key generation with proper indexing
  - _Requirements: 1.4, 5.1, 5.2, 5.3, 5.4_

- [ ] 4. Integrate dual-SnugBox into existing CUDA rasterizer
  - Modify duplicateToTilesTouched function in auxiliary.h to use dual-box algorithm
  - Preserve existing function interface and data structures
  - Maintain compatibility with current preprocessing pipeline
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 5. Add robust error handling and edge case management
  - Implement degenerate ellipse detection and graceful fallback
  - Add numerical stability safeguards for extreme aspect ratios
  - Include boundary clamping for screen-space coordinates
  - Apply opacity thresholding consistent with original implementation
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 6. Optimize performance and minimize computational overhead
  - Use efficient GPU trigonometric functions (atan2f, cosf, fabsf)
  - Minimize branching in device code for better SIMD utilization
  - Ensure O(1) computational complexity per Gaussian
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 7. Create comprehensive unit tests for mathematical functions
  - Test extreme point computation accuracy against analytical solutions
  - Verify tilt angle calculation across full angular range
  - Validate stretching factor behavior at critical angles (0°, 45°, 90°)
  - _Requirements: 1.1, 2.1, 2.2, 2.3, 2.4_

- [ ] 8. Implement integration tests for tile culling system
  - Verify no duplicate (tile_index, gaussian_index) pairs in output
  - Test coverage completeness for various ellipse orientations
  - Validate proper handling of overlapping box regions
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 9. Add debugging and validation capabilities
  - Implement debug mode with detailed logging of box construction
  - Add visual debugging output for tile coverage verification
  - Include performance counters for tile count comparisons
  - _Requirements: 3.4, 4.3_

- [ ] 10. Conduct performance benchmarking and validation
  - Compare tile counts against single SnugBox and AccuTile methods
  - Measure rendering performance improvements across diverse scenes
  - Validate maintained rendering quality with no visual artifacts
  - _Requirements: 4.1, 4.2, 4.3_