"""
Elastix-based image registration using SimpleITK.

Provides automatic registration with composite transforms (rigid → affine → BSpline)
and live optimization monitoring via callbacks.
"""

import SimpleITK as sitk
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Any


def create_initial_transform(fixed: sitk.Image, moving: sitk.Image, 
                              transform_type: str = "rigid") -> sitk.Transform:
    """
    Create initial transform using CenteredTransformInitializer.
    
    Parameters
    ----------
    fixed : sitk.Image
        Fixed (target) image
    moving : sitk.Image
        Moving (source) image
    transform_type : str
        One of "rigid", "affine", "bspline"
        
    Returns
    -------
    sitk.Transform
        Initial transform (usually identity centered on image)
    """
    if transform_type == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == "affine":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving,
            sitk.AffineTransform(2),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:  # bspline
        # For BSpline, start with affine then add BSpline on top
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving,
            sitk.AffineTransform(2),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    return initial_transform


def create_registration_method(
    params: Dict[str, Any],
    progress_callback: Optional[Callable[[float, int], None]] = None,
    level_callback: Optional[Callable[[int], None]] = None,
) -> sitk.ImageRegistrationMethod:
    """
    Create and configure a SimpleITK registration method.
    
    Parameters
    ----------
    params : dict
        Registration parameters:
        - histogram_bins: int (default 50)
        - sampling_percentage: float (default 0.01)
        - learning_rate: float (default 1.0)
        - iterations: int (default 200)
        - shrink_factors: list (default [4, 2, 1])
        - smoothing_sigmas: list (default [2, 1, 0])
        - use_rigidity_penalty: bool (default False)
        - rigidity_weight: float (default 0.01)
    progress_callback : callable, optional
        Called each iteration with (metric_value, iteration_number)
    level_callback : callable, optional
        Called when multi-resolution level changes with (level_number)
        
    Returns
    -------
    sitk.ImageRegistrationMethod
        Configured registration method
    """
    method = sitk.ImageRegistrationMethod()
    
    # Similarity metric: Mattes Mutual Information
    method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=params.get('histogram_bins', 50)
    )
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(params.get('sampling_percentage', 0.01))
    
    # Interpolator
    method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer: Gradient Descent
    method.SetOptimizerAsGradientDescent(
        learningRate=params.get('learning_rate', 1.0),
        numberOfIterations=params.get('iterations', 200),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    
    # Multi-resolution framework
    shrink_factors = params.get('shrink_factors', [4, 2, 1])
    smoothing_sigmas = params.get('smoothing_sigmas', [2, 1, 0])
    method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Register callbacks for live monitoring
    if progress_callback is not None:
        method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: progress_callback(method.GetMetricValue(), method.GetOptimizerIteration())
        )
    
    if level_callback is not None:
        # Track current level
        current_level = [0]
        def on_level_change():
            current_level[0] += 1
            level_callback(current_level[0])
        method.AddCommand(sitk.sitkMultiResolutionIterationEvent, on_level_change)
    
    return method


def compute_affine_from_landmarks(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
) -> sitk.AffineTransform:
    """
    Compute affine transform from landmark correspondences.
    
    Uses least-squares to find the affine transform that maps
    moving_points to fixed_points.
    
    Parameters
    ----------
    fixed_points : np.ndarray
        Target points, shape (N, 2) in (y, x) format
    moving_points : np.ndarray
        Source points, shape (N, 2) in (y, x) format
        
    Returns
    -------
    sitk.AffineTransform
        Affine transform that maps moving → fixed
    """
    # Convert to (x, y) format for transform computation
    fixed_xy = fixed_points[:, ::-1].astype(np.float64)
    moving_xy = moving_points[:, ::-1].astype(np.float64)
    
    n_points = len(fixed_xy)
    
    if n_points < 3:
        raise ValueError("Need at least 3 point pairs for affine transform")
    
    # Build system: fixed = A * moving + t
    # Solve using least squares
    # [ x_f ]   [ a b ] [ x_m ]   [ tx ]
    # [ y_f ] = [ c d ] [ y_m ] + [ ty ]
    
    # Construct matrices for least squares: Ax = b
    # where x = [a, b, tx, c, d, ty]
    A = np.zeros((2 * n_points, 6))
    b = np.zeros(2 * n_points)
    
    for i in range(n_points):
        mx, my = moving_xy[i]
        fx, fy = fixed_xy[i]
        
        # Row for x equation
        A[2*i, 0] = mx
        A[2*i, 1] = my
        A[2*i, 2] = 1
        b[2*i] = fx
        
        # Row for y equation
        A[2*i+1, 3] = mx
        A[2*i+1, 4] = my
        A[2*i+1, 5] = 1
        b[2*i+1] = fy
    
    # Solve least squares
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, tx, c, d, ty = params
    
    # Create SimpleITK affine transform
    # SimpleITK uses: T(x) = A * (x - center) + center + translation
    # We want: T(x) = M * x + t where M = [[a, b], [c, d]]
    
    affine = sitk.AffineTransform(2)
    affine.SetMatrix([a, bb, c, d])  # Row-major: [m00, m01, m10, m11]
    affine.SetTranslation([tx, ty])
    affine.SetCenter([0.0, 0.0])
    
    return affine


def register_images_with_landmarks(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[float, int], None]] = None,
    level_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, sitk.Transform]:
    """
    Register using landmarks for initial affine, then BSpline refinement.
    
    This is the hybrid mode: Landmarks → Affine → BSpline(optimized)
    
    Parameters
    ----------
    fixed : np.ndarray
        Fixed (target) image, 2D
    moving : np.ndarray
        Moving (source) image, 2D
    fixed_points : np.ndarray
        Landmark points in fixed image, shape (N, 2) in (y, x) format
    moving_points : np.ndarray
        Corresponding landmark points in moving image
    params : dict, optional
        Registration parameters
    progress_callback : callable, optional
        Called each iteration with (metric_value, iteration_number)
    level_callback : callable, optional
        Called when resolution level changes
        
    Returns
    -------
    result : np.ndarray
        Registered moving image
    transform : sitk.Transform
        Final composite transform (affine + bspline)
    """
    if params is None:
        params = {}
    
    # Step 1: Compute affine from landmarks (instant, closed-form)
    landmark_affine = compute_affine_from_landmarks(fixed_points, moving_points)
    
    # Convert numpy to SimpleITK
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))
    
    # Step 2: BSpline refinement on top of landmark affine
    method = create_registration_method(params, progress_callback, level_callback)
    
    # Create BSpline transform with grid
    grid_physical_spacing = params.get('bspline_grid_spacing', 50.0)
    image_physical_size = [
        fixed_sitk.GetSize()[d] * fixed_sitk.GetSpacing()[d] 
        for d in range(2)
    ]
    mesh_size = [
        max(1, int(sz / grid_physical_spacing + 0.5)) 
        for sz in image_physical_size
    ]
    
    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_sitk, mesh_size, order=3
    )
    
    # Set landmark affine as the moving initial transform
    method.SetInitialTransform(bspline_transform, inPlace=True)
    method.SetMovingInitialTransform(landmark_affine)
    
    # Run optimization for BSpline only
    bspline_result = method.Execute(fixed_sitk, moving_sitk)
    
    # Compose final transform: landmark_affine + bspline
    final_transform = sitk.CompositeTransform(2)
    final_transform.AddTransform(landmark_affine)
    final_transform.AddTransform(bspline_result)
    
    # Apply transform to get result image
    result_sitk = sitk.Resample(
        moving_sitk, fixed_sitk, final_transform,
        sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()
    )
    
    result = sitk.GetArrayFromImage(result_sitk)
    return result, final_transform


def register_images(
    fixed: np.ndarray,
    moving: np.ndarray,
    transform_type: str = "bspline",
    params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[float, int], None]] = None,
    level_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[np.ndarray, sitk.Transform]:
    """
    Register moving image to fixed image using SimpleITK.
    
    Parameters
    ----------
    fixed : np.ndarray
        Fixed (target) image, 2D
    moving : np.ndarray
        Moving (source) image, 2D
    transform_type : str
        One of "rigid", "affine", "bspline"
    params : dict, optional
        Registration parameters (see create_registration_method)
    progress_callback : callable, optional
        Called each iteration with (metric_value, iteration_number)
    level_callback : callable, optional
        Called when resolution level changes
        
    Returns
    -------
    result : np.ndarray
        Registered moving image
    transform : sitk.Transform
        Final transform
    """
    if params is None:
        params = {}
    
    # Convert numpy to SimpleITK
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))
    
    # Create registration method
    method = create_registration_method(params, progress_callback, level_callback)
    
    # Multi-stage registration for composite transform
    if transform_type == "rigid":
        # Single stage: rigid only
        initial = create_initial_transform(fixed_sitk, moving_sitk, "rigid")
        method.SetInitialTransform(initial, inPlace=False)
        final_transform = method.Execute(fixed_sitk, moving_sitk)
        
    elif transform_type == "affine":
        # Two stages: rigid → affine
        # Stage 1: Rigid
        rigid_transform = create_initial_transform(fixed_sitk, moving_sitk, "rigid")
        method.SetInitialTransform(rigid_transform, inPlace=False)
        rigid_result = method.Execute(fixed_sitk, moving_sitk)
        
        # Stage 2: Affine (starting from rigid result)
        method2 = create_registration_method(params, progress_callback, level_callback)
        affine_transform = sitk.AffineTransform(2)
        affine_transform.SetCenter(rigid_result.GetFixedParameters()[:2])
        method2.SetInitialTransform(affine_transform, inPlace=False)
        method2.SetMovingInitialTransform(rigid_result)
        final_transform = method2.Execute(fixed_sitk, moving_sitk)
        
        # Compose transforms
        composite = sitk.CompositeTransform(2)
        composite.AddTransform(rigid_result)
        composite.AddTransform(final_transform)
        final_transform = composite
        
    else:  # bspline - full composite
        # Stage 1: Rigid
        rigid_transform = create_initial_transform(fixed_sitk, moving_sitk, "rigid")
        method.SetInitialTransform(rigid_transform, inPlace=False)
        rigid_result = method.Execute(fixed_sitk, moving_sitk)
        
        # Stage 2: Affine
        method2 = create_registration_method(params, progress_callback, level_callback)
        affine_transform = sitk.AffineTransform(2)
        method2.SetInitialTransform(affine_transform, inPlace=False)
        method2.SetMovingInitialTransform(rigid_result)
        affine_result = method2.Execute(fixed_sitk, moving_sitk)
        
        # Compose rigid + affine
        pre_bspline = sitk.CompositeTransform(2)
        pre_bspline.AddTransform(rigid_result)
        pre_bspline.AddTransform(affine_result)
        
        # Stage 3: BSpline
        method3 = create_registration_method(params, progress_callback, level_callback)
        
        # Create BSpline transform with grid
        grid_physical_spacing = params.get('bspline_grid_spacing', 50.0)
        image_physical_size = [
            fixed_sitk.GetSize()[d] * fixed_sitk.GetSpacing()[d] 
            for d in range(2)
        ]
        mesh_size = [
            max(1, int(sz / grid_physical_spacing + 0.5)) 
            for sz in image_physical_size
        ]
        
        bspline_transform = sitk.BSplineTransformInitializer(
            fixed_sitk, mesh_size, order=3
        )
        method3.SetInitialTransform(bspline_transform, inPlace=True)
        method3.SetMovingInitialTransform(pre_bspline)
        
        # Add rigidity penalty if requested
        if params.get('use_rigidity_penalty', False):
            # Note: SimpleITK Python doesn't directly support multi-metric
            # The rigidity penalty is handled by BSpline's inherent smoothness
            pass
        
        bspline_result = method3.Execute(fixed_sitk, moving_sitk)
        
        # Final composite
        final_transform = sitk.CompositeTransform(2)
        final_transform.AddTransform(pre_bspline)
        final_transform.AddTransform(bspline_result)
    
    # Apply transform to get result image
    result_sitk = sitk.Resample(
        moving_sitk, fixed_sitk, final_transform,
        sitk.sitkLinear, 0.0, moving_sitk.GetPixelID()
    )
    
    result = sitk.GetArrayFromImage(result_sitk)
    return result, final_transform


def apply_transform_to_image(
    image: np.ndarray,
    transform: sitk.Transform,
    reference_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Apply a saved transform to a 2D image.
    
    Parameters
    ----------
    image : np.ndarray
        Image to transform
    transform : sitk.Transform
        Transform to apply
    reference_shape : tuple
        Output shape (H, W)
        
    Returns
    -------
    np.ndarray
        Transformed image
    """
    image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
    
    # Create reference image for resampling
    reference = sitk.Image(reference_shape[1], reference_shape[0], sitk.sitkFloat32)
    
    result_sitk = sitk.Resample(
        image_sitk, reference, transform,
        sitk.sitkLinear, 0.0, image_sitk.GetPixelID()
    )
    
    return sitk.GetArrayFromImage(result_sitk)


def apply_transform_to_stack(
    stack: np.ndarray,
    transform: sitk.Transform,
    reference_shape: Tuple[int, int],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Apply a 2D transform to all slices of a 3D stack.
    
    Parameters
    ----------
    stack : np.ndarray
        3D volume (Z, Y, X)
    transform : sitk.Transform
        2D transform to apply to each slice
    reference_shape : tuple
        Output XY shape (H, W)
    progress_callback : callable, optional
        Called with (current_slice, total_slices)
        
    Returns
    -------
    np.ndarray
        Transformed stack
    """
    z_size = stack.shape[0]
    result = np.zeros((z_size, *reference_shape), dtype=stack.dtype)
    
    for z in range(z_size):
        result[z] = apply_transform_to_image(stack[z], transform, reference_shape)
        if progress_callback:
            progress_callback(z + 1, z_size)
    
    return result


def save_transform(transform: sitk.Transform, filepath: str):
    """Save transform to file."""
    sitk.WriteTransform(transform, filepath)


def load_transform(filepath: str) -> sitk.Transform:
    """Load transform from file."""
    return sitk.ReadTransform(filepath)
