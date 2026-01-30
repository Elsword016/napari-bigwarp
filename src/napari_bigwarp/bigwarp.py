import cv2
import json
import numpy as np


def bigwarp(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
):
    """Apply TPS transform to warp moving image to fixed image coordinate space.
    
    Parameters
    ----------
    fixed : np.ndarray
        Fixed/target image (used for determining output shape)
    moving : np.ndarray  
        Moving/source image to be warped
    fixed_points : np.ndarray
        Control points in fixed image (Nx2, row-col order)
    moving_points : np.ndarray
        Corresponding control points in moving image (Nx2, row-col order)
    
    Returns
    -------
    np.ndarray
        Warped moving image aligned to fixed image
    """
    moving_pts_init = np.array(
        [
            [0.0, 0.0],
            [0.0, moving.shape[1]],
            [moving.shape[0], moving.shape[1]],
            [moving.shape[0], 0.0],
        ]
    )
    fixed_pts_init = np.array(
        [
            [0.0, 0.0],
            [0.0, fixed.shape[1]],
            [fixed.shape[0], fixed.shape[1]],
            [fixed.shape[0], 0.0],
        ]
    )
    moving_pts = np.concatenate([moving_pts_init, moving_points[:, ::-1]])
    fixed_pts = np.concatenate([fixed_pts_init, fixed_points[:, ::-1]])

    matches = [cv2.DMatch(i, i, 0) for i in range(len(moving_pts))]
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(fixed_pts[None, ...], moving_pts[None, ...], matches)
    out_img = tps.warpImage(np.array(moving))
    return out_img


def apply_transform_from_file(
    transform_path: str,
    moving: np.ndarray,
    fixed_shape: tuple = None,
) -> np.ndarray:
    """Apply a saved TPS transform to any image.
    
    Parameters
    ----------
    transform_path : str
        Path to the JSON transform file exported from BigWarp
    moving : np.ndarray
        Image to warp (should match the moving image dimensions from transform)
    fixed_shape : tuple, optional
        Output shape. If None, uses the fixed_image_shape from the transform file
    
    Returns
    -------
    np.ndarray
        Warped image
    
    Example
    -------
    >>> from napari_bigwarp.bigwarp import apply_transform_from_file
    >>> warped = apply_transform_from_file("my_transform.json", my_image)
    """
    with open(transform_path, "r") as f:
        transform_data = json.load(f)
    
    fixed_points = np.array(transform_data["fixed_points"])
    moving_points = np.array(transform_data["moving_points"])
    
    if fixed_shape is None:
        fixed_shape = tuple(transform_data["fixed_image_shape"])
    
    # Create a dummy fixed image of the right shape
    fixed_dummy = np.zeros(fixed_shape[:2], dtype=moving.dtype)
    
    return bigwarp(fixed_dummy, moving, fixed_points, moving_points)


def get_tps_transformer(fixed_points: np.ndarray, moving_points: np.ndarray, fixed_shape: tuple, moving_shape: tuple):
    """Get the OpenCV TPS transformer object for custom operations.
    
    Parameters
    ----------
    fixed_points : np.ndarray
        Control points in fixed image (Nx2, row-col order)
    moving_points : np.ndarray
        Corresponding control points in moving image (Nx2, row-col order)
    fixed_shape : tuple
        Shape of the fixed image (height, width)
    moving_shape : tuple
        Shape of the moving image (height, width)
    
    Returns
    -------
    cv2.ThinPlateSplineShapeTransformer
        The fitted TPS transformer
    """
    moving_pts_init = np.array([
        [0.0, 0.0],
        [0.0, moving_shape[1]],
        [moving_shape[0], moving_shape[1]],
        [moving_shape[0], 0.0],
    ])
    fixed_pts_init = np.array([
        [0.0, 0.0],
        [0.0, fixed_shape[1]],
        [fixed_shape[0], fixed_shape[1]],
        [fixed_shape[0], 0.0],
    ])
    
    moving_pts = np.concatenate([moving_pts_init, moving_points[:, ::-1]])
    fixed_pts = np.concatenate([fixed_pts_init, fixed_points[:, ::-1]])
    
    matches = [cv2.DMatch(i, i, 0) for i in range(len(moving_pts))]
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(fixed_pts[None, ...], moving_pts[None, ...], matches)
    return tps


def compute_deformation_field(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    output_shape: tuple,
    moving_shape: tuple = None,
) -> tuple:
    """Compute the dense deformation field from TPS control points.
    
    The deformation field represents, for each pixel in the fixed image,
    the displacement needed to map to the corresponding location in the moving image.
    
    Parameters
    ----------
    fixed_points : np.ndarray
        Control points in fixed image (Nx2, row-col order)
    moving_points : np.ndarray
        Corresponding control points in moving image (Nx2, row-col order)
    output_shape : tuple
        Shape of the output deformation field (height, width)
    moving_shape : tuple, optional
        Shape of the moving image. If None, uses output_shape
    
    Returns
    -------
    tuple of (displacement_y, displacement_x)
        Two arrays of shape output_shape containing the y and x displacements
        at each pixel. displacement[y, x] = (dy, dx) means the pixel at (y, x)
        in the fixed image corresponds to (y + dy, x + dx) in the moving image.
    """
    if moving_shape is None:
        moving_shape = output_shape
    
    height, width = output_shape[:2]
    
    # Create coordinate grids in the fixed image space
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
    
    # Stack into (N, 1, 2) format for cv2.transform
    # Note: OpenCV uses (x, y) order, so we swap
    coords_fixed = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).reshape(-1, 1, 2)
    
    # Get TPS transformer
    tps = get_tps_transformer(fixed_points, moving_points, output_shape, moving_shape)
    
    # Apply TPS transform to get corresponding coordinates in moving image space
    # This gives us where each fixed pixel maps to in the moving image
    coords_moving = tps.applyTransformation(coords_fixed.astype(np.float32))[1]
    
    # Reshape back to image dimensions
    coords_moving = coords_moving.reshape(height, width, 2)
    
    # Compute displacements (moving - fixed)
    # coords_moving is in (x, y) order from OpenCV
    displacement_x = coords_moving[:, :, 0] - x_coords
    displacement_y = coords_moving[:, :, 1] - y_coords
    
    return displacement_y, displacement_x


def save_deformation_field(
    displacement_y: np.ndarray,
    displacement_x: np.ndarray,
    filepath: str,
    scale: tuple = None,
):
    """Save deformation field to file.
    
    Parameters
    ----------
    displacement_y : np.ndarray
        Y displacement field
    displacement_x : np.ndarray
        X displacement field
    filepath : str
        Output path. Supports .npy (numpy) and .tiff/.tif (multi-page TIFF)
    scale : tuple, optional
        Physical scale (scale_y, scale_x) in units like microns/pixel.
        If provided, displacements are also saved in physical units.
    """
    if filepath.endswith('.npy'):
        # Save as numpy archive with both components
        np.save(filepath, np.stack([displacement_y, displacement_x], axis=0))
    elif filepath.endswith(('.tiff', '.tif')):
        # Save as multi-page TIFF (2 pages: dy, dx)
        import tifffile
        stack = np.stack([displacement_y, displacement_x], axis=0).astype(np.float32)
        tifffile.imwrite(filepath, stack, metadata={'axes': 'CYX', 'channels': ['displacement_y', 'displacement_x']})
    else:
        # Default to numpy format
        np.save(filepath + '.npy', np.stack([displacement_y, displacement_x], axis=0))


# =============================================================================
# Similarity / Affine / Rigid Transforms (for 10x ↔ 100x registration)
# =============================================================================

def compute_similarity_transform(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
) -> dict:
    """Compute similarity transform (scale + rotation + translation) from landmarks.
    
    Parameters
    ----------
    fixed_points : np.ndarray
        Control points in fixed image (Nx2, row-col order)
    moving_points : np.ndarray
        Corresponding control points in moving image (Nx2, row-col order)
    
    Returns
    -------
    dict with keys:
        'matrix': 3x3 transformation matrix
        'scale': uniform scale factor
        'rotation': rotation in radians
        'translation': (ty, tx) translation
        'residuals': per-point residual errors
        'mean_error': mean residual error in pixels
    """
    from skimage.transform import SimilarityTransform
    
    # Napari points are (row, col) = (y, x), but skimage expects (x, y)
    fixed_xy = fixed_points[:, ::-1]
    moving_xy = moving_points[:, ::-1]
    
    T = SimilarityTransform()
    success = T.estimate(moving_xy, fixed_xy)
    
    if not success:
        raise ValueError("Failed to estimate similarity transform")
    
    # Compute residuals (in xy space)
    transformed = T(moving_xy)
    residuals = np.sqrt(np.sum((transformed - fixed_xy) ** 2, axis=1))
    
    return {
        'matrix': T.params,
        'scale': T.scale,
        'rotation': T.rotation,
        'translation': T.translation,
        'residuals': residuals,
        'mean_error': np.mean(residuals),
        'transform_type': 'similarity',
    }


def compute_affine_transform(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
) -> dict:
    """Compute affine transform from landmarks.
    
    Parameters
    ----------
    fixed_points : np.ndarray
        Control points in fixed image (Nx2, row-col order)
    moving_points : np.ndarray
        Corresponding control points in moving image (Nx2, row-col order)
    
    Returns
    -------
    dict with keys:
        'matrix': 3x3 transformation matrix
        'residuals': per-point residual errors
        'mean_error': mean residual error in pixels
    """
    from skimage.transform import AffineTransform
    
    # Napari points are (row, col) = (y, x), but skimage expects (x, y)
    fixed_xy = fixed_points[:, ::-1]
    moving_xy = moving_points[:, ::-1]
    
    T = AffineTransform()
    success = T.estimate(moving_xy, fixed_xy)
    
    if not success:
        raise ValueError("Failed to estimate affine transform")
    
    # Compute residuals (in xy space)
    transformed = T(moving_xy)
    residuals = np.sqrt(np.sum((transformed - fixed_xy) ** 2, axis=1))
    
    return {
        'matrix': T.params,
        'scale': T.scale,
        'rotation': T.rotation,
        'translation': T.translation,
        'shear': T.shear,
        'residuals': residuals,
        'mean_error': np.mean(residuals),
        'transform_type': 'affine',
    }


def compute_rigid_transform(
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
) -> dict:
    """Compute rigid transform (rotation + translation only) from landmarks.
    
    Parameters
    ----------
    fixed_points : np.ndarray
        Control points in fixed image (Nx2, row-col order)
    moving_points : np.ndarray
        Corresponding control points in moving image (Nx2, row-col order)
    
    Returns
    -------
    dict with keys:
        'matrix': 3x3 transformation matrix
        'rotation': rotation in radians
        'translation': (ty, tx) translation
        'residuals': per-point residual errors
        'mean_error': mean residual error in pixels
    """
    from skimage.transform import EuclideanTransform
    
    # Napari points are (row, col) = (y, x), but skimage expects (x, y)
    fixed_xy = fixed_points[:, ::-1]
    moving_xy = moving_points[:, ::-1]
    
    T = EuclideanTransform()
    success = T.estimate(moving_xy, fixed_xy)
    
    if not success:
        raise ValueError("Failed to estimate rigid transform")
    
    # Compute residuals (in xy space)
    transformed = T(moving_xy)
    residuals = np.sqrt(np.sum((transformed - fixed_xy) ** 2, axis=1))
    
    return {
        'matrix': T.params,
        'rotation': T.rotation,
        'translation': T.translation,
        'residuals': residuals,
        'mean_error': np.mean(residuals),
        'transform_type': 'rigid',
    }


def apply_linear_transform(
    moving: np.ndarray,
    transform_matrix: np.ndarray,
    output_shape: tuple = None,
) -> np.ndarray:
    """Apply a linear transform (similarity/affine/rigid) to an image.
    
    Parameters
    ----------
    moving : np.ndarray
        Image to transform
    transform_matrix : np.ndarray
        3x3 transformation matrix
    output_shape : tuple, optional
        Output shape. If None, uses moving image shape.
    
    Returns
    -------
    np.ndarray
        Transformed image
    """
    from skimage.transform import warp, AffineTransform
    
    if output_shape is None:
        output_shape = moving.shape[:2]
    
    T = AffineTransform(matrix=transform_matrix)
    
    return warp(
        moving.astype(float),
        T.inverse,
        output_shape=output_shape,
        preserve_range=True
    ).astype(moving.dtype)


def bigwarp_linear(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_points: np.ndarray,
    moving_points: np.ndarray,
    transform_type: str = 'similarity',
) -> tuple:
    """Apply linear transform to warp moving image to fixed image coordinate space.
    
    Parameters
    ----------
    fixed : np.ndarray
        Fixed/target image (used for determining output shape)
    moving : np.ndarray
        Moving/source image to be warped
    fixed_points : np.ndarray
        Control points in fixed image (Nx2, row-col order)
    moving_points : np.ndarray
        Corresponding control points in moving image (Nx2, row-col order)
    transform_type : str
        One of 'similarity', 'affine', 'rigid'
    
    Returns
    -------
    tuple of (warped_image, transform_info)
    """
    if transform_type == 'similarity':
        result = compute_similarity_transform(fixed_points, moving_points)
    elif transform_type == 'affine':
        result = compute_affine_transform(fixed_points, moving_points)
    elif transform_type == 'rigid':
        result = compute_rigid_transform(fixed_points, moving_points)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    warped = apply_linear_transform(moving, result['matrix'], fixed.shape[:2])
    
    return warped, result
