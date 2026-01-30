"""
Automatic landmark detection for vessel/structure MIPs.

Detects endpoints and bifurcations in skeletonized images,
matches them between fixed and moving images using local descriptors,
and filters matches using RANSAC.
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, filters, exposure


def preprocess_mip(image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
    """Preprocess MIP for skeleton extraction.
    
    Parameters
    ----------
    image : np.ndarray
        Input MIP image (can be any dtype)
    enhance_contrast : bool
        Whether to apply contrast enhancement
    
    Returns
    -------
    np.ndarray
        Preprocessed binary image ready for skeletonization
    """
    # Convert to float and normalize
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    if enhance_contrast:
        # Adaptive histogram equalization
        img = exposure.equalize_adapthist(img, clip_limit=0.03)
    
    # Gaussian blur to reduce noise
    img = filters.gaussian(img, sigma=1.0)
    
    # Otsu thresholding
    thresh = filters.threshold_otsu(img)
    binary = img > thresh
    
    # Clean up small objects
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    
    return binary.astype(np.uint8)


def extract_skeleton_keypoints(
    binary: np.ndarray,
    detect_endpoints: bool = True,
    detect_bifurcations: bool = True,
) -> np.ndarray:
    """Extract keypoints from skeleton (endpoints and bifurcations).
    
    Parameters
    ----------
    binary : np.ndarray
        Binary image of structures
    detect_endpoints : bool
        Include skeleton endpoints (degree-1 nodes)
    detect_bifurcations : bool
        Include bifurcation points (degree-3+ nodes)
    
    Returns
    -------
    np.ndarray
        Keypoint coordinates as Nx2 array (row, col)
    """
    # Skeletonize
    skeleton = morphology.skeletonize(binary > 0)
    
    # Compute degree at each skeleton pixel using convolution
    # Count neighbors using a 3x3 kernel
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    neighbor_count = neighbor_count * skeleton  # Only count on skeleton pixels
    
    keypoints = []
    
    if detect_endpoints:
        # Endpoints have exactly 1 neighbor
        endpoints = (neighbor_count == 1) & skeleton
        endpoint_coords = np.array(np.where(endpoints)).T  # (row, col) format
        keypoints.append(endpoint_coords)
    
    if detect_bifurcations:
        # Bifurcations have 3+ neighbors
        bifurcations = (neighbor_count >= 3) & skeleton
        bifurc_coords = np.array(np.where(bifurcations)).T
        keypoints.append(bifurc_coords)
    
    if keypoints:
        return np.vstack(keypoints)
    return np.array([]).reshape(0, 2)


def compute_local_descriptors(
    image: np.ndarray,
    keypoints: np.ndarray,
    patch_size: int = 32,
) -> np.ndarray:
    """Compute local descriptors around each keypoint.
    
    Uses HOG-like features for rotation-invariant matching.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image
    keypoints : np.ndarray
        Keypoint coordinates (Nx2, row-col format)
    patch_size : int
        Size of patch around each keypoint
    
    Returns
    -------
    np.ndarray
        Descriptors as NxD array
    """
    half = patch_size // 2
    img = image.astype(np.float32)
    if img.max() > 1:
        img = img / img.max()
    
    # Pad image
    padded = np.pad(img, half, mode='reflect')
    
    descriptors = []
    for r, c in keypoints:
        r, c = int(r), int(c)
        # Extract patch (adjusted for padding)
        patch = padded[r:r + patch_size, c:c + patch_size]
        
        if patch.shape != (patch_size, patch_size):
            # Edge case: create zero descriptor
            descriptors.append(np.zeros(128))
            continue
        
        # Convert to uint8 for ORB
        patch_uint8 = (patch * 255).astype(np.uint8)
        
        # Compute gradient histogram (simple HOG-like)
        gx = cv2.Sobel(patch_uint8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch_uint8, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Create histogram in 8 orientation bins across 4x4 cells (= 128 dims like SIFT)
        n_cells = 4
        n_bins = 8
        cell_size = patch_size // n_cells
        descriptor = []
        
        for ci in range(n_cells):
            for cj in range(n_cells):
                cell_mag = mag[ci*cell_size:(ci+1)*cell_size, cj*cell_size:(cj+1)*cell_size]
                cell_angle = angle[ci*cell_size:(ci+1)*cell_size, cj*cell_size:(cj+1)*cell_size]
                
                # Histogram of oriented gradients
                hist, _ = np.histogram(
                    cell_angle.ravel(),
                    bins=n_bins,
                    range=(-np.pi, np.pi),
                    weights=cell_mag.ravel()
                )
                descriptor.extend(hist)
        
        # Normalize descriptor
        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        descriptors.append(descriptor)
    
    return np.array(descriptors)


def match_keypoints(
    kp1: np.ndarray,
    desc1: np.ndarray,
    kp2: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.75,
) -> tuple:
    """Match keypoints using descriptor similarity and ratio test.
    
    Parameters
    ----------
    kp1, kp2 : np.ndarray
        Keypoint coordinates (Nx2)
    desc1, desc2 : np.ndarray
        Descriptors (NxD)
    ratio_thresh : float
        Lowe's ratio test threshold
    
    Returns
    -------
    tuple of (matched_kp1, matched_kp2)
        Matched keypoint pairs
    """
    if len(kp1) == 0 or len(kp2) == 0 or len(desc1) == 0 or len(desc2) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    # Use BFMatcher with L2 norm (for float descriptors)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Find 2 best matches for ratio test
    try:
        matches = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    except cv2.error:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    # Apply ratio test
    good_matches = []
    for match in matches:
        if len(match) >= 2:
            m, n = match
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        elif len(match) == 1:
            # Only one match found, include it if descriptor distance is low
            if match[0].distance < 0.5:
                good_matches.append(match[0])
    
    if len(good_matches) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    # Extract matched keypoints
    matched_kp1 = np.array([kp1[m.queryIdx] for m in good_matches])
    matched_kp2 = np.array([kp2[m.trainIdx] for m in good_matches])
    
    return matched_kp1, matched_kp2


def ransac_filter(
    kp1: np.ndarray,
    kp2: np.ndarray,
    model: str = "affine",
    reproj_thresh: float = 5.0,
    confidence: float = 0.99,
    max_iters: int = 2000,
) -> tuple:
    """Filter matches using RANSAC to find geometrically consistent correspondences.
    
    Parameters
    ----------
    kp1, kp2 : np.ndarray
        Matched keypoint pairs (Nx2)
    model : str
        Transform model: "affine", "homography", or "rigid"
    reproj_thresh : float
        RANSAC reprojection threshold in pixels
    confidence : float
        RANSAC confidence level
    max_iters : int
        Maximum RANSAC iterations
    
    Returns
    -------
    tuple of (filtered_kp1, filtered_kp2, transform_matrix)
        Inlier keypoints and estimated transform
    """
    if len(kp1) < 4:
        return kp1, kp2, None
    
    # Convert to (x, y) format for OpenCV
    pts1 = kp1[:, ::-1].astype(np.float32)  # (row, col) -> (x, y)
    pts2 = kp2[:, ::-1].astype(np.float32)
    
    if model == "homography":
        M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh, 
                                      confidence=confidence, maxIters=max_iters)
    elif model == "affine":
        M, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC,
                                        ransacReprojThreshold=reproj_thresh,
                                        confidence=confidence, maxIters=max_iters)
    else:  # rigid
        M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                               ransacReprojThreshold=reproj_thresh,
                                               confidence=confidence, maxIters=max_iters)
    
    if mask is None:
        return kp1, kp2, M
    
    mask = mask.ravel().astype(bool)
    return kp1[mask], kp2[mask], M


def auto_detect_landmarks(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    detect_endpoints: bool = True,
    detect_bifurcations: bool = True,
    ransac_model: str = "affine",
    ransac_thresh: float = 10.0,
    min_matches: int = 4,
    ratio_thresh: float = 0.75,
) -> tuple:
    """Automatically detect and match landmarks between two images.
    
    Parameters
    ----------
    fixed_image : np.ndarray
        Fixed/target image
    moving_image : np.ndarray
        Moving/source image
    detect_endpoints : bool
        Include skeleton endpoints
    detect_bifurcations : bool
        Include bifurcation points
    ransac_model : str
        RANSAC model: "affine", "homography", or "rigid"
    ransac_thresh : float
        RANSAC reprojection threshold in pixels
    min_matches : int
        Minimum number of matches required
    ratio_thresh : float
        Lowe's ratio test threshold (0.5-0.95). Higher = more matches
    
    Returns
    -------
    tuple of (fixed_points, moving_points, info_dict)
        Matched landmark pairs and detection info
    """
    info = {}
    
    # Preprocess images
    binary_fixed = preprocess_mip(fixed_image)
    binary_moving = preprocess_mip(moving_image)
    
    # Extract keypoints
    kp_fixed = extract_skeleton_keypoints(binary_fixed, detect_endpoints, detect_bifurcations)
    kp_moving = extract_skeleton_keypoints(binary_moving, detect_endpoints, detect_bifurcations)
    
    info['keypoints_fixed'] = len(kp_fixed)
    info['keypoints_moving'] = len(kp_moving)
    
    if len(kp_fixed) < min_matches or len(kp_moving) < min_matches:
        info['error'] = f"Not enough keypoints detected (fixed: {len(kp_fixed)}, moving: {len(kp_moving)})"
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), info
    
    # Compute descriptors
    desc_fixed = compute_local_descriptors(fixed_image, kp_fixed)
    desc_moving = compute_local_descriptors(moving_image, kp_moving)
    
    # Match keypoints with ratio threshold
    matched_fixed, matched_moving = match_keypoints(
        kp_fixed, desc_fixed, kp_moving, desc_moving, ratio_thresh=ratio_thresh
    )
    info['matches_before_ransac'] = len(matched_fixed)
    
    if len(matched_fixed) < min_matches:
        info['error'] = f"Not enough matches (found {len(matched_fixed)}, need {min_matches})"
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), info
    
    # RANSAC filtering
    filtered_fixed, filtered_moving, transform = ransac_filter(
        matched_fixed, matched_moving, model=ransac_model, reproj_thresh=ransac_thresh
    )
    
    info['matches_after_ransac'] = len(filtered_fixed)
    info['transform_matrix'] = transform.tolist() if transform is not None else None
    
    return filtered_fixed, filtered_moving, info
