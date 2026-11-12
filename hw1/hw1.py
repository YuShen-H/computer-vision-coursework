
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Image resize
MAX_IMAGE_WIDTH = 1200

# Brightness / Exposure Adjustment
BRIGHTNESS_LOW_THRESHOLD = 80
BRIGHTNESS_HIGH_THRESHOLD = 180
BRIGHTEN_GAMMA = 0.65
DARKEN_GAMMA = 1.40

# Noise Removal
GAUSSIAN_BLUR_KSIZE = (5, 5)
MEDIAN_BLUR_KSIZE = 5

# Edge Connection & Filtering
# Keep the original path relatively sensitive for dark/simple backgrounds.
MIN_EDGE_LENGTH = 50
EDGE_CLOSE_KERNEL = (5, 5)
EDGE_DILATE_KERNEL = (3, 3)
EDGE_DILATE_ITERATIONS = 2

# Extra edge branches for difficult backgrounds
USE_MULTI_SCALE_EDGES = True
SECONDARY_EDGE_MIN_LENGTH = 80
HEAVY_EDGE_MIN_LENGTH = 110
HEAVY_BLUR_KSIZE = (9, 9)

# Document Candidate
MIN_DOCUMENT_AREA_RATIO = 0.10
MAX_DOCUMENT_AREA_RATIO = 0.90
MIN_ASPECT_RATIO = 1.15
MAX_ASPECT_RATIO = 2.20
MIN_RECTANGULARITY = 0.65

# Candidate scoring
TARGET_ASPECT_RATIO = 1.414
EDGE_SUPPORT_BAND = 4
MIN_EDGE_SUPPORT = 0.10
BORDER_MARGIN_RATIO = 0.03
SUSPICIOUS_BORDER_POINTS = 1
FALLBACK_SCORE_THRESHOLD = 0.72

# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Threshold
USE_ADAPTIVE_THRESHOLD = True
ADAPTIVE_BLOCK_SIZE = 21
ADAPTIVE_C = 40
THRESHOLD_VALUE = 127

# GrabCut fallback
# It is intentionally used only when the edge solution looks suspicious.
USE_GRABCUT_FALLBACK = True
GRABCUT_BORDER_RATIO = 0.08
GRABCUT_ITERATIONS = 5
GRABCUT_MIN_AREA_RATIO = 0.15


# ============================================================
# Utility Functions
# ============================================================

def resize_image(image):
    """Resize image while preserving aspect ratio."""
    height, width = image.shape[:2]

    if width <= MAX_IMAGE_WIDTH:
        return image

    scale = MAX_IMAGE_WIDTH / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# Point Ordering
# ============================================================

def order_points(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.array(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]

    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]

    return ordered


# ============================================================
# Perspective Transformation
# ============================================================

def four_point_transform(image, points):
    """Perform perspective transformation using four corners."""
    rect = order_points(points)

    top_left = rect[0]
    top_right = rect[1]
    bottom_right = rect[2]
    bottom_left = rect[3]

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    max_height = int(max(height_right, height_left))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


# ============================================================
# Pipeline Step 1: Grayscale Conversion
# ============================================================

def grayscale_conversion(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# Pipeline Step 2: Brightness Adjustment
# ============================================================

def brightness_adjustment(gray):
    mean_brightness = np.mean(gray)

    if mean_brightness < BRIGHTNESS_LOW_THRESHOLD:
        gamma = BRIGHTEN_GAMMA
        normalized = gray.astype(np.float32) / 255.0
        adjusted = np.power(normalized, gamma) * 255.0
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        action = "Brightened"
    elif mean_brightness > BRIGHTNESS_HIGH_THRESHOLD:
        gamma = DARKEN_GAMMA
        normalized = gray.astype(np.float32) / 255.0
        adjusted = np.power(normalized, gamma) * 255.0
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        action = "Darkened"
    else:
        adjusted = gray.copy()
        action = "No adjustment"

    return adjusted, mean_brightness, action


# ============================================================
# Pipeline Step 3: Noise Removal
# ============================================================

def noise_removal(gray):
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KSIZE, 0)
    filtered = cv2.medianBlur(blurred, MEDIAN_BLUR_KSIZE)
    return filtered


# ============================================================
# Pipeline Step 4: Edge Detection
# ============================================================

def edge_detection(filtered):
    """Original Otsu-driven Canny path."""
    high_thresh, _ = cv2.threshold(
        filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low_thresh = 0.5 * high_thresh
    return cv2.Canny(filtered, int(low_thresh), int(high_thresh))


def build_edge_variants(filtered, gray):
    """
    Keep multiple edge paths:
      1. original edge path -> good for dark/simple scenes
      2. CLAHE edge path    -> helps low-contrast document boundaries
      3. heavy blur path    -> suppresses small background texture
    """
    variants = [("normal", edge_detection(filtered))]

    if not USE_MULTI_SCALE_EDGES:
        return variants

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    ).apply(gray)
    clahe_filtered = cv2.GaussianBlur(clahe, (7, 7), 0)
    variants.append(("clahe", edge_detection(clahe_filtered)))

    heavy_filtered = cv2.GaussianBlur(filtered, HEAVY_BLUR_KSIZE, 0)
    variants.append(("heavy", edge_detection(heavy_filtered)))

    return variants


# ============================================================
# Geometry Functions
# ============================================================

def calculate_angle(point_a, point_b, point_c):
    vector1 = point_a - point_b
    vector2 = point_c - point_b

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def calculate_angle_score(points):
    points = order_points(points)

    angles = []
    for i in range(4):
        point_a = points[(i - 1) % 4]
        point_b = points[i]
        point_c = points[(i + 1) % 4]
        angles.append(calculate_angle(point_a, point_b, point_c))

    errors = [abs(angle - 90.0) for angle in angles]
    mean_error = np.mean(errors)

    return max(0.0, 1.0 - mean_error / 45.0)


def calculate_aspect_ratio(points):
    points = order_points(points)

    side_lengths = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(p2 - p1))

    short_side = min(side_lengths)
    long_side = max(side_lengths)

    if short_side == 0:
        return 999.0

    return long_side / short_side


def calculate_rectangularity(points):
    area = abs(cv2.contourArea(points))
    rect = cv2.minAreaRect(points)
    width, height = rect[1][0], rect[1][1]

    if width <= 0 or height <= 0:
        return 0.0

    rect_area = width * height

    if rect_area == 0:
        return 0.0

    return area / rect_area


# ============================================================
# Candidate Quality Features
# ============================================================

def calculate_edge_support(edges, points, band=4):
    """
    Measures how continuously the image edge follows all four
    document sides. This is important on a fancy background:
    random texture may form a quadrilateral, but usually does not
    produce four continuous edges.
    """
    points = order_points(points).astype(np.float32)

    height, width = edges.shape[:2]
    support_values = []

    offsets = [(dx, dy)
               for dx in range(-band, band + 1)
               for dy in range(-band, band + 1)]

    for i in range(4):
        a = points[i]
        b = points[(i + 1) % 4]

        length = np.linalg.norm(b - a)
        sample_count = max(40, min(500, int(length)))

        t = np.linspace(0.0, 1.0, sample_count)
        xs = a[0] + (b[0] - a[0]) * t
        ys = a[1] + (b[1] - a[1]) * t

        hit = np.zeros(sample_count, dtype=bool)

        for dx, dy in offsets:
            xx = np.round(xs + dx).astype(np.int32)
            yy = np.round(ys + dy).astype(np.int32)

            valid = (
                (xx >= 0) & (xx < width) &
                (yy >= 0) & (yy < height)
            )

            hit[valid] |= edges[yy[valid], xx[valid]] > 0

        support_values.append(float(np.mean(hit)))

    return float(np.mean(support_values)), support_values


def calculate_border_penalty(points, width, height):
    """
    Stronger than the original border score.
    A false quadrilateral created by floor/wall/background lines
    often touches the image boundary.
    """
    margin = min(width, height) * BORDER_MARGIN_RATIO
    points = order_points(points)

    count = sum(
        1 for x, y in points
        if x < margin or x > width - margin or
           y < margin or y > height - margin
    )

    return count


def calculate_intensity_contrast(gray, points):
    """
    Compare a narrow inside band with a narrow outside band around
    each side. A real page boundary often has a consistent
    intensity transition; random background lines are less stable.
    """
    points = order_points(points).astype(np.float32)

    h, w = gray.shape
    values = []

    for i in range(4):
        a = points[i]
        b = points[(i + 1) % 4]

        length = np.linalg.norm(b - a)
        sample_count = max(40, min(400, int(length)))
        t = np.linspace(0.05, 0.95, sample_count)

        xs = a[0] + (b[0] - a[0]) * t
        ys = a[1] + (b[1] - a[1]) * t

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        norm = max(np.hypot(dx, dy), 1e-6)

        # Left-hand normal.
        nx = -dy / norm
        ny = dx / norm

        inside_x = np.round(xs - nx * 8).astype(np.int32)
        inside_y = np.round(ys - ny * 8).astype(np.int32)

        outside_x = np.round(xs + nx * 8).astype(np.int32)
        outside_y = np.round(ys + ny * 8).astype(np.int32)

        valid_in = (
            (inside_x >= 0) & (inside_x < w) &
            (inside_y >= 0) & (inside_y < h)
        )
        valid_out = (
            (outside_x >= 0) & (outside_x < w) &
            (outside_y >= 0) & (outside_y < h)
        )

        if valid_in.sum() < 10 or valid_out.sum() < 10:
            values.append(0.0)
            continue

        in_mean = float(np.mean(gray[inside_y[valid_in], inside_x[valid_in]]))
        out_mean = float(np.mean(gray[outside_y[valid_out], outside_x[valid_out]]))

        # 0~1 normalization. 64 gray levels is already a strong
        # average transition for a document boundary.
        contrast = min(abs(in_mean - out_mean) / 64.0, 1.0)
        values.append(contrast)

    return float(np.mean(values))


def score_candidate(points, edges, original_image):
    """
    Unified candidate score.
    Geometry remains important, but boundary continuity and
    frame-touching are added to reduce false positives.
    """
    image_height, image_width = original_image.shape[:2]
    image_area = image_height * image_width

    candidate_area = abs(cv2.contourArea(points))
    area_ratio = candidate_area / image_area

    if area_ratio < MIN_DOCUMENT_AREA_RATIO:
        return None

    if area_ratio > MAX_DOCUMENT_AREA_RATIO:
        return None

    aspect_ratio = calculate_aspect_ratio(points)
    if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
        return None

    rectangularity = calculate_rectangularity(points)
    if rectangularity < MIN_RECTANGULARITY:
        return None

    angle_score = calculate_angle_score(points)

    if area_ratio < 0.15:
        area_score = area_ratio / 0.15
    elif area_ratio <= 0.60:
        area_score = 1.0
    else:
        area_score = max(
            0.0,
            1.0 - (area_ratio - 0.60) / 0.30
        )

    center_x, center_y = np.mean(points[:, 0]), np.mean(points[:, 1])
    image_center_x, image_center_y = (
        image_width / 2.0,
        image_height / 2.0
    )

    normalized_distance = np.sqrt(
        ((center_x - image_center_x) / image_width) ** 2 +
        ((center_y - image_center_y) / image_height) ** 2
    )

    center_score = max(0.0, 1.0 - normalized_distance * 1.5)

    border_count = calculate_border_penalty(
        points, image_width, image_height
    )

    # New continuous-boundary feature.
    edge_support, side_support = calculate_edge_support(
        edges, points, EDGE_SUPPORT_BAND
    )

    intensity_contrast = calculate_intensity_contrast(
        cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY),
        points
    )

    aspect_score = max(
        0.0,
        1.0 - abs(aspect_ratio - TARGET_ASPECT_RATIO) /
        TARGET_ASPECT_RATIO
    )

    # Border points are not impossible, but a candidate touching the
    # image frame is suspicious. Penalize gradually instead of rejecting
    # it outright, so partially cropped pages can still work.
    border_score = max(0.0, 1.0 - 0.45 * border_count)

    score = (
        0.15 * area_score +
        0.15 * rectangularity +
        0.12 * aspect_score +
        0.13 * angle_score +
        0.08 * center_score +
        0.17 * edge_support +
        0.10 * intensity_contrast +
        0.10 * border_score
    )

    return {
        "score": float(score),
        "points": points.astype(np.float32),
        "area": float(candidate_area),
        "area_ratio": float(area_ratio),
        "aspect_ratio": float(aspect_ratio),
        "rectangularity": float(rectangularity),
        "angle_score": float(angle_score),
        "center_score": float(center_score),
        "border_count": int(border_count),
        "border_score": float(border_score),
        "edge_support": float(edge_support),
        "side_support": side_support,
        "intensity_contrast": float(intensity_contrast),
    }


# ============================================================
# Edge-Based Candidate Generation
# ============================================================

def extract_edge_candidates(edges, original_image, min_edge_length):
    image_height, image_width = original_image.shape[:2]
    image_area = image_height * image_width

    min_area = image_area * MIN_DOCUMENT_AREA_RATIO
    max_area = image_area * MAX_DOCUMENT_AREA_RATIO

    raw_contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    clean_edges = np.zeros_like(edges)

    # Keep long structures, but allow this threshold to vary by branch.
    for contour in raw_contours:
        if cv2.arcLength(contour, False) > min_edge_length:
            cv2.drawContours(
                clean_edges,
                [contour],
                -1,
                255,
                1
            )

    # Do not use an overly aggressive dilation for the heavy/texture path.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        EDGE_CLOSE_KERNEL
    )
    connected_edges = cv2.morphologyEx(
        clean_edges,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        EDGE_DILATE_KERNEL
    )
    connected_edges = cv2.dilate(
        connected_edges,
        dilate_kernel,
        iterations=EDGE_DILATE_ITERATIONS
    )

    contours, _ = cv2.findContours(
        connected_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area < min_area:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area < min_area or hull_area > max_area:
            continue

        perimeter = cv2.arcLength(hull, True)
        if perimeter == 0:
            continue

        for epsilon_ratio in [
            0.01, 0.015, 0.02, 0.025, 0.03,
            0.04, 0.05, 0.06, 0.08, 0.10
        ]:
            approximation = cv2.approxPolyDP(
                hull,
                epsilon_ratio * perimeter,
                True
            )

            if len(approximation) != 4:
                continue

            points = approximation.reshape(4, 2).astype(np.float32)
            scored = score_candidate(
                points,
                edges,
                original_image
            )

            if scored is not None:
                candidates.append(scored)

            break

    return candidates, connected_edges


# ============================================================
# GrabCut Fallback
# ============================================================

def grabcut_candidate(original_image, edge_reference):
    """
    Use GrabCut only as a fallback for busy backgrounds or suspicious
    edge solutions. The central rectangle is an intentional prior:
    the existing pipeline already favors centered documents.
    """
    if not USE_GRABCUT_FALLBACK:
        return None

    height, width = original_image.shape[:2]

    margin_x = int(width * GRABCUT_BORDER_RATIO)
    margin_y = int(height * GRABCUT_BORDER_RATIO)

    rect_w = width - 2 * margin_x
    rect_h = height - 2 * margin_y

    if rect_w <= 0 or rect_h <= 0:
        return None

    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            original_image,
            mask,
            (margin_x, margin_y, rect_w, rect_h),
            bgd_model,
            fgd_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_RECT
        )
    except cv2.error:
        return None

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Stabilize the segmentation before finding its outer boundary.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < original_image.shape[0] * original_image.shape[1] * GRABCUT_MIN_AREA_RATIO:
            continue

        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)

        if perimeter == 0:
            continue

        # GrabCut usually gives a very clean page silhouette, so a
        # slightly tighter epsilon range is enough.
        for epsilon_ratio in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
            approximation = cv2.approxPolyDP(
                hull,
                epsilon_ratio * perimeter,
                True
            )

            if len(approximation) != 4:
                continue

            points = approximation.reshape(4, 2).astype(np.float32)

            scored = score_candidate(
                points,
                edge_reference,
                original_image
            )

            if scored is not None:
                scored["source"] = "grabcut"
                return scored

    return None


# ============================================================
# Pipeline Step 5: Document Boundary Detection
# ============================================================

def document_boundary_detection(
    edges,
    original_image,
    gray=None,
    filtered=None
):
    """
    Hybrid detector:
      - Edge path is kept for dark / skewed / clean backgrounds.
      - CLAHE + stronger blur provide extra edge candidates.
      - GrabCut is only activated when the edge result looks suspicious.
    """
    if gray is None:
        gray = grayscale_conversion(original_image)

    if filtered is None:
        filtered = noise_removal(gray)

    edge_variants = build_edge_variants(filtered, gray)

    all_candidates = []
    connected_debug = edges.copy()

    for variant_name, variant_edges in edge_variants:
        if variant_name == "normal":
            min_edge_length = MIN_EDGE_LENGTH
        elif variant_name == "clahe":
            min_edge_length = SECONDARY_EDGE_MIN_LENGTH
        else:
            min_edge_length = HEAVY_EDGE_MIN_LENGTH

        candidates, connected = extract_edge_candidates(
            variant_edges,
            original_image,
            min_edge_length
        )

        for candidate in candidates:
            candidate["source"] = variant_name
            all_candidates.append(candidate)

        # Keep the normal branch for the saved intermediate image.
        if variant_name == "normal":
            connected_debug = connected

    if not all_candidates:
        best = None
    else:
        all_candidates.sort(
            key=lambda candidate: candidate["score"],
            reverse=True
        )
        best = all_candidates[0]

    # Decide whether the edge result is suspicious.
    suspicious = (
        best is None or
        best["score"] < FALLBACK_SCORE_THRESHOLD or
        best["border_count"] >= SUSPICIOUS_BORDER_POINTS or
        best["edge_support"] < MIN_EDGE_SUPPORT
    )

    if suspicious:
        print("       Edge solution looks suspicious -> trying GrabCut...")
        grabcut = grabcut_candidate(
            original_image,
            connected_debug
        )

        if grabcut is not None:
            print(
                f"       GrabCut candidate score: "
                f"{grabcut['score']:.3f}"
            )

            if best is None or grabcut["score"] > best["score"]:
                best = grabcut

    if best is None:
        return None, None

    print("       Best candidate:")
    print(f"       Source           : {best.get('source', 'edge')}")
    print(f"       Score            : {best['score']:.3f}")
    print(f"       Area ratio       : {best['area_ratio']:.3f}")
    print(f"       Aspect ratio     : {best['aspect_ratio']:.3f}")
    print(f"       Rectangularity   : {best['rectangularity']:.3f}")
    print(f"       Angle score      : {best['angle_score']:.3f}")
    print(f"       Edge support     : {best['edge_support']:.3f}")
    print(f"       Intensity contrast: {best['intensity_contrast']:.3f}")
    print(f"       Border points    : {best['border_count']}")

    boundary_image = original_image.copy()
    document_points = best["points"].astype(np.int32)

    cv2.drawContours(
        boundary_image,
        [document_points],
        -1,
        (0, 255, 0),
        5
    )

    for x, y in document_points:
        cv2.circle(
            boundary_image,
            (int(x), int(y)),
            10,
            (0, 0, 255),
            -1
        )

    cv2.putText(
        boundary_image,
        f"Score: {best['score']:.3f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    return document_points, boundary_image


# ============================================================
# Pipeline Step 6: Contrast Enhancement
# ============================================================

def contrast_enhancement(gray):
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE
    )
    return clahe.apply(gray)


# ============================================================
# Pipeline Step 7: Thresholding
# ============================================================

def thresholding(enhanced):
    if USE_ADAPTIVE_THRESHOLD:
        block_size = (
            ADAPTIVE_BLOCK_SIZE
            if ADAPTIVE_BLOCK_SIZE % 2 == 1
            else ADAPTIVE_BLOCK_SIZE + 1
        )

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            ADAPTIVE_C
        )
    else:
        _, binary = cv2.threshold(
            enhanced,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return binary


# ============================================================
# Save Results
# ============================================================

def save_results(
    output_dir,
    image_name,
    original,
    gray,
    brightness_adjusted,
    filtered,
    edges,
    boundary,
    warped,
    enhanced,
    binary
):
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem
    image_output_dir = output_dir / stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(image_output_dir / "01_original.jpg"),
        original
    )
    cv2.imwrite(
        str(image_output_dir / "02_grayscale.jpg"),
        gray
    )
    cv2.imwrite(
        str(image_output_dir / "03_brightness_adjustment.jpg"),
        brightness_adjusted
    )
    cv2.imwrite(
        str(image_output_dir / "04_noise_removal.jpg"),
        filtered
    )
    cv2.imwrite(
        str(image_output_dir / "05_edge_detection.jpg"),
        edges
    )

    if boundary is not None:
        cv2.imwrite(
            str(image_output_dir / "06_document_boundary.jpg"),
            boundary
        )

    if warped is not None:
        cv2.imwrite(
            str(image_output_dir / "07_perspective.jpg"),
            warped
        )

    if enhanced is not None:
        cv2.imwrite(
            str(image_output_dir / "08_contrast_enhancement.jpg"),
            enhanced
        )

    if binary is not None:
        cv2.imwrite(
            str(image_output_dir / "09_threshold_final.jpg"),
            binary
        )


# ============================================================
# Process One Image
# ============================================================

def process_image(image_path):
    print(
        f"\n{'=' * 70}\n"
        f"Processing: {image_path.name}\n"
        f"{'=' * 70}"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Cannot read image.")
        return False

    original = resize_image(image)

    gray = grayscale_conversion(original)
    print("[1/7] Grayscale conversion")

    brightness_adjusted, mean_brightness, action = (
        brightness_adjustment(gray)
    )
    print(
        f"[2/7] Brightness adjustment\n"
        f"       Mean brightness: {mean_brightness:.2f}\n"
        f"       Action: {action}"
    )

    filtered = noise_removal(brightness_adjusted)
    print("[3/7] Gaussian + Median noise removal")

    edges = edge_detection(filtered)
    print("[4/7] Canny edge detection (Auto-Otsu)")

    points, boundary = document_boundary_detection(
        edges,
        original,
        gray=gray,
        filtered=filtered
    )
    print("[5/7] Document boundary detection")

    if points is None:
        print("       WARNING: Document boundary NOT found.")

        save_results(
            OUTPUT_DIR,
            image_path.name,
            original,
            gray,
            brightness_adjusted,
            filtered,
            edges,
            None,
            None,
            None,
            None
        )
        return False

    print("       Document boundary found.")

    warped = four_point_transform(
        original,
        points
    )
    print("[6/7] Perspective transformation")

    warped_gray = grayscale_conversion(warped)
    enhanced = contrast_enhancement(warped_gray)
    print("       CLAHE contrast enhancement")

    binary = thresholding(enhanced)
    print("[7/7] Thresholding")

    save_results(
        OUTPUT_DIR,
        image_path.name,
        original,
        gray,
        brightness_adjusted,
        filtered,
        edges,
        boundary,
        warped,
        enhanced,
        binary
    )

    print("       Result saved.")
    return True


# ============================================================
# Main
# ============================================================

def main():
    print(
        f"{'=' * 70}\n"
        "OpenCV Enhanced Document Scanner - HW1 (Balanced)\n"
        f"{'=' * 70}"
    )

    if not INPUT_DIR.exists():
        print(
            f"\nERROR: Input directory '{INPUT_DIR}' does not exist.\n"
            "Please create the data folder and put images inside."
        )
        return

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]

    image_files = []

    for extension in extensions:
        image_files.extend(INPUT_DIR.glob(extension))

    image_files = sorted(
        list(set(image_files)),
        key=lambda x: x.name
    )

    if len(image_files) == 0:
        print("\nERROR: No images found in data folder.")
        return

    print(f"\nFound {len(image_files)} image(s).")

    success_count = sum(
        1
        for image_path in image_files
        if process_image(image_path)
    )

    print(
        f"\n{'=' * 70}\n"
        "Processing Summary\n"
        f"{'=' * 70}"
    )
    print(f"Total images : {len(image_files)}")
    print(f"Success      : {success_count}")
    print(f"Failed       : {len(image_files) - success_count}")
    print(f"Results      : {OUTPUT_DIR}\n{'=' * 70}")


if __name__ == "__main__":
    main()
