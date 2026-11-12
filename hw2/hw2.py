from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "hw2"
OUTPUT_DIR = ROOT / "output" / "task3"
REFERENCE_PATH = IMAGE_DIR / "source.png"


# ============================================================
# Experiment parameters
# ============================================================

LOWE_RATIOS = (0.60, 0.65, 0.70, 0.75)
RANSAC_THRESHOLDS = (3.0, 5.0, 8.0)

BASELINE_RATIO = 0.75
BASELINE_RANSAC = 5.0

# Normal Homography decision thresholds
MIN_STANDARD_INLIERS = 8
MIN_STANDARD_INLIER_RATIO = 0.40

# Affine fallback thresholds
MIN_AFFINE_MATCHES = 20
MIN_AFFINE_INLIERS = 12
MIN_AFFINE_INLIER_RATIO = 0.25

# Ground truth used only for evaluation
POSITIVE_IMAGES = {
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
}


# ============================================================
# Image input and output
# ============================================================

def read_image(path):
    """
    Read an image from a path that may contain non-ASCII
    characters.
    """
    image_data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {path}"
        )

    return image


def save_image(path, image):
    """
    Save an image to a path that may contain non-ASCII
    characters.
    """
    success, encoded_image = cv2.imencode(
        path.suffix,
        image,
    )

    if not success:
        raise OSError(
            f"Unable to save image: {path}"
        )

    encoded_image.tofile(str(path))


# ============================================================
# Boundary validation
# ============================================================

def validate_boundary(
    boundary,
    scene_shape,
    maximum_area_ratio=0.90,
):
    """
    Check whether a projected boundary is geometrically valid.
    """
    if boundary is None:
        return False, 0.0

    if not np.isfinite(boundary).all():
        return False, 0.0

    scene_height, scene_width = scene_shape[:2]

    boundary_area = abs(
        cv2.contourArea(boundary)
    )

    scene_area = scene_height * scene_width
    area_ratio = boundary_area / scene_area

    x_coordinates = boundary[:, :, 0]
    y_coordinates = boundary[:, :, 1]

    boundary_inside_image = bool(
        (x_coordinates >= 0).all()
        and (x_coordinates < scene_width).all()
        and (y_coordinates >= 0).all()
        and (y_coordinates < scene_height).all()
    )

    boundary_is_convex = bool(
        cv2.isContourConvex(boundary)
    )

    area_is_valid = bool(
        0.001 < area_ratio < maximum_area_ratio
    )

    boundary_valid = bool(
        boundary_inside_image
        and boundary_is_convex
        and area_is_valid
    )

    return boundary_valid, area_ratio


# ============================================================
# Homography localization
# ============================================================

def project_homography_boundary(
    homography,
    reference_shape,
    scene_shape,
):
    """
    Project all four corners of the reference image using
    Homography.
    """
    if homography is None:
        return None, False, 0.0

    reference_height, reference_width = (
        reference_shape[:2]
    )

    reference_corners = np.float32(
        [
            [0, 0],
            [reference_width - 1, 0],
            [
                reference_width - 1,
                reference_height - 1,
            ],
            [0, reference_height - 1],
        ]
    ).reshape(-1, 1, 2)

    projected_corners = (
        cv2.perspectiveTransform(
            reference_corners,
            homography,
        )
    )

    boundary_valid, area_ratio = (
        validate_boundary(
            projected_corners,
            scene_shape,
            maximum_area_ratio=0.90,
        )
    )

    return (
        projected_corners,
        boundary_valid,
        area_ratio,
    )


# ============================================================
# Affine fallback localization
# ============================================================

def estimate_affine_fallback(
    reference_points,
    scene_points,
    reference_shape,
    scene_shape,
    ransac_threshold,
):
    """
    Estimate an affine transformation when Homography does not
    produce a valid object boundary.
    """
    if reference_points is None:
        return None, False, 0, 0.0, 0.0

    if scene_points is None:
        return None, False, 0, 0.0, 0.0

    if len(reference_points) < 3:
        return None, False, 0, 0.0, 0.0

    cv2.setRNGSeed(42)

    affine_matrix, affine_mask = (
        cv2.estimateAffine2D(
            reference_points,
            scene_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=(
                ransac_threshold
            ),
            maxIters=3000,
            confidence=0.99,
            refineIters=10,
        )
    )

    if affine_matrix is None:
        return None, False, 0, 0.0, 0.0

    if affine_mask is None:
        return None, False, 0, 0.0, 0.0

    affine_inliers = int(
        affine_mask.sum()
    )

    affine_inlier_ratio = (
        affine_inliers
        / len(reference_points)
    )

    reference_height, reference_width = (
        reference_shape[:2]
    )

    reference_corners = np.float32(
        [
            [0, 0],
            [reference_width - 1, 0],
            [
                reference_width - 1,
                reference_height - 1,
            ],
            [0, reference_height - 1],
        ]
    ).reshape(-1, 1, 2)

    affine_boundary = cv2.transform(
        reference_corners,
        affine_matrix,
    )

    affine_valid, area_ratio = (
        validate_boundary(
            affine_boundary,
            scene_shape,
            maximum_area_ratio=0.50,
        )
    )

    return (
        affine_boundary,
        affine_valid,
        affine_inliers,
        affine_inlier_ratio,
        area_ratio,
    )


# ============================================================
# Object detection
# ============================================================

def detect_object(
    reference_keypoints,
    reference_descriptors,
    scene_keypoints,
    scene_descriptors,
    reference_shape,
    scene_shape,
    matcher,
    ratio_threshold,
    ransac_threshold,
):
    """
    Perform feature matching, Lowe ratio filtering, Homography
    estimation, and affine fallback localization.
    """
    if scene_descriptors is None:
        matching_pairs = []
    else:
        matching_pairs = matcher.knnMatch(
            reference_descriptors,
            scene_descriptors,
            k=2,
        )

    good_matches = []

    for pair in matching_pairs:
        if len(pair) != 2:
            continue

        best_match, second_match = pair

        if (
            best_match.distance
            < ratio_threshold
            * second_match.distance
        ):
            good_matches.append(best_match)

    reference_points = None
    scene_points = None
    homography = None
    homography_mask = None

    if len(good_matches) >= 4:
        reference_points = np.float32(
            [
                reference_keypoints[
                    match.queryIdx
                ].pt
                for match in good_matches
            ]
        )

        scene_points = np.float32(
            [
                scene_keypoints[
                    match.trainIdx
                ].pt
                for match in good_matches
            ]
        )

        cv2.setRNGSeed(42)

        homography, homography_mask = (
            cv2.findHomography(
                reference_points,
                scene_points,
                cv2.RANSAC,
                ransac_threshold,
            )
        )

    if homography_mask is None:
        homography_inliers = 0
    else:
        homography_inliers = int(
            homography_mask.sum()
        )

    if good_matches:
        homography_inlier_ratio = (
            homography_inliers
            / len(good_matches)
        )
    else:
        homography_inlier_ratio = 0.0

    (
        homography_boundary,
        homography_valid,
        homography_area_ratio,
    ) = project_homography_boundary(
        homography,
        reference_shape,
        scene_shape,
    )

    normal_detection = bool(
        homography_valid
        and homography_inliers
        >= MIN_STANDARD_INLIERS
        and homography_inlier_ratio
        >= MIN_STANDARD_INLIER_RATIO
    )

    affine_boundary = None
    affine_valid = False
    affine_inliers = 0
    affine_inlier_ratio = 0.0
    affine_area_ratio = 0.0

    should_try_affine = bool(
        not normal_detection
        and len(good_matches)
        >= MIN_AFFINE_MATCHES
    )

    if should_try_affine:
        (
            affine_boundary,
            affine_valid,
            affine_inliers,
            affine_inlier_ratio,
            affine_area_ratio,
        ) = estimate_affine_fallback(
            reference_points,
            scene_points,
            reference_shape,
            scene_shape,
            ransac_threshold,
        )

    affine_detection = bool(
        affine_valid
        and affine_inliers
        >= MIN_AFFINE_INLIERS
        and affine_inlier_ratio
        >= MIN_AFFINE_INLIER_RATIO
    )

    if normal_detection:
        prediction = "Found"
        final_boundary = homography_boundary
        localization_method = "Homography"
        final_area_ratio = (
            homography_area_ratio
        )

    elif affine_detection:
        prediction = "Found"
        final_boundary = affine_boundary
        localization_method = (
            "Affine fallback"
        )
        final_area_ratio = affine_area_ratio

    else:
        prediction = "Not Found"
        final_boundary = None
        localization_method = "None"
        final_area_ratio = (
            homography_area_ratio
        )

    return {
        "matches": len(good_matches),
        "homography_inliers": (
            homography_inliers
        ),
        "homography_inlier_ratio": round(
            homography_inlier_ratio,
            3,
        ),
        "homography_valid": (
            homography_valid
        ),
        "affine_inliers": affine_inliers,
        "affine_inlier_ratio": round(
            affine_inlier_ratio,
            3,
        ),
        "affine_valid": affine_valid,
        "localization_method": (
            localization_method
        ),
        "projected_area_ratio": round(
            final_area_ratio,
            4,
        ),
        "prediction": prediction,
        "boundary": final_boundary,
    }


# ============================================================
# Console output
# ============================================================

def print_image_result(
    index,
    total,
    result,
):
    status = (
        "PASS"
        if result["correct"]
        else "FAIL"
    )

    print(
        f"\n[{index}/{total}] "
        f"{result['image']}"
    )

    print(
        f"  Scene keypoints     : "
        f"{result['keypoints']}"
    )

    print(
        f"  Good matches        : "
        f"{result['matches']}"
    )

    print(
        f"  Homography inliers  : "
        f"{result['homography_inliers']}"
    )

    print(
        f"  Homography ratio    : "
        f"{result['homography_inlier_ratio']:.3f}"
    )

    print(
        f"  Homography valid    : "
        f"{result['homography_valid']}"
    )

    print(
        f"  Affine inliers      : "
        f"{result['affine_inliers']}"
    )

    print(
        f"  Affine ratio        : "
        f"{result['affine_inlier_ratio']:.3f}"
    )

    print(
        f"  Affine valid        : "
        f"{result['affine_valid']}"
    )

    print(
        f"  Localization method : "
        f"{result['localization_method']}"
    )

    print(
        f"  Projected area      : "
        f"{result['projected_area_ratio']:.4f}"
    )

    print(
        f"  Prediction          : "
        f"{result['prediction']}"
    )

    print(
        f"  Expected            : "
        f"{result['expected']}"
    )

    print(
        f"  Result              : "
        f"{status}"
    )


def print_parameter_comparison(
    rows,
    image_count,
):
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON")
    print("=" * 70)

    print(
        f"{'Lowe ratio':<14}"
        f"{'RANSAC px':<14}"
        f"{'Correct':<14}"
        f"{'Accuracy'}"
    )

    print("-" * 70)

    summaries = []

    for ratio in LOWE_RATIOS:
        for threshold in (
            RANSAC_THRESHOLDS
        ):
            group = [
                row
                for row in rows
                if row["ratio"] == ratio
                and row["ransac_px"]
                == threshold
            ]

            correct_count = sum(
                row["correct"]
                for row in group
            )

            accuracy = (
                correct_count / image_count
            )

            summaries.append(
                (
                    accuracy,
                    ratio,
                    threshold,
                )
            )

            print(
                f"{ratio:<14.2f}"
                f"{threshold:<14.1f}"
                f"{f'{correct_count}/{image_count}':<14}"
                f"{accuracy * 100:>6.1f}%"
            )

    best_accuracy = max(
        summary[0]
        for summary in summaries
    )

    best_settings = [
        summary
        for summary in summaries
        if summary[0] == best_accuracy
    ]

    settings_text = ", ".join(
        f"ratio={ratio:.2f}, "
        f"RANSAC={threshold:.1f}px"
        for _, ratio, threshold
        in best_settings
    )

    print("-" * 70)

    print(
        f"Best observed accuracy: "
        f"{best_accuracy * 100:.1f}%"
    )

    print(
        f"Best setting(s): "
        f"{settings_text}"
    )


# ============================================================
# Main experiment
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_paths = sorted(
        path
        for path in IMAGE_DIR.glob("test*")
        if path.suffix.lower()
        in {".png", ".jpg", ".jpeg"}
    )

    if not test_paths:
        raise FileNotFoundError(
            f"No test images found in: "
            f"{IMAGE_DIR}"
        )

    reference_image = read_image(
        REFERENCE_PATH
    )

    reference_gray = cv2.cvtColor(
        reference_image,
        cv2.COLOR_BGR2GRAY,
    )

    sift = cv2.SIFT_create()

    (
        reference_keypoints,
        reference_descriptors,
    ) = sift.detectAndCompute(
        reference_gray,
        None,
    )

    if reference_descriptors is None:
        raise RuntimeError(
            "No SIFT features were found "
            "in the reference image."
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    print("=" * 70)
    print(
        "TASK 3 OBJECT DETECTION EXPERIMENT"
    )
    print("=" * 70)

    print(
        f"OpenCV version       : "
        f"{cv2.__version__}"
    )

    print(
        f"Reference image      : "
        f"{REFERENCE_PATH.name}"
    )

    print(
        f"Reference keypoints  : "
        f"{len(reference_keypoints)}"
    )

    print(
        f"Test images          : "
        f"{len(test_paths)}"
    )

    print(
        f"Baseline settings    : "
        f"ratio={BASELINE_RATIO}, "
        f"RANSAC={BASELINE_RANSAC}px"
    )

    all_rows = []
    baseline_rows = []

    for index, test_path in enumerate(
        test_paths,
        start=1,
    ):
        scene_image = read_image(
            test_path
        )

        scene_gray = cv2.cvtColor(
            scene_image,
            cv2.COLOR_BGR2GRAY,
        )

        (
            scene_keypoints,
            scene_descriptors,
        ) = sift.detectAndCompute(
            scene_gray,
            None,
        )

        expected_result = (
            "Found"
            if test_path.stem
            in POSITIVE_IMAGES
            else "Not Found"
        )

        for ratio in LOWE_RATIOS:
            for threshold in (
                RANSAC_THRESHOLDS
            ):
                detection = detect_object(
                    reference_keypoints,
                    reference_descriptors,
                    scene_keypoints,
                    scene_descriptors,
                    reference_image.shape,
                    scene_image.shape,
                    matcher,
                    ratio,
                    threshold,
                )

                row = {
                    "image": test_path.name,
                    "ratio": ratio,
                    "ransac_px": threshold,
                    "keypoints": len(
                        scene_keypoints
                    ),
                    "matches": detection[
                        "matches"
                    ],
                    "homography_inliers": (
                        detection[
                            "homography_inliers"
                        ]
                    ),
                    "homography_inlier_ratio": (
                        detection[
                            "homography_inlier_ratio"
                        ]
                    ),
                    "homography_valid": (
                        detection[
                            "homography_valid"
                        ]
                    ),
                    "affine_inliers": (
                        detection[
                            "affine_inliers"
                        ]
                    ),
                    "affine_inlier_ratio": (
                        detection[
                            "affine_inlier_ratio"
                        ]
                    ),
                    "affine_valid": (
                        detection[
                            "affine_valid"
                        ]
                    ),
                    "localization_method": (
                        detection[
                            "localization_method"
                        ]
                    ),
                    "projected_area_ratio": (
                        detection[
                            "projected_area_ratio"
                        ]
                    ),
                    "prediction": detection[
                        "prediction"
                    ],
                    "expected": (
                        expected_result
                    ),
                    "correct": (
                        detection["prediction"]
                        == expected_result
                    ),
                }

                all_rows.append(row)

                is_baseline = bool(
                    ratio == BASELINE_RATIO
                    and threshold
                    == BASELINE_RANSAC
                )

                if not is_baseline:
                    continue

                baseline_rows.append(row)

                print_image_result(
                    index,
                    len(test_paths),
                    row,
                )

                result_image = (
                    scene_image.copy()
                )

                if (
                    detection["prediction"]
                    == "Found"
                    and detection["boundary"]
                    is not None
                ):
                    cv2.polylines(
                        result_image,
                        [
                            np.int32(
                                detection[
                                    "boundary"
                                ]
                            )
                        ],
                        True,
                        (0, 255, 0),
                        3,
                    )

                if (
                    detection["prediction"]
                    == "Found"
                ):
                    text_color = (
                        0,
                        180,
                        0,
                    )
                else:
                    text_color = (
                        0,
                        0,
                        255,
                    )

                cv2.putText(
                    result_image,
                    detection["prediction"],
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color,
                    2,
                )

                cv2.putText(
                    result_image,
                    detection[
                        "localization_method"
                    ],
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )

                output_path = (
                    OUTPUT_DIR
                    / (
                        f"{test_path.stem}"
                        f"_result.jpg"
                    )
                )

                save_image(
                    output_path,
                    result_image,
                )

    correct_count = sum(
        row["correct"]
        for row in baseline_rows
    )

    accuracy = (
        correct_count
        / len(baseline_rows)
    )

    true_positive = sum(
        row["prediction"] == "Found"
        and row["expected"] == "Found"
        for row in baseline_rows
    )

    true_negative = sum(
        row["prediction"]
        == "Not Found"
        and row["expected"]
        == "Not Found"
        for row in baseline_rows
    )

    false_positive = sum(
        row["prediction"] == "Found"
        and row["expected"]
        == "Not Found"
        for row in baseline_rows
    )

    false_negative = sum(
        row["prediction"]
        == "Not Found"
        and row["expected"] == "Found"
        for row in baseline_rows
    )

    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    print(
        f"Correct images : "
        f"{correct_count}/"
        f"{len(baseline_rows)}"
    )

    print(
        f"Accuracy       : "
        f"{accuracy * 100:.1f}%"
    )

    print(
        f"True positive  : "
        f"{true_positive}"
    )

    print(
        f"True negative  : "
        f"{true_negative}"
    )

    print(
        f"False positive : "
        f"{false_positive}"
    )

    print(
        f"False negative : "
        f"{false_negative}"
    )

    print_parameter_comparison(
        all_rows,
        len(test_paths),
    )

    output_data = {
        "opencv_version": (
            cv2.__version__
        ),
        "reference_image": (
            REFERENCE_PATH.name
        ),
        "reference_keypoints": len(
            reference_keypoints
        ),
        "baseline_settings": {
            "lowe_ratio": (
                BASELINE_RATIO
            ),
            "ransac_threshold_px": (
                BASELINE_RANSAC
            ),
            "minimum_standard_inliers": (
                MIN_STANDARD_INLIERS
            ),
            "minimum_standard_ratio": (
                MIN_STANDARD_INLIER_RATIO
            ),
            "minimum_affine_matches": (
                MIN_AFFINE_MATCHES
            ),
            "minimum_affine_inliers": (
                MIN_AFFINE_INLIERS
            ),
            "minimum_affine_ratio": (
                MIN_AFFINE_INLIER_RATIO
            ),
        },
        "baseline_summary": {
            "correct": correct_count,
            "total": len(
                baseline_rows
            ),
            "accuracy": round(
                accuracy,
                4,
            ),
            "true_positive": (
                true_positive
            ),
            "true_negative": (
                true_negative
            ),
            "false_positive": (
                false_positive
            ),
            "false_negative": (
                false_negative
            ),
        },
        "experiments": all_rows,
    }

    json_path = (
        OUTPUT_DIR / "results.json"
    )

    json_path.write_text(
        json.dumps(
            output_data,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Result images  : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Experiment data: "
        f"{json_path}"
    )

    print(
        "Experiment completed successfully."
    )


if __name__ == "__main__":
    main()