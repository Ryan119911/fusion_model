"""Paper-parameterised flexible hairy-brush footprint and image metrics.

The geometric model follows the B-BSM described in the accompanying paper:
the physical brush has a 48 mm bundle and a 6 mm radius, ``(h, alpha, beta)``
is mapped to ``(Lt, Lh, Lr)`` with the reported linear regressions, and two
symmetric cubic Bezier curves form the closed contact footprint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image


_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


@dataclass(frozen=True)
class BrushDimensions:
    """Lengths of the paper's standard brush footprint, in metres."""

    tip_length_m: float
    heel_length_m: float
    half_width_m: float


@dataclass(frozen=True)
class DynamicFootprintState:
    """Seven-state dynamic brush result plus its virtual B-BSM pose."""

    x_m: float
    y_m: float
    root_x_m: float
    root_y_m: float
    press_depth_mm: float
    half_width_m: float
    drag_length_m: float
    offset_m: float
    heading_rad: float
    virtual_h_mm: float
    virtual_alpha_rad: float
    virtual_beta_rad: float
    dimensions: BrushDimensions


@dataclass(frozen=True)
class PaperBrushParameters:
    """No. 1 brush and regression parameters reported by the paper."""

    bundle_length_m: float = 0.048
    bundle_radius_m: float = 0.006
    press_depth_min_mm: float = 11.0
    press_depth_max_mm: float = 20.0
    tilt_max_rad: float = math.radians(10.0)
    rotation_max_rad: float = math.radians(5.0)
    # Wang et al.'s dynamic virtual brush uses 0.02 for both inertia terms.
    width_inertia: float = 0.02
    drag_inertia: float = 0.02
    # Figure 9 reports Lt/Lh/Lr around 0.3--1.8 for a 6 mm-radius brush.
    # Those plotted regression lengths are therefore centimetres.
    regression_length_unit_m: float = 0.01
    lt_coefficients: Tuple[float, float, float, float] = (
        0.0672,
        0.0263,
        0.0191,
        0.0267,
    )
    lh_coefficients: Tuple[float, float, float, float] = (
        0.0196,
        0.0039,
        0.0073,
        0.0372,
    )
    lr_coefficients: Tuple[float, float, float, float] = (
        0.0239,
        0.0061,
        0.0096,
        0.1137,
    )

    def dimensions(
        self, press_depth_mm: float, alpha_rad: float, beta_rad: float
    ) -> BrushDimensions:
        """Evaluate Eq. (6), retaining the paper's radian angle convention."""

        values = (float(press_depth_mm), abs(float(alpha_rad)), abs(float(beta_rad)))

        def regress(coefficients: Sequence[float]) -> float:
            k1, k2, k3, epsilon = coefficients
            return max(
                0.0,
                (k1 * values[0] + k2 * values[1] + k3 * values[2] + epsilon)
                * self.regression_length_unit_m,
            )

        return BrushDimensions(
            tip_length_m=regress(self.lt_coefficients),
            heel_length_m=regress(self.lh_coefficients),
            half_width_m=regress(self.lr_coefficients),
        )

    def inverse_virtual_pose(
        self,
        dimensions: BrushDimensions,
        *,
        reference_pose: Sequence[float] | None = None,
        max_depth_mm: float | None = None,
        regularization: float = 1.0e-4,
    ) -> Tuple[float, float, float]:
        """Invert Eq. (6) with bounded ridge least squares.

        The published regression is nearly rank-deficient in the two tilt
        columns.  A plain unconstrained inverse therefore produces large
        negative angle values and, after taking absolute values, the wrong
        saturated pose.  The V17 inverse uses the same reference-regularized
        formulation; this NumPy implementation additionally enumerates the
        small set of bound-active solutions so the ROS path remains stable
        without depending on SciPy.
        """

        if regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        lower = np.array(
            [self.press_depth_min_mm, 0.0, 0.0], dtype=np.float64
        )
        upper_depth = self.press_depth_max_mm
        if max_depth_mm is not None:
            if not math.isfinite(float(max_depth_mm)):
                raise ValueError("max_depth_mm must be finite when provided")
            upper_depth = min(upper_depth, float(max_depth_mm))
        if upper_depth < lower[0]:
            raise ValueError(
                "max_depth_mm is smaller than the brush calibration minimum"
            )
        upper = np.array(
            [upper_depth, self.tilt_max_rad, self.rotation_max_rad],
            dtype=np.float64,
        )
        if reference_pose is None:
            reference = 0.5 * (lower + upper)
        else:
            reference = np.asarray(reference_pose, dtype=np.float64).reshape(-1)
            if reference.size != 3 or not np.all(np.isfinite(reference)):
                raise ValueError("reference_pose must contain three finite values")
            reference = np.abs(reference)
            reference = np.clip(reference, lower, upper)

        matrix = np.array(
            [
                self.lt_coefficients[:3],
                self.lh_coefficients[:3],
                self.lr_coefficients[:3],
            ],
            dtype=np.float64,
        )
        epsilon = np.array(
            [
                self.lt_coefficients[3],
                self.lh_coefficients[3],
                self.lr_coefficients[3],
            ],
            dtype=np.float64,
        )
        measured = np.asarray(
            [dimensions.tip_length_m, dimensions.heel_length_m, dimensions.half_width_m],
            dtype=np.float64,
        ) / self.regression_length_unit_m

        best: tuple[float, np.ndarray] | None = None
        # -1 = lower bound, 0 = free, +1 = upper bound.  Every convex
        # bounded least-squares optimum is represented by one of these active
        # sets; infeasible free solutions are discarded.
        import itertools

        for active in itertools.product((-1, 0, 1), repeat=3):
            fixed = [index for index, value in enumerate(active) if value]
            free = [index for index, value in enumerate(active) if not value]
            candidate = np.zeros(3, dtype=np.float64)
            for index in fixed:
                candidate[index] = (
                    lower[index] if active[index] < 0 else upper[index]
                )
            if free:
                design = matrix[:, free]
                target = measured - epsilon
                if fixed:
                    target = target - matrix[:, fixed] @ candidate[fixed]
                lhs = design.T @ design + float(regularization) * np.eye(len(free))
                rhs = design.T @ target + float(regularization) * reference[free]
                candidate[free] = np.linalg.solve(lhs, rhs)
            if np.any(candidate < lower - 1.0e-9) or np.any(
                candidate > upper + 1.0e-9
            ):
                continue
            residual = matrix @ candidate + epsilon - measured
            objective = float(
                residual @ residual
                + float(regularization)
                * ((candidate - reference) @ (candidate - reference))
            )
            if best is None or objective < best[0]:
                best = (objective, candidate)
        if best is None:
            raise RuntimeError("bounded brush-pose inversion found no feasible solution")
        solution = best[1]
        return float(solution[0]), float(solution[1]), float(solution[2])

        matrix = np.array(
            [
                self.lt_coefficients[:3],
                self.lh_coefficients[:3],
                self.lr_coefficients[:3],
            ],
            dtype=np.float64,
        )
        epsilon = np.array(
            [
                self.lt_coefficients[3],
                self.lh_coefficients[3],
                self.lr_coefficients[3],
            ],
            dtype=np.float64,
        )
        measured = np.array(
            [
                dimensions.tip_length_m,
                dimensions.heel_length_m,
                dimensions.half_width_m,
            ],
            dtype=np.float64,
        ) / self.regression_length_unit_m
        solution = np.linalg.lstsq(matrix, measured - epsilon, rcond=None)[0]
        return (
            float(np.clip(solution[0], self.press_depth_min_mm, self.press_depth_max_mm)),
            float(np.clip(abs(solution[1]), 0.0, self.tilt_max_rad)),
            float(np.clip(abs(solution[2]), 0.0, self.rotation_max_rad)),
        )

    def point_in_calibrated_domain(
        self, press_depth_mm: float, alpha_rad: float, beta_rad: float
    ) -> bool:
        """Return whether a contact pose is inside the paper's sampled domain."""

        tolerance = 1.0e-9
        return (
            self.press_depth_min_mm - tolerance
            <= press_depth_mm
            <= self.press_depth_max_mm + tolerance
            and abs(alpha_rad) <= self.tilt_max_rad + tolerance
            and abs(beta_rad) <= self.rotation_max_rad + tolerance
        )


class FlexibleBrushModel:
    """Dynamic-model to virtual-pose to B-BSM footprint bridge.

    ``w`` and ``d`` follow the discrete inertial updates from Wang et al.;
    their targets come from the paper-parameterised B-BSM regression.  The
    friction/snap offset uses the previous root position and the physical
    48 mm bundle as its hard deformation bound.
    """

    def __init__(self, parameters: PaperBrushParameters):
        self.parameters = parameters

    def initial_state(
        self,
        *,
        x_m: float,
        y_m: float,
        press_depth_mm: float,
        alpha_rad: float,
        beta_rad: float,
        heading_rad: float,
    ) -> DynamicFootprintState:
        # Resetting w/d to zero at every DOWN event produces the tapered start
        # of a real stroke; the first dynamic step then approaches the target.
        empty = BrushDimensions(0.0, 0.0, 0.0)
        h, alpha, beta = self.parameters.inverse_virtual_pose(empty)
        return DynamicFootprintState(
            x_m=x_m,
            y_m=y_m,
            root_x_m=x_m,
            root_y_m=y_m,
            press_depth_mm=press_depth_mm,
            half_width_m=0.0,
            drag_length_m=0.0,
            offset_m=0.0,
            heading_rad=heading_rad,
            virtual_h_mm=h,
            virtual_alpha_rad=alpha,
            virtual_beta_rad=beta,
            dimensions=BrushDimensions(0.0, 0.0, 0.0),
        )

    def step(
        self,
        previous: DynamicFootprintState,
        *,
        x_m: float,
        y_m: float,
        press_depth_mm: float,
        alpha_rad: float,
        beta_rad: float,
        heading_rad: float,
    ) -> DynamicFootprintState:
        target = self.parameters.dimensions(press_depth_mm, alpha_rad, beta_rad)
        target_drag = target.tip_length_m + target.heel_length_m
        width = (
            previous.half_width_m * self.parameters.width_inertia
            + target.half_width_m * (1.0 - self.parameters.width_inertia)
        )
        drag = (
            previous.drag_length_m * self.parameters.drag_inertia
            + target_drag * (1.0 - self.parameters.drag_inertia)
        )

        # Wang et al. define x/y as the handle (end-effector) projection and
        # X_root as a separate state derived from offset and orientation.  If
        # the new handle position remains within the available deformation,
        # paper friction holds the old root fixed.  Otherwise the root snaps
        # to the maximum-deformation location behind the handle.
        delta_x = x_m - previous.root_x_m
        delta_y = y_m - previous.root_y_m
        required_hold = math.hypot(delta_x, delta_y)
        free_offset = min(drag, 0.5 * self.parameters.bundle_length_m)
        offset = min(free_offset, required_hold)
        if required_hold > 1.0e-12:
            path_heading = math.atan2(delta_y, delta_x)
        else:
            path_heading = heading_rad
        if required_hold <= free_offset:
            root_x = previous.root_x_m
            root_y = previous.root_y_m
        else:
            root_x = x_m - offset * math.cos(path_heading)
            root_y = y_m - offset * math.sin(path_heading)

        target_total = max(target_drag, 1.0e-12)
        dynamic_dimensions = BrushDimensions(
            tip_length_m=drag * target.tip_length_m / target_total,
            heel_length_m=drag * target.heel_length_m / target_total,
            half_width_m=width,
        )
        virtual_h, virtual_alpha, virtual_beta = self.parameters.inverse_virtual_pose(
            dynamic_dimensions
        )
        return DynamicFootprintState(
            x_m=x_m,
            y_m=y_m,
            root_x_m=root_x,
            root_y_m=root_y,
            press_depth_mm=press_depth_mm,
            half_width_m=width,
            drag_length_m=drag,
            offset_m=offset,
            # ``heading_rad`` is the absolute page orientation of the brush
            # footprint.  The root offset above follows the path tangent,
            # while the V17 gamma-conditioned brush may be twisted relative
            # to that tangent.
            heading_rad=heading_rad,
            virtual_h_mm=virtual_h,
            virtual_alpha_rad=virtual_alpha,
            virtual_beta_rad=virtual_beta,
            dimensions=dynamic_dimensions,
        )

    def polygon(
        self, state: DynamicFootprintState, *, bezier_samples: int = 14
    ) -> np.ndarray:
        """Create a footprint from the virtual pose obtained by the bridge."""

        return footprint_polygon_from_dimensions(
            state.root_x_m,
            state.root_y_m,
            state.heading_rad,
            state.dimensions,
            bezier_samples=bezier_samples,
        )


def _cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    samples: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(4, int(samples)), dtype=np.float64)[:, None]
    one_minus_t = 1.0 - t
    return (
        p0 * one_minus_t**3
        + 3.0 * p1 * t * one_minus_t**2
        + 3.0 * p2 * t**2 * one_minus_t
        + p3 * t**3
    )


def footprint_polygon(
    center_x_m: float,
    center_y_m: float,
    heading_rad: float,
    press_depth_mm: float,
    alpha_rad: float,
    beta_rad: float,
    parameters: PaperBrushParameters,
    *,
    bezier_samples: int = 14,
) -> np.ndarray:
    """Return the closed Eq. (8)--(9) footprint as an ``N x 2`` array."""

    dimensions = parameters.dimensions(press_depth_mm, alpha_rad, beta_rad)
    return footprint_polygon_from_dimensions(
        center_x_m,
        center_y_m,
        heading_rad,
        dimensions,
        bezier_samples=bezier_samples,
    )


def footprint_polygon_from_dimensions(
    center_x_m: float,
    center_y_m: float,
    heading_rad: float,
    dimensions: BrushDimensions,
    *,
    bezier_samples: int = 14,
) -> np.ndarray:
    """Build the symmetric cubic-Bezier footprint for known dimensions."""

    lt = dimensions.tip_length_m
    lh = dimensions.heel_length_m
    lr = dimensions.half_width_m

    p0 = np.array([-lt, 0.0], dtype=np.float64)
    p3 = np.array([lh, 0.0], dtype=np.float64)
    p1x = (lt - 4.0 * lh) / 3.0
    p2x = lh
    upper = _cubic_bezier(
        p0,
        np.array([p1x, 4.0 * lr / 3.0]),
        np.array([p2x, 4.0 * lr / 3.0]),
        p3,
        bezier_samples,
    )
    lower = _cubic_bezier(
        p0,
        np.array([p1x, -4.0 * lr / 3.0]),
        np.array([p2x, -4.0 * lr / 3.0]),
        p3,
        bezier_samples,
    )
    local = np.vstack((upper, lower[-2:0:-1]))

    cosine, sine = math.cos(heading_rad), math.sin(heading_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.array([center_x_m, center_y_m])


def triangle_fan(polygon: np.ndarray) -> np.ndarray:
    """Triangulate the convex B-BSM footprint for an RViz TRIANGLE_LIST."""

    if polygon.shape[0] < 3:
        return np.empty((0, 3, 2), dtype=np.float64)
    center = np.mean(polygon, axis=0)
    triangles = []
    for index in range(polygon.shape[0]):
        triangles.append((center, polygon[index], polygon[(index + 1) % len(polygon)]))
    return np.asarray(triangles, dtype=np.float64)


def _fill_polygon(mask: np.ndarray, polygon: np.ndarray) -> None:
    """Fill a polygon into a float mask using a vectorised even-odd rule."""

    height, width = mask.shape
    x_min = max(0, int(math.floor(float(np.min(polygon[:, 0])))))
    x_max = min(width - 1, int(math.ceil(float(np.max(polygon[:, 0])))))
    y_min = max(0, int(math.floor(float(np.min(polygon[:, 1])))))
    y_max = min(height - 1, int(math.ceil(float(np.max(polygon[:, 1])))))
    if x_min > x_max or y_min > y_max:
        return

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5,
        np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5,
    )
    inside = np.zeros(grid_x.shape, dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        crosses = (y0 > grid_y) != (y1 > grid_y)
        denominator = y1 - y0
        if abs(float(denominator)) < 1.0e-12:
            previous = current
            continue
        x_at_y = (x1 - x0) * (grid_y - y0) / denominator + x0
        inside ^= crosses & (grid_x < x_at_y)
        previous = current
    region = mask[y_min : y_max + 1, x_min : x_max + 1]
    np.maximum(region, inside.astype(np.float32), out=region)


def render_world_footprints(
    footprints: Iterable[np.ndarray],
    *,
    paper_width_m: float,
    paper_height_m: float,
    paper_offset_x_m: float,
    paper_offset_y_m: float,
    image_size: int = 128,
    supersampling: int = 2,
) -> np.ndarray:
    """Rasterise world-coordinate footprints into an antialiased ink mask."""

    scale = max(1, int(supersampling))
    raster_size = image_size * scale
    canvas = np.zeros((raster_size, raster_size), dtype=np.float32)
    left = paper_offset_x_m - 0.5 * paper_width_m
    top = paper_offset_y_m + 0.5 * paper_height_m
    for footprint in footprints:
        pixels = np.empty_like(footprint, dtype=np.float64)
        pixels[:, 0] = (footprint[:, 0] - left) / paper_width_m * raster_size
        # ROS/world y increases upward, while an image row increases downward.
        # Convert the paper's upper edge to row zero so the simulated glyph has
        # the same vertical orientation as the target PNG.
        pixels[:, 1] = (top - footprint[:, 1]) / paper_height_m * raster_size
        _fill_polygon(canvas, pixels)

    if scale == 1:
        return canvas
    image = Image.fromarray(np.uint8(np.clip(canvas, 0.0, 1.0) * 255), mode="L")
    image = image.resize((image_size, image_size), _LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_normalized_target(
    path: Path, *, image_size: int, margin_ratio: float
) -> np.ndarray:
    """Crop a target's ink bounds and place it in the ROS paper margin."""

    gray = Image.open(path).convert("L")
    ink = 1.0 - np.asarray(gray, dtype=np.float32) / 255.0
    return normalize_ink_mask(
        ink, image_size=image_size, margin_ratio=margin_ratio
    )


def normalize_ink_mask(
    mask: np.ndarray, *, image_size: int, margin_ratio: float
) -> np.ndarray:
    """Crop and align an ink mask using the paper's comparison protocol."""

    ink = np.asarray(mask, dtype=np.float32)
    foreground = ink > (8.0 / 255.0)
    if not bool(np.any(foreground)):
        raise ValueError("ink mask contains no foreground")
    ys, xs = np.nonzero(foreground)
    cropped = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    margin = max(0, min(image_size // 3, int(round(image_size * margin_ratio))))
    inner_size = max(1, image_size - 2 * margin)
    resized = Image.fromarray(np.uint8(np.clip(cropped, 0.0, 1.0) * 255), mode="L")
    resized = resized.resize((inner_size, inner_size), _LANCZOS)
    target = np.zeros((image_size, image_size), dtype=np.float32)
    target[margin : margin + inner_size, margin : margin + inner_size] = (
        np.asarray(resized, dtype=np.float32) / 255.0
    )
    return target


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine image similarity (CSIM)."""

    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-15:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def _gaussian_filter(image: np.ndarray, *, size: int = 11, sigma: float = 1.5) -> np.ndarray:
    radius = size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(image, dtype=np.float64), radius, mode="reflect")
    horizontal = np.zeros((image.shape[0] + 2 * radius, image.shape[1]), dtype=np.float64)
    for index, weight in enumerate(kernel):
        horizontal += weight * padded[:, index : index + image.shape[1]]
    output = np.zeros(image.shape, dtype=np.float64)
    for index, weight in enumerate(kernel):
        output += weight * horizontal[index : index + image.shape[0], :]
    return output


def structural_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Standard local-window SSIM for images with data range 1."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("SSIM expects two equally sized 2-D images")
    mu_left = _gaussian_filter(left)
    mu_right = _gaussian_filter(right)
    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_cross = mu_left * mu_right
    sigma_left = _gaussian_filter(left * left) - mu_left_sq
    sigma_right = _gaussian_filter(right * right) - mu_right_sq
    sigma_cross = _gaussian_filter(left * right) - mu_cross
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (
        sigma_left + sigma_right + c2
    )
    score = np.mean(numerator / np.maximum(denominator, 1.0e-15))
    return float(np.clip(score, -1.0, 1.0))


def evaluate_masks(
    simulated: np.ndarray,
    target: np.ndarray,
    *,
    minimum_csim: float = 0.9136,
    minimum_ssim: float = 0.7042,
    calibrated_domain_valid: bool = True,
) -> dict:
    """Evaluate trajectory feasibility with the paper's two image metrics."""

    csim = cosine_similarity(simulated, target)
    ssim = structural_similarity(simulated, target)
    return {
        "csim": csim,
        "ssim": ssim,
        "minimum_csim": float(minimum_csim),
        "minimum_ssim": float(minimum_ssim),
        "calibrated_domain_valid": bool(calibrated_domain_valid),
        "feasible": bool(
            calibrated_domain_valid
            and csim >= minimum_csim
            and ssim >= minimum_ssim
        ),
        "threshold_basis": "paper Table 4 minima for robot writing",
    }


def save_evaluation_artifacts(
    output_directory: Path,
    simulated: np.ndarray,
    target: np.ndarray,
    report: dict,
) -> Tuple[Path, Path]:
    """Save a machine-readable report and a target/simulation/difference panel."""

    output_directory.mkdir(parents=True, exist_ok=True)
    target_display = np.uint8((1.0 - np.clip(target, 0.0, 1.0)) * 255)
    simulated_display = np.uint8((1.0 - np.clip(simulated, 0.0, 1.0)) * 255)
    difference = np.uint8(np.abs(simulated - target) * 255)
    panel = np.concatenate((target_display, simulated_display, difference), axis=1)
    panel_path = output_directory / "brush_evaluation.png"
    Image.fromarray(panel, mode="L").save(panel_path)

    report_path = output_directory / "brush_evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path, panel_path


def parameters_as_dict(parameters: PaperBrushParameters) -> dict:
    """Return JSON-safe model provenance."""

    result = asdict(parameters)
    for key in ("lt_coefficients", "lh_coefficients", "lr_coefficients"):
        result[key] = list(result[key])
    return result
