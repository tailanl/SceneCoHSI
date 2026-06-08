"""Scene coordinate normalization: align scene + motion coordinate systems.

Ensures consistent units, up-axis, ground plane, forward direction,
and origin alignment between 3D scenes and human motion data.
"""

import numpy as np
from typing import Optional, Tuple


def center_scene(
    scene_points: np.ndarray,
    target_up: str = "Z",
    source_up: str = "Y",
) -> Tuple[np.ndarray, np.ndarray]:
    """Center scene points and optionally swap axes to match target coordinate system.

    Args:
        scene_points: [N, 3] or [..., 3] array of scene points.
        target_up: Desired up-axis: 'Y' or 'Z'.
        source_up: Source up-axis: 'Y' or 'Z'.

    Returns:
        centered_points: Centered (mean subtracted) scene points with correct up-axis.
        center: [3] center point removed from the scene.
    """
    points = scene_points.copy()
    center = points.reshape(-1, 3).mean(axis=0)
    points = points - center

    if target_up.upper() != source_up.upper():
        if target_up.upper() == "Z" and source_up.upper() == "Y":
            points = points[..., [0, 2, 1]]
        elif target_up.upper() == "Y" and source_up.upper() == "Z":
            points = points[..., [0, 2, 1]]

    return points, center


def align_ground(
    scene_points: np.ndarray,
    ground_height: float = 0.0,
) -> np.ndarray:
    """Shift scene so that the lowest point rests at ground_height.

    Args:
        scene_points: [N, 3] array (Z-up convention).
        ground_height: Desired ground Z coordinate.

    Returns:
        Shifted scene points.
    """
    points = scene_points.copy().astype(np.float64)
    min_z = float(points[..., 2].min())
    points[..., 2] += (ground_height - min_z)
    return points


def normalize_motion_to_scene(
    motion_root: np.ndarray,
    scene_center: np.ndarray,
) -> np.ndarray:
    """Align motion root trajectory to scene center.

    Args:
        motion_root: [T, 3] root positions (Z-up).
        scene_center: [3] scene center point.

    Returns:
        Aligned root positions.
    """
    root = motion_root.copy()
    root[:, :2] -= scene_center[:2]
    return root


def check_scene_validity(
    voxel_grid: np.ndarray,
    motion_root_xy: np.ndarray,
    scene_bounds: Optional[np.ndarray] = None,
) -> dict:
    """Run basic sanity checks on scene and motion alignment.

    Args:
        voxel_grid: [X, Y, Z] binary occupancy grid.
        motion_root_xy: [T, 2] root trajectory on ground plane.
        scene_bounds: [2, 3] optional scene bounding box (min, max).

    Returns:
        Dict with check results: passed (bool) and warnings (list).
    """
    results = {"passed": True, "warnings": []}

    occupancy_ratio = voxel_grid.mean()
    if occupancy_ratio == 0:
        results["passed"] = False
        results["warnings"].append("voxel_grid is all zeros (empty scene)")
    elif occupancy_ratio == 1.0:
        results["passed"] = False
        results["warnings"].append("voxel_grid is all ones (fully occupied)")

    if np.isnan(voxel_grid).any():
        results["passed"] = False
        results["warnings"].append("voxel_grid contains NaN")

    if np.isnan(motion_root_xy).any():
        results["passed"] = False
        results["warnings"].append("motion_root_xy contains NaN")

    if scene_bounds is not None:
        min_b, max_b = np.asarray(scene_bounds[0]), np.asarray(scene_bounds[1])
        out_of_bounds = (
            (motion_root_xy[:, 0] < min_b[0]).sum()
            + (motion_root_xy[:, 0] > max_b[0]).sum()
            + (motion_root_xy[:, 1] < min_b[1]).sum()
            + (motion_root_xy[:, 1] > max_b[1]).sum()
        )
        if out_of_bounds > 0:
            results["warnings"].append(
                f"Root trajectory has {out_of_bounds} points outside scene bounds"
            )

    return results


def check_units(scene_bounds: np.ndarray, max_expected_extent: float = 50.0) -> dict:
    """Check that scene units are in meters (not mm or cm).

    Strategy: if the scene extent is > max_expected_extent meters,
    it's likely in mm and needs conversion.

    Args:
        scene_bounds: [2, 3] (min, max) bounding box.
        max_expected_extent: Maximum reasonable scene size in meters.

    Returns:
        dict with 'is_meters': bool, 'extent': float, 'suggested_scale': float.
    """
    extent = np.linalg.norm(scene_bounds[1] - scene_bounds[0])
    is_meters = extent <= max_expected_extent

    suggested_scale = 1.0
    if not is_meters:
        if extent > 1000:
            suggested_scale = 0.001
        elif extent > 100:
            suggested_scale = 0.01
        else:
            suggested_scale = 0.1

    return {
        "is_meters": is_meters,
        "extent": float(extent),
        "suggested_scale": suggested_scale,
        "warning": ("" if is_meters else
                     f"Scene extent {extent:.1f} > {max_expected_extent:.0f}, "
                     f"suggest scale={suggested_scale}"),
    }


def check_forward_direction(
    root_positions: np.ndarray,
    expected_forward_axis: str = "-Y",
) -> dict:
    """Check that forward direction matches expected convention.

    Kimodo uses -Y as the default forward direction.

    Args:
        root_positions: [T, 3] root positions.
        expected_forward_axis: e.g. "+X", "-Y", "+Z".

    Returns:
        dict with primary direction info.
    """
    if len(root_positions) < 2:
        return {"checked": False, "reason": "too few frames"}

    displacements = root_positions[1:] - root_positions[:-1]
    mean_disp = displacements.mean(axis=0)

    max_idx = np.argmax(np.abs(mean_disp))
    max_val = mean_disp[max_idx]
    axes = ["X", "Y", "Z"]
    sign = "+" if max_val >= 0 else "-"
    primary_dir = f"{sign}{axes[max_idx]}"

    matches = (primary_dir.upper() == expected_forward_axis.upper())

    return {
        "checked": True,
        "primary_direction": primary_dir,
        "expected_forward": expected_forward_axis,
        "matches": matches,
        "mean_displacement": mean_disp.tolist(),
    }


__all__ = [
    "center_scene",
    "align_ground",
    "normalize_motion_to_scene",
    "check_scene_validity",
    "check_units",
    "check_forward_direction",
]
