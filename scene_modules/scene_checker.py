"""Scene-motion collision, contact, and sensitivity checking utilities.

Provides functions to compute all C-class (scene adaptation) and E-class (ablation)
evaluation metrics from Kimodo_SceneCo_gpt.md Chapters 15-17.
"""

import numpy as np
from typing import List, Optional, Tuple


def check_collision(
    joint_positions: np.ndarray,
    voxel_grid: np.ndarray,
    voxel_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.1,
    joint_radius: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check which joints collide with occupied voxels.

    Args:
        joint_positions: [T, J, 3] joint positions in world coordinates.
        voxel_grid: [X, Y, Z] binary occupancy grid.
        voxel_origin: [3] world-coordinate origin of voxel grid (min corner).
        voxel_size: Size of each voxel in world units.
        joint_radius: Radius around each joint for collision checking.

    Returns:
        collided_mask: [T, J] boolean array.
        penetration_depths: [T, J] approximate penetration distances.
    """
    T, J, _ = joint_positions.shape
    X, Y, Z = voxel_grid.shape

    if voxel_origin is None:
        voxel_origin = np.zeros(3)

    voxel_coords = ((joint_positions - voxel_origin) / voxel_size).astype(np.int32)

    valid = (
        (voxel_coords[..., 0] >= 0) & (voxel_coords[..., 0] < X)
        & (voxel_coords[..., 1] >= 0) & (voxel_coords[..., 1] < Y)
        & (voxel_coords[..., 2] >= 0) & (voxel_coords[..., 2] < Z)
    )

    collided_mask = np.zeros((T, J), dtype=bool)
    penetration_depths = np.zeros((T, J), dtype=np.float32)

    for t in range(T):
        for j in range(J):
            if not valid[t, j]:
                continue
            cx, cy, cz = voxel_coords[t, j]
            if voxel_grid[cx, cy, cz] > 0.5:
                collided_mask[t, j] = True
                penetration_depths[t, j] = joint_radius

    return collided_mask, penetration_depths


def collision_frame_rate(collided_mask: np.ndarray) -> float:
    """CFR: fraction of frames with at least one collision."""
    if collided_mask.size == 0:
        return 0.0
    return float(collided_mask.any(axis=1).mean())


def joint_collision_ratio(collided_mask: np.ndarray) -> float:
    """JCR: fraction of all joint-frame pairs that collide."""
    if collided_mask.size == 0:
        return 0.0
    return float(collided_mask.mean())


def penetration_free_frame_ratio(collided_mask: np.ndarray) -> float:
    """PFFR: fraction of frames with zero collisions."""
    return 1.0 - collision_frame_rate(collided_mask)


def mean_penetration_depth(penetration_depths: np.ndarray, collided_mask: np.ndarray) -> float:
    """MeanPen: mean penetration over collided joints."""
    if not collided_mask.any():
        return 0.0
    return float(penetration_depths[collided_mask].mean())


def max_penetration_depth(penetration_depths: np.ndarray, collided_mask: np.ndarray) -> float:
    """MaxPen: worst single-joint penetration."""
    if not collided_mask.any():
        return 0.0
    return float(penetration_depths[collided_mask].max())


def p95_penetration_depth(penetration_depths: np.ndarray, collided_mask: np.ndarray) -> float:
    """P95Pen: 95th percentile penetration depth."""
    if not collided_mask.any():
        return 0.0
    return float(np.percentile(penetration_depths[collided_mask], 95))


def std_penetration_depth(penetration_depths: np.ndarray, collided_mask: np.ndarray) -> float:
    """StdPen: standard deviation of penetration depth."""
    if not collided_mask.any():
        return 0.0
    return float(penetration_depths[collided_mask].std())


def root_path_intersection(
    root_xy: np.ndarray,
    voxel_grid: np.ndarray,
    voxel_origin: Optional[np.ndarray] = None,
    voxel_size: float = 0.1,
    ground_z_idx: int = 0,
) -> float:
    """OPIR: obstacle path intersection rate.

    Projects occupied voxels onto the ground plane and checks how many
    root positions fall inside occupied areas.

    Args:
        root_xy: [T, 2] root positions on ground (X, Y).
        voxel_grid: [X, Y, Z] binary occupancy.
        voxel_origin: [3] voxel grid origin.
        voxel_size: Voxel size.
        ground_z_idx: Which Z slice to use for ground projection.

    Returns:
        OPIR: fraction of root steps intersecting obstacles.
    """
    if voxel_origin is None:
        voxel_origin = np.zeros(3)

    gz = min(ground_z_idx, voxel_grid.shape[2] - 1)
    ground_occ = voxel_grid[:, :, gz]
    X, Y = ground_occ.shape

    voxel_coords = ((root_xy - voxel_origin[:2]) / voxel_size).astype(np.int32)

    valid = (
        (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < X)
        & (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < Y)
    )

    intersections = 0
    for t in range(len(root_xy)):
        if valid[t]:
            cx, cy = voxel_coords[t]
            if ground_occ[cx, cy] > 0.5:
                intersections += 1

    if len(root_xy) == 0:
        return 0.0
    return float(intersections) / len(root_xy)


def target_object_distance(
    root_xy: np.ndarray,
    object_center_xy: np.ndarray,
    mode: str = "min",
) -> float:
    """TargetDist: distance from root trajectory to target object.

    Args:
        root_xy: [T, 2] root positions on ground.
        object_center_xy: [2] target object center position.
        mode: 'min' for closest approach, 'final' for last-frame distance.

    Returns:
        Distance in world units.
    """
    if len(root_xy) == 0:
        return float("inf")

    if mode == "min":
        return float(np.min(np.linalg.norm(root_xy - object_center_xy, axis=1)))
    elif mode == "final":
        return float(np.linalg.norm(root_xy[-1] - object_center_xy))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def check_scene_sensitivity(
    joint_positions_list: List[np.ndarray],
    voxel_grids: List[np.ndarray],
    voxel_size: float = 0.1,
) -> dict:
    """Check if scene has effect on generated motions (Scene Sensitivity).

    If CFR is the same across different scenes, SceneCo has no effect.

    Args:
        joint_positions_list: list of [T, J, 3] for each scene.
        voxel_grids: list of [X, Y, Z] for each scene.
        voxel_size: Voxel size.

    Returns:
        dict with per-scene CFR and variance across scenes.
    """
    cfrs = []
    for joints, voxel in zip(joint_positions_list, voxel_grids):
        collided, _ = check_collision(joints, voxel, voxel_size=voxel_size)
        cfrs.append(collision_frame_rate(collided))

    mean_cfr = float(np.mean(cfrs)) if cfrs else 0.0
    std_cfr = float(np.std(cfrs)) if cfrs else 0.0

    return {
        "per_scene_cfr": cfrs,
        "mean_cfr": mean_cfr,
        "std_cfr": std_cfr,
        "has_effect": std_cfr > 1e-6,
        "cv": (std_cfr / max(mean_cfr, 1e-8)),
    }


def test_random_scene(
    true_joint_positions: np.ndarray,
    true_voxel: np.ndarray,
    random_voxel: np.ndarray,
    voxel_size: float = 0.1,
) -> dict:
    """Test E-class: Random Scene test.

    Motions generated with a random scene should have worse CFR/JCR.
    """
    true_collided, true_depth = check_collision(true_joint_positions, true_voxel, voxel_size=voxel_size)
    rand_collided, rand_depth = check_collision(true_joint_positions, random_voxel, voxel_size=voxel_size)

    return {
        "true_cfr": collision_frame_rate(true_collided),
        "random_cfr": collision_frame_rate(rand_collided),
        "true_jcr": joint_collision_ratio(true_collided),
        "random_jcr": joint_collision_ratio(rand_collided),
        "delta_cfr": collision_frame_rate(true_collided) - collision_frame_rate(rand_collided),
        "delta_jcr": joint_collision_ratio(true_collided) - joint_collision_ratio(rand_collided),
    }


def test_empty_scene(
    joint_positions: np.ndarray,
    voxel_size: float = 0.1,
) -> dict:
    """Test E-class: Empty Scene test.

    Motions with empty scene should have CFR=0 and JCR=0.
    """
    empty_grid = np.zeros((8, 8, 8), dtype=np.float32)
    collided, depth = check_collision(joint_positions, empty_grid, voxel_size=voxel_size)

    return {
        "empty_cfr": collision_frame_rate(collided),
        "empty_jcr": joint_collision_ratio(collided),
        "cfr_zero": (collision_frame_rate(collided) == 0.0),
        "jcr_zero": (joint_collision_ratio(collided) == 0.0),
    }


def test_shuffled_scene(
    joint_positions: np.ndarray,
    true_voxel: np.ndarray,
    voxel_size: float = 0.1,
) -> dict:
    """Test E-class: Shuffled Scene test.

    Shuffle voxel positions; result should differ from true scene.
    """
    shuffled = true_voxel.flatten().copy()
    np.random.shuffle(shuffled)
    shuffled = shuffled.reshape(true_voxel.shape)

    true_collided, _ = check_collision(joint_positions, true_voxel, voxel_size=voxel_size)
    shuf_collided, _ = check_collision(joint_positions, shuffled, voxel_size=voxel_size)

    return {
        "true_cfr": collision_frame_rate(true_collided),
        "shuffled_cfr": collision_frame_rate(shuf_collided),
        "delta_cfr": abs(collision_frame_rate(true_collided) - collision_frame_rate(shuf_collided)),
        "has_effect": abs(collision_frame_rate(true_collided) - collision_frame_rate(shuf_collided)) > 1e-6,
    }


def compute_contact_success(
    posed_joints: np.ndarray,
    object_center_xyz: np.ndarray,
    object_bounds: np.ndarray,
    contact_type: str = "sit",
    pelvis_idx: int = 0,
    hand_indices: Optional[List[int]] = None,
    foot_indices: Optional[List[int]] = None,
) -> dict:
    """15.8: Contact Success Rate — check if contact with object succeeded.

    Determines success differently per contact_type:
      - 'sit': pelvis near seat surface, low velocity at end
      - 'touch': hands near object surface
      - 'stand': feet near ground

    Args:
        posed_joints: [T, J, 3] joint positions.
        object_center_xyz: [3] object center in world.
        object_bounds: [2, 3] (min, max) object bounds.
        contact_type: 'sit', 'touch', or 'stand'.
        pelvis_idx: joint index for pelvis.
        hand_indices: [left_hand, right_hand] indices.
        foot_indices: [left_foot, right_foot] indices.

    Returns:
        dict with success (bool) and contact_quality (float 0-1).
    """
    if foot_indices is None:
        foot_indices = [7, 10]
    if hand_indices is None:
        hand_indices = [20, 21]

    T = posed_joints.shape[0]
    if T < 2:
        return {"success": False, "contact_quality": 0.0, "reason": "too few frames"}

    obj_min, obj_max = object_bounds[0].copy(), object_bounds[1].copy()

    if contact_type == "sit":
        seat_center = object_center_xyz.copy()
        seat_radius = float(np.linalg.norm(obj_max[:2] - obj_min[:2]) * 0.4)

        last_third = posed_joints[T * 2 // 3:]
        pelvis_last = last_third[:, pelvis_idx, :]
        pelvis_xy = pelvis_last[:, :2]
        pelvis_z = pelvis_last[:, 2]

        dist_to_seat = np.linalg.norm(pelvis_xy - seat_center[:2], axis=1)
        near_seat = dist_to_seat < seat_radius

        target_z = (obj_min[2] + obj_max[2]) * 0.55
        z_ok = np.abs(pelvis_z - target_z) < 0.3

        last_vel = np.linalg.norm(pelvis_last[1:] - pelvis_last[:-1], axis=1)
        settled = last_vel.mean() < 0.1 if len(last_vel) > 0 else True

        success = bool(np.any(near_seat & z_ok)) and settled
        quality = float(np.mean(near_seat * z_ok)) if np.any(near_seat) else 0.0

    elif contact_type == "touch":
        obj_surface_z = obj_max[2]
        contact_frames = 0
        for hi in hand_indices:
            hand_z = posed_joints[-10:, hi, 2] if T >= 10 else posed_joints[:, hi, 2]
            contact_frames += int(np.sum(np.abs(hand_z - obj_surface_z) < 0.2))
        max_contact = max(1, (min(T, 10)) * len(hand_indices))
        quality = min(1.0, contact_frames / max_contact)
        success = quality > 0.3
        seat_radius = 0.0

    elif contact_type == "stand":
        foot_zs = np.stack([posed_joints[:, fi, 2] for fi in foot_indices], axis=1)
        on_ground = np.abs(foot_zs) < 0.1
        quality = float(np.mean(on_ground))
        success = quality > 0.5
        seat_radius = 0.0

    else:
        return {"success": False, "contact_quality": 0.0, "reason": f"unknown contact_type: {contact_type}"}

    return {
        "success": success,
        "contact_quality": float(quality),
        "contact_type": contact_type,
    }


def compute_empty_scene_path_ratio(
    root_xy_scene: np.ndarray,
    root_xy_empty: np.ndarray,
) -> dict:
    """15.12: Empty Scene Path Length Ratio.

    Path_Length_Ratio = length(generated with empty scene) / length(generated with real scene).

    Should be in [0.9, 1.1] if model doesn't over-avoid.

    Args:
        root_xy_scene: [T, 2] root trajectory with real scene.
        root_xy_empty: [T, 2] root trajectory with empty scene.

    Returns:
        dict with ratio, passed, and lengths.
    """
    if len(root_xy_scene) < 2 or len(root_xy_empty) < 2:
        return {"ratio": None, "passed": False, "reason": "too few frames"}

    len_scene = float(np.sum(np.linalg.norm(root_xy_scene[1:] - root_xy_scene[:-1], axis=1)))
    len_empty = float(np.sum(np.linalg.norm(root_xy_empty[1:] - root_xy_empty[:-1], axis=1)))

    if len_scene < 1e-8:
        return {"ratio": None, "passed": False, "reason": "zero scene path length"}

    ratio = len_empty / len_scene
    passed = 0.9 <= ratio <= 1.1

    return {
        "ratio": ratio,
        "passed": passed,
        "scene_path_length": len_scene,
        "empty_path_length": len_empty,
    }


__all__ = [
    "check_collision",
    "collision_frame_rate",
    "joint_collision_ratio",
    "penetration_free_frame_ratio",
    "mean_penetration_depth",
    "max_penetration_depth",
    "p95_penetration_depth",
    "std_penetration_depth",
    "root_path_intersection",
    "target_object_distance",
    "check_scene_sensitivity",
    "test_random_scene",
    "test_empty_scene",
    "test_shuffled_scene",
    "compute_contact_success",
    "compute_empty_scene_path_ratio",
]
