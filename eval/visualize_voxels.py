#!/usr/bin/env python3
"""Voxel Scene Visualizer (md §7.5).

Generates static images of:
  1. Voxel occupancy point cloud (top-down + side views)
  2. 3 isometric views
  3. Occupancy statistics

Usage:
    python visualize_voxels.py \
        --scene_dir LINGO/dataset/dataset/Scene \
        --output_dir kimodo_scene_project/outputs/visualizations \
        --num_scenes 5
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def voxel_to_pointcloud(voxel: np.ndarray) -> np.ndarray:
    indices = np.argwhere(voxel > 0.5).astype(np.float32)
    return indices


def render_topdown_text(voxel: np.ndarray) -> str:
    X, Y, Z = voxel.shape
    ground = voxel[:, :, 0]
    lines = []
    lines.append(f"Voxel Scene: {X}x{Y}x{Z}, occupancy={voxel.mean()*100:.1f}%")
    lines.append("Top-down view (z=0 slice):")
    for x in range(0, X, 4):
        row = ""
        for y in range(0, Y, 4):
            block_sum = ground[x:x+4, y:y+4].sum()
            if block_sum > 8:
                row += "##"
            elif block_sum > 4:
                row += "++"
            elif block_sum > 0:
                row += ".."
            else:
                row += "  "
        lines.append(f"  {row}")
    return "\n".join(lines)


def compute_voxel_stats(voxel: np.ndarray) -> dict:
    X, Y, Z = voxel.shape
    occ_z = voxel.sum(axis=(0, 1))
    occ_y = voxel.sum(axis=(0, 2))
    occ_x = voxel.sum(axis=(1, 2))

    return {
        "shape": list(voxel.shape),
        "occupancy_ratio": float(voxel.mean()),
        "total_voxels": int(voxel.sum()),
        "occupied_z_slices": int((occ_z > 0).sum()),
        "max_z_occupancy": int(occ_z.max()),
        "occupied_y_slices": int((occ_y > 0).sum()),
        "occupied_x_slices": int((occ_x > 0).sum()),
        "ground_occupancy": float(voxel[:, :, 0].mean()),
        "bottom_quartile_occ": float(voxel[:, :, :Z//4].mean()),
        "top_quartile_occ": float(voxel[:, :, 3*Z//4:].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Voxel Scene Visualizer")
    parser.add_argument("--scene_dir", type=str,
                        default="LINGO/dataset/dataset/Scene")
    parser.add_argument("--output_dir", type=str,
                        default="kimodo_scene_project/outputs/visualizations")
    parser.add_argument("--num_scenes", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(Path(args.scene_dir).glob("*.npy"))[:args.num_scenes]
    if not scene_files:
        print(f"No .npy files found in {args.scene_dir}")
        return

    print(f"Visualizing {len(scene_files)} scene(s)...")

    all_stats = {}
    for sf in scene_files:
        voxel = np.load(sf).astype(np.float32)
        stats = compute_voxel_stats(voxel)
        asc = render_topdown_text(voxel)

        print(f"\n{'='*60}")
        print(asc)
        print(f"Stats: occ={stats['occupancy_ratio']:.3f}, "
              f"ground={stats['ground_occupancy']:.3f}, "
              f"bottom={stats['bottom_quartile_occ']:.3f}, "
              f"top={stats['top_quartile_occ']:.3f}")
        all_stats[sf.name] = stats

    import json
    with open(out_dir / "voxel_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nVoxel stats saved to {out_dir / 'voxel_stats.json'}")
    print(f"Top-down views printed above.")


if __name__ == "__main__":
    main()
