#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm


def collect_xyz_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.xyz"))
    if not files:
        raise FileNotFoundError(f"No .xyz files found in: {folder}")
    return files


def load_points_from_xyz(file_path: Path) -> np.ndarray:
    points = np.loadtxt(file_path)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"File has invalid xyz format: {file_path}")
    return points[:, :3]


def build_global_map(
    xyz_files: list[Path],
    poses: np.ndarray,
    end: int | None,
) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses shape (N, 4, 4), got: {poses.shape}")

    total_frames = min(len(xyz_files), len(poses))
    if total_frames == 0:
        raise ValueError("No frames available for map building.")

    if end is None:
        end = total_frames
    end = min(end, total_frames)
    selected_indices = list(range(0, end))
    if not selected_indices:
        raise ValueError("Selected frame range is empty. Check end_episode.")

    transformed_chunks: list[np.ndarray] = []
    for idx in tqdm(selected_indices, desc="Building global map"):
        points = load_points_from_xyz(xyz_files[idx])
        ones = np.ones((points.shape[0], 1), dtype=points.dtype)
        hom_points = np.hstack((points, ones))

        transformed_hom = (poses[idx] @ hom_points.T).T
        w = transformed_hom[:, 3:4]
        w = np.where(np.abs(w) < 1e-12, 1.0, w)
        transformed_points = transformed_hom[:, :3] / w
        transformed_chunks.append(transformed_points)

    return np.vstack(transformed_chunks)


def visualize_points(points: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Global map points: {points.shape[0]}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Global map", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)

    state = {"running": True, "zoom": 0.7}
    vis.get_view_control().set_zoom(state["zoom"])

    def zoom_in(_: o3d.visualization.Visualizer) -> bool:
        state["zoom"] = max(0.02, state["zoom"] - 0.05)
        vis.get_view_control().set_zoom(state["zoom"])
        print(f"zoom: {state['zoom']:.2f}")
        return False

    def zoom_out(_: o3d.visualization.Visualizer) -> bool:
        state["zoom"] = min(2.0, state["zoom"] + 0.05)
        vis.get_view_control().set_zoom(state["zoom"])
        print(f"zoom: {state['zoom']:.2f}")
        return False

    def quit_viewer(_: o3d.visualization.Visualizer) -> bool:
        state["running"] = False
        return False

    vis.register_key_callback(ord("="), zoom_in)
    vis.register_key_callback(ord("+"), zoom_in)
    vis.register_key_callback(ord("-"), zoom_out)
    vis.register_key_callback(ord("_"), zoom_out)
    vis.register_key_callback(334, zoom_in)   # numpad +
    vis.register_key_callback(333, zoom_out)  # numpad -
    vis.register_key_callback(ord("Q"), quit_viewer)
    vis.register_key_callback(ord("q"), quit_viewer)

    print("Controls: '+' / '=' zoom in, '-' zoom out, 'q' quit.")
    try:
        while state["running"]:
            vis.poll_events()
            vis.update_renderer()
    finally:
        vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and visualize a global map from xyz frames and poses."
    )
    parser.add_argument("xyz_dir", type=Path, help="Path to folder with .xyz frames.")
    parser.add_argument("poses_path", type=Path, help="Path to poses .npy file.")
    parser.add_argument(
        "end_episode",
        nargs="?",
        type=int,
        default=None,
        help="End frame index (exclusive). If omitted, use all frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    xyz_dir = args.xyz_dir.expanduser().resolve()
    poses_path = args.poses_path.expanduser().resolve()

    if not xyz_dir.exists() or not xyz_dir.is_dir():
        raise NotADirectoryError(f"xyz folder does not exist or is not a directory: {xyz_dir}")
    if not poses_path.exists() or not poses_path.is_file():
        raise FileNotFoundError(f"poses file does not exist: {poses_path}")
    if args.end_episode is not None and args.end_episode <= 0:
        raise ValueError("end_episode must be > 0")

    xyz_files = collect_xyz_files(xyz_dir)
    poses = np.load(poses_path)

    global_map = build_global_map(
        xyz_files=xyz_files,
        poses=poses,
        end=args.end_episode,
    )

    visualize_points(global_map)


if __name__ == "__main__":
    main()
