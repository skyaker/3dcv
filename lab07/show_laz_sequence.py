#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from tqdm import tqdm


# Spherical grid parameters (RoboSense RS-32)
_NUM_CHANNELS = 32
_H_RES        = 1800
_ELEV_MIN     = -16.0
_ELEV_MAX     =  15.0


def _build_range_grid(file_path: Path) -> np.ndarray:
    """Return a (NUM_CHANNELS x H_RES) array of ranges (0 = empty)."""
    las = laspy.read(str(file_path))
    xyz = np.vstack((las.x, las.y, las.z)).T
    r = np.linalg.norm(xyz, axis=1)
    valid = r > 0.1
    xyz, r = xyz[valid], r[valid]
    elev = np.degrees(np.arcsin(np.clip(xyz[:, 2] / r, -1.0, 1.0)))
    azim = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    row = np.round(
        (elev - _ELEV_MIN) / (_ELEV_MAX - _ELEV_MIN) * (_NUM_CHANNELS - 1)
    ).astype(int)
    col = (np.round((azim + 180.0) / 360.0 * _H_RES).astype(int)) % _H_RES
    mask = (row >= 0) & (row < _NUM_CHANNELS)
    row, col, r = row[mask], col[mask], r[mask]
    grid = np.zeros((_NUM_CHANNELS, _H_RES), dtype=np.float32)
    order = np.argsort(r)[::-1]  # far first -> near overwrites
    grid[row[order], col[order]] = r[order]
    return grid


def _grid_to_image(grid: np.ndarray, row_scale: int = 20) -> o3d.geometry.Image:
    """Convert range grid to a 2-D depth map image (uint8 RGB).

    Each row = one lidar channel, each column = one azimuth step.
    row_scale stretches rows so the image is tall enough to see clearly.
    Empty cells are black; filled cells are coloured by range (Blue->Red).
    """
    valid = grid > 0
    norm = np.zeros_like(grid, dtype=np.float64)
    if valid.any():
        rmin, rmax = grid[valid].min(), grid[valid].max()
        norm[valid] = np.clip((grid[valid] - rmin) / max(rmax - rmin, 1e-9), 0.0, 1.0)

    anchors_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    anchors_r = np.array([0.0,  0.0, 0.0, 1.0,  1.0], dtype=np.float64)
    anchors_g = np.array([0.0,  1.0, 1.0, 1.0,  0.0], dtype=np.float64)
    anchors_b = np.array([1.0,  1.0, 0.0, 0.0,  0.0], dtype=np.float64)

    n = norm.ravel()
    rgb = np.column_stack((
        np.interp(n, anchors_x, anchors_r),
        np.interp(n, anchors_x, anchors_g),
        np.interp(n, anchors_x, anchors_b),
    )).reshape(_NUM_CHANNELS, _H_RES, 3)

    rgb[~valid] = 0.0
    rgb = np.flipud(rgb)  # flip so ch0 (lowest) is at bottom
    rgb = np.repeat(rgb, row_scale, axis=0)
    return o3d.geometry.Image((rgb * 255).astype(np.uint8))


def intensity_to_rgb(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Robust normalization: ignore extreme outliers and stretch useful range.
    p2, p98 = np.percentile(intensity, [2, 98])
    if p98 <= p2:
        p2 = np.min(intensity)
        p98 = np.max(intensity)
    span = max(p98 - p2, 1e-9)
    norm = np.clip((intensity - p2) / span, 0.0, 1.0)

    # Slight gamma correction makes mid-tones more visible.
    norm = np.power(norm, 0.8)

    # Blue -> Cyan -> Green -> Yellow -> Red
    anchors_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    anchors_r = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    anchors_g = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    anchors_b = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    r = np.interp(norm, anchors_x, anchors_r)
    g = np.interp(norm, anchors_x, anchors_g)
    b = np.interp(norm, anchors_x, anchors_b)
    return np.column_stack((r, g, b))


def laz_to_pointcloud(file_path: Path, use_intensity_color: bool) -> o3d.geometry.PointCloud:
    las = laspy.read(str(file_path))
    xyz = np.vstack((las.x, las.y, las.z)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if use_intensity_color and hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float64)
        colors = intensity_to_rgb(intensity)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def collect_laz_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.laz"))
    if not files:
        raise FileNotFoundError(f"No .laz files found in: {folder}")
    return files


def convert_laz_folder_to_xyz(
    files: list[Path],
    output_dir: Path,
    xyz_fmt: str = "%.6f",
    delimiter: str = " ",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for laz_file in tqdm(files, desc="Converting LAZ -> XYZ"):
        las = laspy.read(str(laz_file))
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        out_path = output_dir / f"{laz_file.stem}.xyz"
        np.savetxt(out_path, xyz, fmt=xyz_fmt, delimiter=delimiter)

    print(f"Saved {len(files)} xyz files to: {output_dir}")


def visualize_sequence(
    files: list[Path],
    fps: float,
    loop: bool,
    use_intensity_color: bool,
) -> None:
    first = laz_to_pointcloud(files[0], use_intensity_color=use_intensity_color)
    if len(first.points) == 0:
        raise ValueError(f"First cloud is empty: {files[0]}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LAZ sequence viewer", width=1280, height=720)

    pcd = first
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

    # --- second window: depth map (2-D range image) ---
    first_grid = _build_range_grid(files[0])
    first_img  = _grid_to_image(first_grid)

    img_h = _NUM_CHANNELS * 20   # row_scale=20
    img_w = _H_RES

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Depth map (spherical projection)", width=img_w, height=img_h,
                       left=0, top=740)
    vis2.add_geometry(first_img)
    vis2.poll_events()
    vis2.update_renderer()

    state2 = {"running": True}

    # --- end second window setup ---

    frame_delay = 1.0 / fps if fps > 0 else 0.0

    try:
        first_frame = True
        while state["running"] and state2["running"]:
            for idx, laz_file in enumerate(files, start=1):
                if not state["running"] or not state2["running"]:
                    break
                if first_frame:
                    first_frame = False
                else:
                    current = laz_to_pointcloud(laz_file, use_intensity_color=use_intensity_color)
                    if len(current.points) == 0:
                        print(f"[skip] {laz_file.name} is empty")
                        continue
                    pcd.points = current.points
                    pcd.colors = current.colors

                    grid = _build_range_grid(laz_file)
                    new_img = _grid_to_image(grid)
                    vis2.clear_geometries()
                    vis2.add_geometry(new_img)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

                vis2.poll_events()
                vis2.update_renderer()

                print(f"[{idx}/{len(files)}] {laz_file.name}")
                if frame_delay > 0:
                    time.sleep(frame_delay)

            if not loop:
                break

        while state["running"]:
            vis.poll_events()
            vis.update_renderer()
            vis2.poll_events()
            vis2.update_renderer()
    finally:
        vis.destroy_window()
        vis2.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show .laz sequence or convert laz -> xyz."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to folder with .laz files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second for switching clouds (default: 5).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop over files continuously.",
    )
    parser.add_argument(
        "--no-intensity-color",
        action="store_true",
        help="Disable colorization by intensity.",
    )
    parser.add_argument(
        "--convert-to-xyz",
        action="store_true",
        help="Convert all .laz files from folder to .xyz and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for .xyz files (default: <folder>/xyz).",
    )
    parser.add_argument(
        "--xyz-fmt",
        type=str,
        default="%.6f",
        help="Floating point format for xyz export (default: %%.6f).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=" ",
        help="Delimiter for xyz export (default: space).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder.expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist or is not a directory: {folder}")

    laz_files = collect_laz_files(folder)
    if args.convert_to_xyz:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else folder / "xyz"
        )
        convert_laz_folder_to_xyz(
            files=laz_files,
            output_dir=output_dir,
            xyz_fmt=args.xyz_fmt,
            delimiter=args.delimiter,
        )
        return

    visualize_sequence(
        files=laz_files,
        fps=args.fps,
        loop=args.loop,
        use_intensity_color=not args.no_intensity_color,
    )


if __name__ == "__main__":
    main()
