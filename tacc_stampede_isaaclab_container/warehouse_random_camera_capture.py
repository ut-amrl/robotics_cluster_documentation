#!/usr/bin/env python3
"""Capture warehouse RGB/depth/semantic data from random camera poses.

Run this script inside an Isaac Sim / Isaac Lab container environment.
"""

import argparse
import os
import random
import sys

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open the warehouse scene, randomize camera pose, and capture RGB/depth/"
            "semantic segmentation data."
        )
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random camera captures to generate (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./_out_warehouse_random_camera",
        help="Output directory for captured data (default: ./_out_warehouse_random_camera).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height in pixels (default: 720).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run headless (recommended for remote nodes).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)

    simulation_app = SimulationApp({"headless": args.headless})

    import carb
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.utils.nucleus import get_assets_root_path

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Could not resolve Isaac Sim assets root. Make sure container assets are available."
        )

    stage_rel_path = "/Isaac/Samples/Replicator/Stage/full_warehouse_worker_and_anim_cameras.usd"
    stage_path = assets_root + stage_rel_path
    carb.log_info(f"Opening stage: {stage_path}")

    usd_context = omni.usd.get_context()
    usd_context.open_stage(stage_path)
    for _ in range(120):
        simulation_app.update()
        stage = usd_context.get_stage()
        if stage is not None:
            break
    else:
        raise RuntimeError(f"Failed to load stage: {stage_path}")

    camera = rep.create.camera(
        position=(0.0, 0.0, 3.0),
        look_at=(0.0, 0.0, 1.2),
        clipping_range=(0.05, 1000.0),
    )
    render_product = rep.create.render_product(camera, (args.width, args.height))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=args.output_dir,
        rgb=True,
        distance_to_camera=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=False,
    )
    writer.attach([render_product])

    # Bounds chosen to keep camera inside/near warehouse aisles.
    x_bounds = (-14.0, 14.0)
    y_bounds = (-8.0, 8.0)
    z_bounds = (1.0, 4.0)

    for idx in range(args.num_samples):
        cam_pos = (
            random.uniform(*x_bounds),
            random.uniform(*y_bounds),
            random.uniform(*z_bounds),
        )
        look_at = (
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(0.8, 2.0),
        )

        with camera:
            rep.modify.pose(position=cam_pos, look_at=look_at)

        rep.orchestrator.step(rt_subframes=4)
        carb.log_info(
            f"[{idx + 1}/{args.num_samples}] Captured pose "
            f"pos={tuple(round(v, 3) for v in cam_pos)} "
            f"look_at={tuple(round(v, 3) for v in look_at)}"
        )

    writer.detach()
    simulation_app.update()
    simulation_app.close()
    print(f"Done. Wrote {args.num_samples} captures to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise
