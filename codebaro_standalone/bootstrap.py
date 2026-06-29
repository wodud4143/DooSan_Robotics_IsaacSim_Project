"""CLI parsing and Isaac Sim / USD stage bootstrap."""

import argparse
import os

from isaacsim import SimulationApp

from . import runtime


def parse_standalone_args():
    parser = argparse.ArgumentParser(
        description="Isaac Sim standalone ROS 2 right-arm controller"
    )
    parser.add_argument(
        "--usd",
        default="/home/rokey/Downloads/CodeBaro/codebaro_world/dual_suction_tf_barcode_LR.usd",
        help="열 로봇/컨베이어 월드 USD 파일의 절대 경로",
    )
    parser.add_argument(
        "--sorter-topic",
        default="/sorter_switch",
        help="Sorter binary_switch를 제어할 ROS 2 Int32 topic",
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=40.0,
        help="박스 자동 소환 간격(초)",
    )
    parser.add_argument(
        "--no-spawn-on-start",
        default=False,
        help="프로그램 시작 직후 첫 박스 자동 소환을 비활성화",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="GUI 없이 실행",
    )
    args, _ = parser.parse_known_args()
    return args


def bootstrap():
    """Create SimulationApp, enable extensions, and open the USD stage."""
    if runtime.simulation_app is not None:
        return runtime.simulation_app

    args = parse_standalone_args()
    runtime.CLI_ARGS = args
    runtime.WORLD_USD_PATH = os.path.abspath(os.path.expanduser(args.usd))
    runtime.SORTER_ROS_TOPIC = str(args.sorter_topic)
    runtime.SPAWN_INTERVAL_SEC = max(0.1, float(args.spawn_interval))
    runtime.SPAWN_ON_START = False  # Preserve the original startup behavior.

    # Isaac/Omni modules must be imported only after SimulationApp exists.
    runtime.simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "renderer": "RayTracedLighting",
        }
    )

    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from omni.physx import get_physx_scene_query_interface

    enable_extension("isaacsim.asset.gen.conveyor")
    enable_extension("isaacsim.ros2.bridge")
    for _ in range(20):
        runtime.simulation_app.update()

    if not os.path.isfile(runtime.WORLD_USD_PATH):
        runtime.simulation_app.close()
        raise FileNotFoundError(
            f"USD world file not found: {runtime.WORLD_USD_PATH}"
        )

    usd_context = omni.usd.get_context()
    runtime.usd_context = usd_context
    stage_open_state = {
        "done": False,
        "success": False,
        "error": "",
    }

    def _on_stage_opened(success, error_message):
        stage_open_state["done"] = True
        stage_open_state["success"] = bool(success)
        stage_open_state["error"] = str(error_message or "")

    open_request_accepted = usd_context.open_stage(
        runtime.WORLD_USD_PATH,
        on_finish_fn=_on_stage_opened,
    )

    if open_request_accepted is False:
        runtime.simulation_app.close()
        raise RuntimeError(
            f"Failed to request USD world open: {runtime.WORLD_USD_PATH}"
        )

    while runtime.simulation_app.is_running() and not stage_open_state["done"]:
        runtime.simulation_app.update()

    if not runtime.simulation_app.is_running():
        runtime.simulation_app.close()
        raise RuntimeError("Isaac Sim closed while opening the USD world.")

    if not stage_open_state["success"]:
        error_text = stage_open_state["error"] or "unknown USD open error"
        runtime.simulation_app.close()
        raise RuntimeError(
            f"Failed to open USD world: {runtime.WORLD_USD_PATH} | {error_text}"
        )

    stage = usd_context.get_stage()
    if stage is None:
        runtime.simulation_app.close()
        raise RuntimeError(
            f"USD stage is unavailable after open: {runtime.WORLD_USD_PATH}"
        )

    runtime.stage = stage
    for _ in range(10):
        runtime.simulation_app.update()

    runtime.physx_query = get_physx_scene_query_interface()
    return runtime.simulation_app
