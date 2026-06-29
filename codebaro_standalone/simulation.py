"""Per-frame Isaac Sim update loop and shared frame helpers."""

import rclpy

from . import runtime


def update_simulation():
    """ROS callbacks, sorter, spawner, home tasks, and one Isaac frame."""
    if not runtime.simulation_app.is_running():
        raise KeyboardInterrupt

    if rclpy.ok():
        # 같은 프레임에 연계 토픽이 여러 개 도착할 수 있으므로
        # 오른팔/왼팔/컨베이어 노드의 준비된 콜백을 모두 비운다.
        for node in (runtime.ROS_NODE, runtime.L_ROS_NODE, runtime.CONVEYOR_ROS_NODE):
            if node is None:
                continue
            for _ in range(8):
                rclpy.spin_once(node, timeout_sec=0.0)

        # 오른팔 경로 결정 및 지연 발행 처리는 기존 로직을 유지한다.
        if runtime.ROS_NODE is not None:
            if hasattr(runtime.ROS_NODE, "poll_cycle_route_trigger"):
                runtime.ROS_NODE.poll_cycle_route_trigger()
            if hasattr(runtime.ROS_NODE, "poll_right_barcode_until_grip"):
                runtime.ROS_NODE.poll_right_barcode_until_grip()
            if hasattr(runtime.ROS_NODE, "process_delayed_publishers"):
                runtime.ROS_NODE.process_delayed_publishers()

    # 현재 동작 중인 팔과 충돌하지 않는 반대 팔의 비블로킹 Home 복귀를
    # 물리 프레임마다 계속 진행한다.
    from .background_home import process_background_home_tasks
    from .sorter import maintain_sorter_graph
    from .spawner import tick_box_spawner

    process_background_home_tasks()
    maintain_sorter_graph()
    tick_box_spawner()
    runtime.simulation_app.update()


def wait_frames(n):
    for _ in range(int(n)):
        update_simulation()


def L_update_simulation():
    """왼팔 코드가 사용하는 공통 시뮬레이션 갱신 어댑터."""
    update_simulation()
