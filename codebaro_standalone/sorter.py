"""Conveyor sorter ActionGraph control and ROS subscriber."""

import omni.graph.core as og
from rclpy.node import Node
from std_msgs.msg import Int32

from .config import SORTER_GRAPH_PATH, SORTER_ROS_TOPIC, SORTER_SWITCH_ATTR

SORTER_GRAPH = None
SORTER_CURRENT_VALUE = False

def initialize_sorter_graph():
    """USD의 컨베이어 Sorter ActionGraph를 확인하고 초기 OFF로 설정한다."""
    global SORTER_GRAPH, SORTER_CURRENT_VALUE

    SORTER_GRAPH = og.get_graph_by_path(SORTER_GRAPH_PATH)

    print("")
    print("==============================================")
    print("=== CHECK SORTER ACTIONGRAPH ===")
    print("GRAPH_PATH :", SORTER_GRAPH_PATH)
    print("SWITCH_ATTR:", SORTER_SWITCH_ATTR)
    print("graph is None:", SORTER_GRAPH is None)
    print(
        "graph valid:",
        SORTER_GRAPH.is_valid() if SORTER_GRAPH is not None else False,
    )
    print("==============================================")

    if SORTER_GRAPH is None or not SORTER_GRAPH.is_valid():
        raise RuntimeError(
            "Sorter ActionGraph invalid: " + SORTER_GRAPH_PATH
        )

    SORTER_CURRENT_VALUE = False
    set_sorter_switch(False, source="startup")


def set_sorter_switch(value, source="ROS"):
    """Sorter ActionGraph의 binary_switch를 즉시 갱신한다."""
    global SORTER_CURRENT_VALUE

    value = bool(value)
    SORTER_CURRENT_VALUE = value

    if SORTER_GRAPH is None or not SORTER_GRAPH.is_valid():
        print("[WARN] sorter graph is not initialized; command ignored:", value)
        return

    og.Controller.attribute(SORTER_SWITCH_ATTR).set(value)
    og.Controller.evaluate_sync(SORTER_GRAPH)

    print(
        f"[conveyor][{source}] binary_switch =",
        "ON / DIVERT" if value else "OFF / STRAIGHT",
    )


def maintain_sorter_graph():
    """현재 sorter 값을 매 시뮬레이션 프레임 ActionGraph에 유지한다."""
    if SORTER_GRAPH is None or not SORTER_GRAPH.is_valid():
        return

    og.Controller.attribute(SORTER_SWITCH_ATTR).set(
        bool(SORTER_CURRENT_VALUE)
    )
    og.Controller.evaluate_sync(SORTER_GRAPH)

class ConveyorSorterSubscriber(Node):
    """Int32 0/1을 받아 컨베이어 Sorter ActionGraph를 제어한다."""

    def __init__(self):
        super().__init__("conveyor_sorter_standalone_subscriber")

        self.subscription = self.create_subscription(
            Int32,
            SORTER_ROS_TOPIC,
            self.sorter_cb,
            10,
        )

        self.get_logger().info(f"subscribed sorter command: {SORTER_ROS_TOPIC}")
        self.get_logger().info("0 -> STRAIGHT, 1 -> DIVERT")

    def sorter_cb(self, msg):
        value = int(msg.data)

        if value == 1:
            set_sorter_switch(True, source=SORTER_ROS_TOPIC)
        elif value == 0:
            set_sorter_switch(False, source=SORTER_ROS_TOPIC)
        else:
            self.get_logger().warning(
                f"invalid sorter value {value}; use 0 or 1"
            )
