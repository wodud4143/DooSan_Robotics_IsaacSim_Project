"""Integrated dual-arm workflow orchestration and shutdown."""

import numpy as np
import omni.timeline
import rclpy
import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper

from . import runtime
from .background_home import *
from .config import *
from .left_arm import *
from .right_arm import *
from .ros_nodes import L_LeftCenterSubscriber, RightCenterSubscriber
from .simulation import update_simulation
from .sorter import ConveyorSorterSubscriber, initialize_sorter_graph
from .spawner import start_box_spawner

simulation_app = runtime.simulation_app

def _hold_right_home_for_signal_delay(ctrl, home_q):
    """Home 자세를 유지하며 완료 이벤트 발행 지연 시간을 확보한다."""
    for _ in range(RIGHT_PHASE_DONE_DELAY_FRAMES):
        set_drive_targets_from_rad(
            RIGHT_ROBOT,
            ctrl["active_joints"],
            home_q,
        )
        ctrl["q"] = home_q.copy()
        ctrl["qd"] = np.zeros_like(home_q)
        update_simulation()


def execute_right_initial_cycle(
    node,
    ctrl,
    initial_pos,
    initial_orientation,
    initial_joint_q,
    cycle_index,
):
    """새 상자를 항상 오른팔이 먼저 검사하는 초기 사이클."""
    raw_target = node.consume_target()
    if raw_target is None:
        return False

    # 이전 사이클의 비블로킹 복귀가 아직 남아 있더라도 새 오른팔 작업이
    # 시작되면 충돌하는 Home 명령을 즉시 중단한다.
    wait_for_background_right_home_completion("RIGHT initial cycle started")
    node.mark_right_initial_busy()

    direct_destination_requested = node.has_external_direct_route()
    direct_route_source = node.get_direct_route_source()
    sorter_value_for_cycle = node.get_sorter_switch_value_for_cycle()

    if FORCE_LEVEL_TOP_GRASP:
        target_orientation = make_level_top_grasp_orientation(
            initial_orientation
        )
    elif KEEP_CAMERA_DOWN_ORIENTATION:
        target_orientation = initial_orientation.copy()
    else:
        target_orientation = RIGHT_GRASP_ORIENTATION.copy()

    print("")
    print("====================================================")
    print(f"=== RIGHT INITIAL CYCLE {cycle_index} START ===")
    print("surface target:", raw_target)
    print("route source:", direct_route_source)
    print("direct destination provisional:", direct_destination_requested)
    print("provisional sorter:", sorter_value_for_cycle)
    print("====================================================")

    command_right_grippers_open(settle_frames=5)

    (
        grip_ok,
        contact_ee,
        contact_tcp,
        target_orientation,
    ) = approach_right_box_surface(
        ctrl=ctrl,
        surface_target=raw_target,
        initial_orientation=initial_orientation,
        target_orientation=target_orientation,
    )

    # 초기 파지 직전까지 barcode_R==1 안정 조건을 반영한다.
    node.stop_right_barcode_monitoring_at_grip()

    direct_destination_requested = node.has_external_direct_route()
    direct_route_source = node.get_direct_route_source()
    sorter_value_for_cycle = node.get_sorter_switch_value_for_cycle()

    if not grip_ok:
        print("[FAIL] Right initial grip failed.")
        open_right_grippers()
        home_reached = return_right_robot_home(
            ctrl=ctrl,
            home_q=initial_joint_q,
            home_ee_pos=initial_pos,
            home_ee_orientation=initial_orientation,
            safe_lift_first=True,
        )
        if not home_reached:
            raise RuntimeError("Right Home was not reached after initial grip failure.")
        node.reset_for_next_box()
        return False

    _, lift_target = lift_right_robot(
        ctrl=ctrl,
        orientation=target_orientation,
    )

    if direct_destination_requested:
        # 시나리오 1: 왼팔 요청 토픽을 절대 발행하지 않는다.
        if sorter_value_for_cycle != 0:
            raise RuntimeError(
                f"Scenario 1 sorter must be 0, got {sorter_value_for_cycle}"
            )

        final_orientation = target_orientation.copy()
        node.disarm_left_barcode_inspection()
        node.publish_sorter_switch(0)

        place_position = place_right_robot_fixed_destination(
            ctrl=ctrl,
            place_orientation=final_orientation,
            grasp_ee_z=contact_ee[2],
        )

        open_right_grippers()
        retreat_right_robot(
            ctrl=ctrl,
            start_pos=place_position,
            retreat_orientation=final_orientation,
        )

        home_reached = return_right_robot_home(
            ctrl=ctrl,
            home_q=initial_joint_q,
            home_ee_pos=initial_pos,
            home_ee_orientation=initial_orientation,
            safe_lift_first=False,
        )
        if not home_reached:
            raise RuntimeError("Right Home was not reached after scenario 1.")

        print("[SCENARIO 1] sorter=0, no left-arm request published")
        node.reset_for_next_box()
        return True

    # 시나리오 2/3/4의 오른팔 초기 회전 검사.
    node.arm_left_barcode_inspection()
    (
        final_orientation,
        final_cumulative_deg,
        left_extra_rotation_performed,
    ) = rotate_until_left_barcode_then_extra_90(
        ctrl=ctrl,
        fixed_position=lift_target,
        start_orientation=target_orientation,
        barcode_node=node,
    )

    print(
        "right final signed rotation deg:",
        RIGHT_ROTATE_SIGN * final_cumulative_deg,
    )

    place_position = place_right_robot_on_original_grasp_axis(
        ctrl=ctrl,
        place_orientation=final_orientation,
        original_grasp_ee_position=contact_ee,
    )
    node.disarm_left_barcode_inspection()

    # 그리퍼 open 명령만 먼저 내리고, 해제 안정화/이탈보다 먼저 다음 팔 신호를 발행한다.
    open_right_grippers(settle_frames=0)

    if left_extra_rotation_performed:
        node.prepare_waiting_for_left(90)
        node.publish_right_extra_done()
        route_label = "SCENARIO 2 OPEN-IMMEDIATE HANDOFF: LEFT 90"
    else:
        node.prepare_waiting_for_left(180)
        node.publish_right_phase_done()
        route_label = "SCENARIO 3/4 OPEN-IMMEDIATE HANDOFF: LEFT 180"

    # open 명령과 완료 토픽을 같은 프레임에 큐잉한 뒤 한 프레임 갱신하여
    # 왼팔이 즉시 좌표 수신을 시작하게 한다.
    update_simulation()
    print(
        "[PIPELINE] RIGHT gripper opened; LEFT start signal published immediately."
    )

    # 신호 발행 후에 박스 분리 안정화와 안전 이탈을 수행한다.
    wait_frames(RELEASE_SETTLE_FRAMES)
    retreat_right_robot(
        ctrl=ctrl,
        start_pos=place_position,
        retreat_orientation=final_orientation,
    )

    print(
        "[PIPELINE] RIGHT release settling/retreat continues after LEFT handoff."
    )

    # # execute_left_cycle()가 블로킹으로 실행되는 동안에도 update_simulation()
    # # 안에서 오른팔 Home 복귀가 계속 진행된다.
    # reverse_rotation_deg = 0.0
    # reverse_reference_orientation = None

    # # barcode_L 미검출로 정확히 270도까지 회전한 경로는 Home 관절각으로
    # # 바로 보내지 않는다. 270 -> 180 -> 90 -> 0 순으로 역회전한 뒤
    # # 관절 Home 복귀를 수행한다.
    # if (
    #     not left_extra_rotation_performed
    #     and abs(
    #         float(final_cumulative_deg)
    #         - float(EE_ROTATE_TOTAL_DEG)
    #     ) <= 1.0e-6
    # ):
    #     reverse_rotation_deg = float(final_cumulative_deg)
    #     reverse_reference_orientation = target_orientation.copy()
    #     print(
    #         "[RIGHT 270 PATH] Exact reverse unwind required before Home:",
    #         f"{reverse_rotation_deg:.0f} -> 0 deg",
    #     )

    start_background_right_home(
        ctrl=ctrl,
        home_q=initial_joint_q,
    )
    print(
        "[PIPELINE] RIGHT Home return is running in parallel with LEFT work."
    )

    print("")
    print("====================================================")
    print(f"=== RIGHT INITIAL CYCLE {cycle_index} DONE ===")
    print(route_label)
    print("====================================================")
    return True


def execute_right_final_delivery(
    node,
    ctrl,
    initial_pos,
    initial_orientation,
    initial_joint_q,
    cycle_index,
):
    """왼팔 작업 뒤 이미 확정된 sorter 값으로 같은 상자를 최종 배송한다."""
    raw_target = node.consume_target()
    if raw_target is None:
        return False

    wait_for_background_right_home_completion("RIGHT final-delivery cycle started")
    sorter_value = node.get_final_sorter_value()
    route_source = node.final_route_source
    node.mark_right_final_busy()

    if FORCE_LEVEL_TOP_GRASP:
        target_orientation = make_level_top_grasp_orientation(
            initial_orientation
        )
    elif KEEP_CAMERA_DOWN_ORIENTATION:
        target_orientation = initial_orientation.copy()
    else:
        target_orientation = RIGHT_GRASP_ORIENTATION.copy()

    print("")
    print("====================================================")
    print(f"=== RIGHT FINAL DELIVERY {cycle_index} START ===")
    print("surface target:", raw_target)
    print("final route source:", route_source)
    print("final sorter:", sorter_value)
    print("barcode rotation inspection: DISABLED")
    print("====================================================")

    command_right_grippers_open(settle_frames=5)
    (
        grip_ok,
        contact_ee,
        contact_tcp,
        target_orientation,
    ) = approach_right_box_surface(
        ctrl=ctrl,
        surface_target=raw_target,
        initial_orientation=initial_orientation,
        target_orientation=target_orientation,
    )

    if not grip_ok:
        print("[FAIL] Right final-delivery grip failed.")
        open_right_grippers()
        home_reached = return_right_robot_home(
            ctrl=ctrl,
            home_q=initial_joint_q,
            home_ee_pos=initial_pos,
            home_ee_orientation=initial_orientation,
            safe_lift_first=True,
        )
        if not home_reached:
            raise RuntimeError("Right Home was not reached after final grip failure.")
        node.rearm_final_target_after_failure()
        return False

    lift_right_robot(ctrl=ctrl, orientation=target_orientation)

    # 최종 목적지 이송 직전에 정확히 한 번 발행한다.
    node.publish_sorter_switch(sorter_value)
    place_position = place_right_robot_fixed_destination(
        ctrl=ctrl,
        place_orientation=target_orientation,
        grasp_ee_z=contact_ee[2],
    )

    open_right_grippers()
    retreat_right_robot(
        ctrl=ctrl,
        start_pos=place_position,
        retreat_orientation=target_orientation,
    )

    home_reached = return_right_robot_home(
        ctrl=ctrl,
        home_q=initial_joint_q,
        home_ee_pos=initial_pos,
        home_ee_orientation=initial_orientation,
        safe_lift_first=False,
    )
    if not home_reached:
        raise RuntimeError("Right Home was not reached after final delivery.")

    print("")
    print("====================================================")
    print(f"=== RIGHT FINAL DELIVERY {cycle_index} DONE ===")
    print("next box will again start with RIGHT initial inspection")
    print("====================================================")

    node.reset_for_next_box()
    return True


def execute_left_cycle(node, right_node, ctrl, cycle_index):
    """오른팔 요청 후 왼팔이 TCP/Raycast 측면 파지로 상자를 회전한다."""
    raw_target = node.consume_target()
    if raw_target is None:
        return False

    wait_for_background_left_home_completion("LEFT cycle started")
    right_node.mark_left_busy()

    # /box_coordinate_center_L은 실제 박스 측면 중심의 월드 표면 좌표다.
    # 기존 EE 직접 이동용 Y 보정값은 사용하지 않는다.
    surface_target = raw_target.copy()
    target_orientation = np.asarray(
        ctrl["home_ee_orientation"],
        dtype=float,
    ).copy()

    print("")
    print("====================================================")
    print(f"=== INTEGRATED LEFT CYCLE {cycle_index} START ===")
    print("start event:", node.start_event_topic)
    print("raw target:", raw_target)
    print("left side surface target:", surface_target)
    print("approach direction:", L_LEFT_APPROACH_DIRECTION)
    print("target suction normal:", L_get_suction_axis_world(target_orientation))
    print("target wrist3 reference axis:", L_get_wrist3_reference_axis_world(target_orientation))
    print("rotate steps:", node.left_rotate_step_limit)
    print("====================================================")

    L_command_left_grippers_open(settle_frames=5)

    grip_ok, contact_ee, contact_tcp = L_approach_left_box_surface(
        ctrl=ctrl,
        surface_target=surface_target,
        target_orientation=target_orientation,
    )

    if not grip_ok:
        print("[FAIL] LEFT TCP/Raycast side grip failed.")
        L_open_left_grippers()
        home_reached = L_return_left_robot_home(ctrl)
        if not home_reached:
            raise RuntimeError("Left Home was not reached after side-grip failure.")

        # 동일 왼팔 단계 재시도
        right_node.process_state = STATE_WAIT_LEFT
        node.target = None
        node.candidate_anchor = None
        node.latest_candidate = None
        node.candidate_since = None
        node.accepting_start = False
        node.start_requested = True
        node.accepting_targets = True
        node.get_logger().info(
            "left side grip failed; current left phase target reception re-armed"
        )
        return False

    print("[LEFT GRIP] contact EE:", contact_ee)
    print("[LEFT GRIP] contact TCP:", contact_tcp)

    (
        lift_start_position,
        _lift_target_position,
        lift_reached,
        actual_lift_position,
    ) = L_lift_left_robot(
        ctrl=ctrl,
        orientation=target_orientation,
    )

    if not lift_reached:
        print("[WARN] Left lift target was not fully reached.")

    # 상승 직후 upper/lower 두 흡착컵이 모두 계속 붙어 있는지 확인한다.
    lift_gripper_interface = surface_gripper.acquire_surface_gripper_interface()
    lift_gripped_map = L_get_gripped_objects_map(lift_gripper_interface)
    lift_grip_ok = L_have_all_grippers_attached(lift_gripped_map)
    print("[LEFT POST-LIFT GRIP CHECK]", lift_gripped_map)
    print("[LEFT POST-LIFT GRIP OK]", lift_grip_ok)

    if not lift_grip_ok:
        print("[FAIL] LEFT lost the box during lift; rotation will not start.")
        L_open_left_grippers()
        home_reached = L_return_left_robot_home(ctrl)
        if not home_reached:
            raise RuntimeError("Left Home was not reached after lift grip loss.")

        right_node.process_state = STATE_WAIT_LEFT
        node.target = None
        node.candidate_anchor = None
        node.latest_candidate = None
        node.candidate_since = None
        node.accepting_start = False
        node.start_requested = True
        node.accepting_targets = True
        node.get_logger().info(
            "left dual grip lost during lift; left target reception re-armed"
        )
        return False

    left_rotate_step_limit = int(
        node.left_rotate_step_limit or L_LEFT_SCAN_STEPS
    )

    (
        final_orientation,
        barcode_detected_by_right_camera,
        left_scan_steps,
        left_scan_last_status,
    ) = L_scan_left_two_faces_with_right_camera(
        ctrl=ctrl,
        fixed_position=actual_lift_position,
        start_orientation=target_orientation,
        barcode_node=node,
        scan_steps=left_rotate_step_limit,
    )

    print(
        "[LEFT SCAN RESULT]",
        "detected=", barcode_detected_by_right_camera,
        "steps=", left_scan_steps,
        "status=", left_scan_last_status,
    )

    if left_scan_last_status == "grip_or_tcp_lost_during_rotation":
        print("[FAIL] LEFT rotation stopped before normal placement for safety.")

        # 안전 중단 시 공중에서 즉시 그리퍼를 열지 않는다.
        # 아직 한쪽이라도 박스를 잡고 있으면 원래 파지 TCP 위치로 먼저 내려놓는다.
        gripper_interface = surface_gripper.acquire_surface_gripper_interface()
        safety_gripped_map = L_get_gripped_objects_map(gripper_interface)
        still_holding_box = L_has_any_gripper_attached(safety_gripped_map)

        print("safety-stop gripped objects:", safety_gripped_map)
        print("still holding box:", still_holding_box)

        if still_holding_box:
            print(
                "[SAFETY PLACE] LEFT still holds the box; "
                "returning it to the original TCP position before opening."
            )
            safety_place_reached, _ = L_place_left_robot(
                ctrl=ctrl,
                place_tcp_target=contact_tcp,
                place_orientation=final_orientation,
            )
            if not safety_place_reached:
                print("[WARN] LEFT safety placement did not fully converge.")

            L_open_left_grippers()

            safety_retreat_reached, _ = L_retreat_left_robot(
                ctrl=ctrl,
                retreat_orientation=final_orientation,
            )
            if not safety_retreat_reached:
                print("[WARN] LEFT safety retreat did not fully converge.")
        else:
            print(
                "[SAFETY STOP] No LEFT gripper is attached; "
                "the box is already detached. Resetting grippers only."
            )
            L_open_left_grippers()

        home_reached = L_return_left_robot_home(ctrl)
        if not home_reached:
            raise RuntimeError(
                "Left Home was not reached after rotation safety stop."
            )

        # 현재 왼팔 단계를 같은 상자 좌표로 다시 시도할 수 있도록 재무장한다.
        right_node.process_state = STATE_WAIT_LEFT
        node.target = None
        node.candidate_anchor = None
        node.latest_candidate = None
        node.candidate_since = None
        node.accepting_start = False
        node.start_requested = True
        node.accepting_targets = True
        node.get_logger().info(
            "left rotation safety stop; left target reception re-armed"
        )
        return False

    # 회전된 자세에서도 원래 흡착 TCP 위치로 정확히 되돌아가 내려놓는다.
    # 파지 당시 EE 좌표를 재사용하면 자세 변화로 TCP가 이동하므로 contact_tcp를 사용한다.
    place_reached, _place_position = L_place_left_robot(
        ctrl=ctrl,
        place_tcp_target=contact_tcp,
        place_orientation=final_orientation,
    )
    if not place_reached:
        print("[WARN] Left place target was not fully reached.")

    # 그리퍼 open 명령 직후 오른팔 최종 배송 신호를 발행한다.
    L_open_left_grippers(settle_frames=0)

    node.publish_left_scan_result(barcode_detected_by_right_camera)
    if left_rotate_step_limit <= 1:
        node.publish_left_rotate_90()
        expected_sorter = 0
        completion_source = "left_90_open_immediate_complete"
    else:
        node.publish_left_rotate_180()
        expected_sorter = 0 if barcode_detected_by_right_camera else 1
        completion_source = (
            "left_180_open_immediate_complete_barcode_"
            + str(int(bool(barcode_detected_by_right_camera)))
        )

    # 내부 상태도 같은 단계에서 열어 오른팔 좌표 수신을 즉시 시작한다.
    right_node.arm_final_delivery(expected_sorter, completion_source)

    # open 명령과 완료 토픽을 반영하는 최소 한 프레임만 갱신한다.
    update_simulation()
    print(
        "[PIPELINE] LEFT gripper opened; RIGHT final signal published immediately."
    )

    # 신호를 먼저 발행한 뒤 박스 분리 안정화와 안전 이탈을 수행한다.
    L_wait_frames(L_RELEASE_SETTLE_FRAMES)
    retreat_reached, _ = L_retreat_left_robot(
        ctrl=ctrl,
        retreat_orientation=final_orientation,
    )
    if not retreat_reached:
        print("[WARN] Left retreat target was not fully reached.")

    print(
        "[PIPELINE] LEFT release settling/retreat continues after RIGHT handoff."
    )

    # execute_right_final_delivery()가 블로킹으로 실행되는 동안에도
    # update_simulation() 안에서 왼팔 Home 복귀가 계속 진행된다.
    start_background_left_home(ctrl)
    print(
        "[PIPELINE] LEFT Home return is running in parallel with RIGHT work."
    )

    print("")
    print("====================================================")
    print(f"=== INTEGRATED LEFT CYCLE {cycle_index} DONE ===")
    print("final sorter:", expected_sorter)
    print("right final delivery is now armed")
    print("====================================================")

    node.arm_for_right_phase_done_start()
    return True


def hold_both_robots_home(
    right_ctrl,
    right_home_q,
    left_ctrl,
):
    """대기 프레임에서 양팔의 저장된 Home 관절 목표를 유지한다."""
    set_drive_targets_from_rad(
        RIGHT_ROBOT,
        right_ctrl["active_joints"],
        right_home_q,
    )
    right_ctrl["q"] = right_home_q.copy()
    right_ctrl["qd"] = np.zeros_like(right_home_q)

    left_home_q = np.asarray(
        left_ctrl["home_q"],
        dtype=float,
    )
    L_set_drive_targets_from_rad(
        L_LEFT_ROBOT,
        left_ctrl["active_joints"],
        left_home_q,
    )
    left_ctrl["q"] = left_home_q.copy()
    left_ctrl["qd"] = np.zeros_like(left_home_q)


def main():

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    for _ in range(10):
        simulation_app.update()

    L_validate_required_prims()

    if not rclpy.ok():
        rclpy.init(args=[])

    initialize_sorter_graph()

    right_node = RightCenterSubscriber()
    left_node = L_LeftCenterSubscriber()
    conveyor_node = ConveyorSorterSubscriber()
    runtime.ROS_NODE = right_node
    runtime.L_ROS_NODE = left_node
    runtime.CONVEYOR_ROS_NODE = conveyor_node

    right_ctrl = setup_rmp_controller()
    left_ctrl = L_setup_rmp_controller()

    raw_initial_pos, raw_initial_orientation = get_world_pose(
        RIGHT_EE_PATH
    )

    configure_surface_grippers()
    print_attachment_diagnostics()
    L_configure_surface_grippers()
    L_print_attachment_diagnostics()
    command_right_grippers_open(settle_frames=5)
    L_command_left_grippers_open(settle_frames=5)

    # 오른팔은 기존 파일과 동일하게 흡착면 수평 보정 후 Home을 저장한다.
    if INITIAL_LEVEL_HOME:
        (
            right_initial_pos,
            right_initial_orientation,
            right_initial_joint_q,
        ) = initialize_level_home_pose(right_ctrl)
    else:
        right_initial_pos, right_initial_orientation = get_world_pose(
            RIGHT_EE_PATH
        )
        right_initial_joint_q = get_robot_joint_positions(right_ctrl)
        right_ctrl["home_q"] = right_initial_joint_q.copy()

    # 왼팔은 EE 위치를 유지한 채 흡착면 수직 + wrist3 수평 자세로 교정한다.
    if L_INITIAL_SIDE_HOME:
        (
            left_initial_pos,
            left_initial_orientation,
            left_initial_joint_q,
        ) = L_initialize_vertical_side_home_pose(left_ctrl)
    else:
        left_initial_pos, left_initial_orientation = L_get_world_pose(
            L_LEFT_EE_PATH
        )
        left_initial_joint_q = L_get_robot_joint_positions(left_ctrl)
        left_ctrl["home_q"] = left_initial_joint_q.copy()
        left_ctrl["home_ee_position"] = left_initial_pos.copy()
        left_ctrl["home_ee_orientation"] = left_initial_orientation.copy()

    print("")
    print("====================================================")
    print("=== DUAL ARM STANDALONE INITIALIZED ===")
    print("USD:", WORLD_USD_PATH)
    print("right Home EE:", right_initial_pos)
    print("right Home joint deg:", np.rad2deg(right_initial_joint_q))
    print("left Home EE:", left_ctrl["home_ee_position"])
    print("left Home joint deg:", np.rad2deg(left_ctrl["home_q"]))
    print("left Home suction normal:", L_get_suction_axis_world(left_ctrl["home_ee_orientation"]))
    print("left Home wrist3 reference axis:", L_get_wrist3_reference_axis_world(left_ctrl["home_ee_orientation"]))
    print("left Home side-normal error deg:", L_get_side_normal_error_deg(left_ctrl["home_ee_orientation"]))
    print("left Home wrist3 horizontal error deg:", L_get_wrist3_horizontal_error_deg(left_ctrl["home_ee_orientation"]))
    print("scheduler: pipelined RIGHT release -> LEFT, LEFT release -> RIGHT final")
    print("sorter topic:", SORTER_ROS_TOPIC)
    print("sorter graph:", SORTER_GRAPH_PATH)
    print("spawn interval sec:", SPAWN_INTERVAL_SEC)
    print("global process speed scale:", PROCESS_SPEED_SCALE)
    print("raycast contact speed scale:", GRASP_APPROACH_SPEED_SCALE)
    print("right Raycast contact step:", CONTACT_APPROACH_STEP)
    print("left Raycast contact step:", L_CONTACT_APPROACH_STEP)
    print("motion interpolation is accelerated; convergence/settle checks stay at 60 Hz")
    print("right initial level frames:", INITIAL_LEVEL_FRAMES)
    print("right initial convergence frames:", INITIAL_LEVEL_CONVERGENCE_FRAMES)
    print("left initial side frames:", L_INITIAL_SIDE_FRAMES)
    print("left initial convergence frames:", L_INITIAL_SIDE_CONVERGENCE_FRAMES)
    print("spawn on start:", SPAWN_ON_START)
    print("spawn position:", tuple(SPAWN_POSITION))
    print("====================================================")

    start_box_spawner()

    right_cycle_index = 0
    left_cycle_index = 0

    right_node.arm_for_next_target()
    left_node.arm_for_right_phase_done_start()

    while simulation_app.is_running():
        process_state = right_node.get_process_state()

        # 왼팔은 오른팔 초기 검사에서 명시적으로 요청한 경우에만 동작한다.
        if process_state in (STATE_WAIT_LEFT, STATE_LEFT_BUSY):
            if (
                process_state == STATE_WAIT_LEFT
                and left_node.start_requested
                and left_node.target is not None
            ):
                left_cycle_index += 1
                execute_left_cycle(
                    node=left_node,
                    right_node=right_node,
                    ctrl=left_ctrl,
                    cycle_index=left_cycle_index,
                )
                continue

            hold_both_robots_home(
                right_ctrl,
                right_initial_joint_q,
                left_ctrl,
            )
            update_simulation()
            continue

        # 왼팔 작업이 끝난 같은 상자를 오른팔이 최종 목적지로 배송한다.
        if process_state == STATE_WAIT_RIGHT_FINAL:
            if right_node.target is not None:
                right_cycle_index += 1
                execute_right_final_delivery(
                    node=right_node,
                    ctrl=right_ctrl,
                    initial_pos=right_initial_pos,
                    initial_orientation=right_initial_orientation,
                    initial_joint_q=right_initial_joint_q,
                    cycle_index=right_cycle_index,
                )
                continue

            hold_both_robots_home(
                right_ctrl,
                right_initial_joint_q,
                left_ctrl,
            )
            update_simulation()
            continue

        # 새 상자는 항상 이 분기에서 오른팔이 먼저 시작한다.
        if process_state == STATE_WAIT_RIGHT_INITIAL:
            if (
                right_node.target is not None
                and right_node.has_cycle_route()
            ):
                right_cycle_index += 1
                execute_right_initial_cycle(
                    node=right_node,
                    ctrl=right_ctrl,
                    initial_pos=right_initial_pos,
                    initial_orientation=right_initial_orientation,
                    initial_joint_q=right_initial_joint_q,
                    cycle_index=right_cycle_index,
                )
                continue

        hold_both_robots_home(
            right_ctrl,
            right_initial_joint_q,
            left_ctrl,
        )
        update_simulation()


def shutdown():
    for node in (runtime.ROS_NODE, runtime.L_ROS_NODE, runtime.CONVEYOR_ROS_NODE):
        if node is None:
            continue
        try:
            node.destroy_node()
        except Exception as exc:
            print("[WARN] ROS node destroy failed:", repr(exc))

    if rclpy.ok():
        rclpy.shutdown()

    if runtime.simulation_app is not None:
        runtime.simulation_app.close()
