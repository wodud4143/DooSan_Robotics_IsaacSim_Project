"""ROS 2 subscribers and publishers for right/left process state."""

import time

import numpy as np
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Int32

from .config import *

class RightCenterSubscriber(Node):
    def __init__(self):
        super().__init__("right_robot_center_subscriber")

        # 박스 좌표 상태
        self.target = None
        self.candidate_anchor = None
        self.latest_candidate = None
        self.candidate_since = None
        self.accepting_targets = False

        # 공정 상태. 새 박스는 항상 WAIT_RIGHT_INITIAL에서 시작한다.
        self.process_state = STATE_WAIT_RIGHT_INITIAL
        self.expected_left_mode = None
        self.final_sorter_value = None
        self.final_route_source = None

        # 왼팔 180도 검사 결과와 완료 이벤트를 같은 사이클에서 묶는다.
        self.left_inspection_result = None
        self.left_inspection_result_received = False
        self.left_inspection_result_time = None

        # /barcode_exist_R == 1 직행 사이클 동안 모든 ROS 입력을 무시한다.
        # 이 잠금은 목적지 배치와 Home 복귀가 완료된 뒤에만 해제한다.
        self.input_lock_active = False
        self.input_lock_reason = None

        # 고정 목적지 직행 분기용 /barcode_exist_R 최신 상태
        self.right_barcode_value = 0
        self.right_barcode_received = False
        self.right_barcode_received_time = None

        # /barcode_exist_R 연속 1 유지 시간 추적.
        # 1이 처음 들어온 시각을 저장하고, 0이 들어오면 즉시 초기화한다.
        self.right_barcode_one_since = None
        self.right_barcode_stable_logged = False

        # 외부 왼팔 회전 신호에 의해 확정되는 목적지 직행 상태.
        self.right_pregrip_monitoring_active = False
        self.cycle_route_armed = False
        self.direct_destination_latched = False
        self.direct_route_source = None
        self.sorter_switch_value_for_cycle = None
        self.barcode_r_direct_override_latched = False

        # 좌표보다 먼저 도착한 왼팔 회전 완료 신호도 보존한다.
        self.left_rotate_180_value = 0
        self.left_rotate_180_received = False
        self.left_rotate_180_pending = False
        self.left_rotate_180_received_time = None

        self.left_rotate_90_value = 0
        self.left_rotate_90_received = False
        self.left_rotate_90_pending = False
        self.left_rotate_90_received_time = None

        # 회전 후 검사용 /barcode_exist_L 상태.
        # 검사 창 안에서 1이 처음 수신된 시각부터 연속 유지 시간을 측정하고,
        # 0이 한 번이라도 수신되면 즉시 초기화한다.
        self.left_barcode_value = 0
        self.left_barcode_received = False
        self.left_barcode_received_time = None
        self.left_barcode_one_since = None
        self.left_barcode_stable_logged = False
        self.left_inspection_enabled = False

        # /right_extra_done 지연 발행 상태.
        # None이면 예약 없음, 정수이면 남은 update_simulation 프레임 수.
        self.pending_right_extra_done_frames = None

        self.subscription = self.create_subscription(
            Point,
            CENTER_TOPIC,
            self.center_cb,
            10,
        )

        self.right_barcode_subscription = self.create_subscription(
            Int32,
            RIGHT_BARCODE_TOPIC,
            self.right_barcode_cb,
            10,
        )

        self.left_barcode_subscription = self.create_subscription(
            Int32,
            LEFT_BARCODE_TOPIC,
            self.left_barcode_cb,
            10,
        )

        self.left_rotate_180_subscription = self.create_subscription(
            Int32,
            LEFT_ROTATE_180_TOPIC,
            self.left_rotate_180_cb,
            10,
        )

        self.left_rotate_90_subscription = self.create_subscription(
            Int32,
            LEFT_ROTATE_90_TOPIC,
            self.left_rotate_90_cb,
            10,
        )

        self.inspection_result_subscription = self.create_subscription(
            Int32,
            INSPECTION_RESULT_TOPIC,
            self.inspection_result_cb,
            10,
        )

        self.right_phase_done_publisher = self.create_publisher(
            Int32,
            RIGHT_PHASE_DONE_TOPIC,
            10,
        )

        self.right_extra_done_publisher = self.create_publisher(
            Int32,
            RIGHT_EXTRA_DONE_TOPIC,
            10,
        )

        self.sorter_switch_publisher = self.create_publisher(
            Int32,
            SORTER_SWITCH_TOPIC,
            10,
        )

        self.get_logger().info(f"subscribed coordinate: {CENTER_TOPIC}")
        self.get_logger().info(
            f"subscribed direct-destination selector: {RIGHT_BARCODE_TOPIC}"
        )
        self.get_logger().info(
            f"subscribed post-rotation detector: {LEFT_BARCODE_TOPIC}"
        )
        self.get_logger().info(
            f"subscribed left 180 trigger: {LEFT_ROTATE_180_TOPIC}"
        )
        self.get_logger().info(
            f"subscribed left 90 trigger: {LEFT_ROTATE_90_TOPIC}"
        )
        self.get_logger().info(
            f"subscribed left inspection result: {INSPECTION_RESULT_TOPIC}"
        )
        self.get_logger().info(
            f"publishing sorter command: {SORTER_SWITCH_TOPIC}"
        )
        self.get_logger().info(
            f"publishing phase completion: {RIGHT_PHASE_DONE_TOPIC}"
        )
        self.get_logger().info(
            f"publishing extra-rotation completion: {RIGHT_EXTRA_DONE_TOPIC}"
        )

    def publish_sorter_switch(self, value):
        """분류기 전환 명령 Int32를 한 번 발행한다."""
        value = int(value)
        if value not in (0, 1):
            raise ValueError("sorter_switch value must be 0 or 1")

        message = Int32()
        message.data = value
        self.sorter_switch_publisher.publish(message)

        self.get_logger().info(
            f"published {SORTER_SWITCH_TOPIC}: data={message.data}"
        )

    def publish_right_phase_done(self):
        """일반 오른팔 배치 완료 신호 Int32(data=1)을 한 번 발행한다."""
        message = Int32()
        message.data = 1
        self.right_phase_done_publisher.publish(message)

        self.get_logger().info(
            f"published {RIGHT_PHASE_DONE_TOPIC}: data={message.data}"
        )

    def publish_right_extra_done(self):
        """
        /barcode_exist_L 검출 후 추가 90도 회전 배치 완료 신호를 발행한다.
        """
        message = Int32()
        message.data = 1
        self.right_extra_done_publisher.publish(message)

        self.get_logger().info(
            f"published {RIGHT_EXTRA_DONE_TOPIC}: data={message.data}"
        )

    def schedule_right_extra_done_after_release(self):
        """
        박스 해제 후 1초 뒤 /right_extra_done=1 발행을 예약한다.

        이 메서드는 대기하지 않는다. 이후 Home 복귀 중 실행되는
        update_simulation()에서 프레임 카운트를 줄여 발행한다.
        """
        self.pending_right_extra_done_frames = int(
            RIGHT_PHASE_DONE_DELAY_FRAMES
        )

        self.get_logger().info(
            f"scheduled {RIGHT_EXTRA_DONE_TOPIC}=1 after "
            f"{RIGHT_PHASE_DONE_DELAY_SECONDS:.1f}s "
            f"({self.pending_right_extra_done_frames} simulation frames)"
        )

    def process_delayed_publishers(self):
        """매 시뮬레이션 프레임에서 예약된 완료 신호를 처리한다."""
        if self.pending_right_extra_done_frames is None:
            return

        self.pending_right_extra_done_frames -= 1

        if self.pending_right_extra_done_frames > 0:
            return

        self.pending_right_extra_done_frames = None
        self.publish_right_extra_done()

    def has_pending_right_extra_done(self):
        return self.pending_right_extra_done_frames is not None

    def reset_coordinate_candidate(self, clear_locked_target=True):
        self.candidate_anchor = None
        self.latest_candidate = None
        self.candidate_since = None

        if clear_locked_target:
            self.target = None

    def lock_all_inputs_until_home(self, reason):
        """
        현재 사이클이 끝나고 Home 복귀가 완료될 때까지 모든 ROS 입력을 잠근다.

        rclpy.spin_once()는 계속 실행되지만 coordinate, barcode L, barcode R
        콜백은 상태를 변경하지 않고 즉시 반환한다.
        """
        self.input_lock_active = True
        self.input_lock_reason = str(reason)
        self.accepting_targets = False
        self.disarm_left_barcode_inspection()

        self.get_logger().info(
            "ALL ROS INPUTS LOCKED UNTIL HOME: "
            f"{self.input_lock_reason}"
        )

    def unlock_all_inputs_after_home(self):
        """Home 복귀 완료 후 입력 잠금을 해제하고 이전 바코드 상태를 폐기한다."""
        was_locked = self.input_lock_active

        self.input_lock_active = False
        self.input_lock_reason = None
        self.right_pregrip_monitoring_active = False
        self.cycle_route_armed = False
        self.direct_destination_latched = False
        self.direct_route_source = None
        self.sorter_switch_value_for_cycle = None
        self.barcode_r_direct_override_latched = False

        self.left_rotate_180_value = 0
        self.left_rotate_180_received = False
        self.left_rotate_180_pending = False
        self.left_rotate_180_received_time = None

        self.left_rotate_90_value = 0
        self.left_rotate_90_received = False
        self.left_rotate_90_pending = False
        self.left_rotate_90_received_time = None

        # 이전 /barcode_exist_R 상태가 다음 사이클에 재사용되지 않도록 초기화.
        self.right_barcode_value = 0
        self.right_barcode_received = False
        self.right_barcode_received_time = None
        self.right_barcode_one_since = None
        self.right_barcode_stable_logged = False

        self.disarm_left_barcode_inspection()

        if was_locked:
            self.get_logger().info(
                "ALL ROS INPUTS UNLOCKED AFTER HOME"
            )

    def _clear_route_state(self):
        self.right_pregrip_monitoring_active = False
        self.cycle_route_armed = False
        self.direct_destination_latched = False
        self.direct_route_source = None
        self.sorter_switch_value_for_cycle = None
        self.barcode_r_direct_override_latched = False

    def _clear_right_barcode_state(self):
        self.right_barcode_value = 0
        self.right_barcode_received = False
        self.right_barcode_received_time = None
        self.right_barcode_one_since = None
        self.right_barcode_stable_logged = False

    def _clear_left_completion_state(self):
        self.left_rotate_180_value = 0
        self.left_rotate_180_received = False
        self.left_rotate_180_pending = False
        self.left_rotate_180_received_time = None

        self.left_rotate_90_value = 0
        self.left_rotate_90_received = False
        self.left_rotate_90_pending = False
        self.left_rotate_90_received_time = None

        self.left_inspection_result = None
        self.left_inspection_result_received = False
        self.left_inspection_result_time = None

    def reset_for_next_box(self):
        """현재 박스 사이클을 완전히 종료하고 다음 박스를 오른팔부터 시작한다."""
        self.input_lock_active = False
        self.input_lock_reason = None
        self.process_state = STATE_WAIT_RIGHT_INITIAL
        self.expected_left_mode = None
        self.final_sorter_value = None
        self.final_route_source = None
        self.pending_right_extra_done_frames = None

        self._clear_route_state()
        self._clear_right_barcode_state()
        self._clear_left_completion_state()
        self.disarm_left_barcode_inspection()
        self.reset_coordinate_candidate(clear_locked_target=True)
        self.accepting_targets = True

        self.get_logger().info(
            "PROCESS RESET: next box must start with RIGHT initial inspection; "
            f"waiting {TARGET_STABLE_SECONDS:.1f}s for stable right target"
        )

    def arm_for_next_target(self):
        """호환용 이름. 다음 박스를 항상 오른팔 초기 검사 상태로 시작한다."""
        self.reset_for_next_box()

    def mark_right_initial_busy(self):
        if self.process_state != STATE_WAIT_RIGHT_INITIAL:
            raise RuntimeError(
                f"Cannot start right initial cycle from {self.process_state}"
            )
        self.process_state = STATE_RIGHT_INITIAL_BUSY
        self.accepting_targets = False
        self.get_logger().info("PROCESS STATE -> RIGHT_INITIAL_BUSY")

    def prepare_waiting_for_left(self, left_mode):
        """오른팔 초기 검사를 끝내고 같은 상자의 왼팔 90/180도 작업만 허용한다."""
        left_mode = int(left_mode)
        if left_mode not in (90, 180):
            raise ValueError("left_mode must be 90 or 180")

        self.input_lock_active = False
        self.input_lock_reason = None
        self.process_state = STATE_WAIT_LEFT
        self.expected_left_mode = left_mode
        self.final_sorter_value = None
        self.final_route_source = None

        self._clear_route_state()
        self._clear_right_barcode_state()
        self._clear_left_completion_state()
        self.disarm_left_barcode_inspection()
        self.reset_coordinate_candidate(clear_locked_target=True)
        self.accepting_targets = False

        self.get_logger().info(
            f"PROCESS STATE -> WAIT_LEFT ({left_mode} deg); "
            "right coordinate reception remains locked"
        )

    def mark_left_busy(self):
        if self.process_state != STATE_WAIT_LEFT:
            raise RuntimeError(
                f"Cannot start left cycle from {self.process_state}"
            )
        self.process_state = STATE_LEFT_BUSY
        self.get_logger().info("PROCESS STATE -> LEFT_BUSY")

    def arm_final_delivery(self, sorter_value, source):
        """왼팔 작업 완료 후 같은 상자의 오른팔 최종 배송 좌표 수신을 허용한다."""
        sorter_value = int(sorter_value)
        if sorter_value not in (0, 1):
            raise ValueError("final sorter value must be 0 or 1")

        if self.process_state == STATE_WAIT_RIGHT_FINAL:
            # 동일 이벤트의 ROS 재수신은 멱등 처리한다.
            return

        if self.process_state not in (STATE_WAIT_LEFT, STATE_LEFT_BUSY):
            self.get_logger().warning(
                f"ignored final-delivery arm from state={self.process_state}, "
                f"source={source}, sorter={sorter_value}"
            )
            return

        self.input_lock_active = False
        self.input_lock_reason = None
        self.process_state = STATE_WAIT_RIGHT_FINAL
        self.final_sorter_value = sorter_value
        self.final_route_source = str(source)

        self._clear_route_state()
        self._clear_right_barcode_state()
        self.disarm_left_barcode_inspection()
        self.reset_coordinate_candidate(clear_locked_target=True)
        self.accepting_targets = True

        self.get_logger().info(
            f"PROCESS STATE -> WAIT_RIGHT_FINAL; source={source}, "
            f"sorter={sorter_value}; waiting stable right target"
        )

    def rearm_final_target_after_failure(self):
        """최종 배송 파지 실패 시 동일 상자의 오른팔 최종 좌표를 다시 기다린다."""
        self.input_lock_active = False
        self.input_lock_reason = None
        self.process_state = STATE_WAIT_RIGHT_FINAL
        self._clear_route_state()
        self._clear_right_barcode_state()
        self.reset_coordinate_candidate(clear_locked_target=True)
        self.accepting_targets = True
        self.get_logger().info(
            "RIGHT final grip failed; final target reception re-armed"
        )

    def mark_right_final_busy(self):
        if self.process_state != STATE_WAIT_RIGHT_FINAL:
            raise RuntimeError(
                f"Cannot start right final delivery from {self.process_state}"
            )
        if self.final_sorter_value not in (0, 1):
            raise RuntimeError("Final sorter value is unresolved")
        self.process_state = STATE_RIGHT_FINAL_BUSY
        self.accepting_targets = False
        self.get_logger().info(
            f"PROCESS STATE -> RIGHT_FINAL_BUSY; sorter={self.final_sorter_value}"
        )

    def try_arm_final_delivery_from_left_topics(self):
        """왼팔 완료 토픽과 검사 결과가 모두 준비되면 최종 배송을 연다."""
        if self.process_state not in (STATE_WAIT_LEFT, STATE_LEFT_BUSY):
            return False

        if self.expected_left_mode == 90:
            if not self.left_rotate_90_pending:
                return False
            self.arm_final_delivery(0, "left_rotate_90_complete")
            return True

        if self.expected_left_mode == 180:
            if not (
                self.left_rotate_180_pending
                and self.left_inspection_result_received
            ):
                return False

            sorter_value = (
                0 if int(self.left_inspection_result) == 1 else 1
            )
            self.arm_final_delivery(
                sorter_value,
                "left_rotate_180_complete_barcode_"
                + str(int(self.left_inspection_result)),
            )
            return True

        return False

    def get_process_state(self):
        return self.process_state

    def get_final_sorter_value(self):
        return self.final_sorter_value

    def consume_target(self):
        if self.target is None:
            return None

        target = self.target.copy()
        self.target = None
        self.accepting_targets = False
        return target

    def get_right_barcode_one_elapsed(self):
        """현재 /barcode_exist_R 연속 1 유지 시간을 초 단위로 반환한다."""
        if not (
            self.right_barcode_received
            and self.right_barcode_value == BARCODE_TRIGGER_VALUE
            and self.right_barcode_one_since is not None
        ):
            return 0.0

        return max(
            0.0,
            time.monotonic() - float(self.right_barcode_one_since),
        )

    def is_right_barcode_one_stable(self):
        """
        /barcode_exist_R이 중간의 0 없이 연속 1초 이상 유지됐는지 반환한다.
        """
        stable = (
            self.right_barcode_received
            and self.right_barcode_value == BARCODE_TRIGGER_VALUE
            and self.right_barcode_one_since is not None
            and self.get_right_barcode_one_elapsed()
            >= RIGHT_BARCODE_STABLE_SECONDS
        )

        if stable and not self.right_barcode_stable_logged:
            self.right_barcode_stable_logged = True
            self.get_logger().info(
                f"{RIGHT_BARCODE_TOPIC} stayed at "
                f"{BARCODE_TRIGGER_VALUE} for "
                f"{self.get_right_barcode_one_elapsed():.3f}s; "
                "direct-destination condition confirmed"
            )

        return bool(stable)

    def should_go_direct_destination(self):
        """
        /barcode_exist_R이 연속 1초 이상 1일 때만 고정 목적지 직행을 선택한다.
        """
        return self.is_right_barcode_one_stable()

    def poll_cycle_route_trigger(self):
        """
        안정된 오른팔 좌표와 barcode_R 메시지가 준비되면 즉시 접근을 시작한다.

        - barcode_R == 0:
            회전/검사 경로로 접근을 시작한다.
        - barcode_R == 1이 이미 1초 연속 유지됨:
            sorter=0 직행 경로로 접근을 시작한다.
        - barcode_R == 1이지만 아직 1초 미만:
            접근은 즉시 시작하되 임시로 회전 경로를 유지한다. 접근 중
            poll_right_barcode_until_grip()이 계속 감시하여 파지 완료 전에
            1초 연속 조건이 충족되면 sorter=0 직행으로 전환한다.

        따라서 barcode_R=1의 안정화가 끝날 때까지 로봇이 정지해 기다리지 않는다.
        """
        if self.process_state != STATE_WAIT_RIGHT_INITIAL:
            return False

        if self.input_lock_active or self.target is None:
            return False

        if self.cycle_route_armed:
            return True

        # 경로를 결정하려면 현재 사이클에서 새 barcode_R 메시지가 최소 한 번은 필요하다.
        if not self.right_barcode_received:
            return False

        if self.is_right_barcode_one_stable():
            self.cycle_route_armed = True
            self.direct_destination_latched = True
            self.direct_route_source = "barcode_R_direct"
            self.sorter_switch_value_for_cycle = 0
            self.right_pregrip_monitoring_active = True
            self.barcode_r_direct_override_latched = False

            self.get_logger().info(
                "initial right route armed: barcode_R direct, sorter=0"
            )
            return True

        barcode_value = int(self.right_barcode_value)

        if barcode_value == 0:
            route_source = "barcode_R_rotate"
            log_text = (
                "initial right route armed immediately: "
                "barcode_R=0, rotate/inspect path"
            )
        elif barcode_value == BARCODE_TRIGGER_VALUE:
            # 1초 연속 조건을 기다리며 정지하지 않는다. 접근 중 계속 확인한다.
            route_source = "barcode_R_pending_stability"
            log_text = (
                "initial right approach armed immediately: barcode_R=1 is "
                "not stable yet; monitoring continues until grip completion"
            )
        else:
            self.get_logger().warning(
                f"unsupported {RIGHT_BARCODE_TOPIC} value={barcode_value}; "
                "waiting for 0 or 1"
            )
            return False

        self.cycle_route_armed = True
        self.direct_destination_latched = False
        self.direct_route_source = route_source
        self.sorter_switch_value_for_cycle = None
        self.right_pregrip_monitoring_active = True
        self.barcode_r_direct_override_latched = False
        self.get_logger().info(log_text)
        return True

    def poll_right_barcode_until_grip(self):
        """
        파지 전 감시 구간에서 /barcode_exist_R이 연속 1초 이상 1이면:
          - sorter=0으로 확정
          - 모든 ROS 입력 잠금
          - 목적지 직행 상태 유지

        중간에 data=0이 들어오면 연속 유지 타이머는 즉시 초기화된다.
        """
        if not self.right_pregrip_monitoring_active:
            return False

        if self.input_lock_active:
            return self.barcode_r_direct_override_latched

        if not self.is_right_barcode_one_stable():
            return False

        self.cycle_route_armed = True
        self.barcode_r_direct_override_latched = True
        self.direct_destination_latched = True
        self.direct_route_source = "barcode_R_direct_override"
        self.sorter_switch_value_for_cycle = 0
        self.right_pregrip_monitoring_active = False

        stable_elapsed = self.get_right_barcode_one_elapsed()

        self.get_logger().info(
            f"{RIGHT_BARCODE_TOPIC} == {BARCODE_TRIGGER_VALUE} maintained "
            f"for {stable_elapsed:.3f}s before grip completion; "
            "sorter=0 and direct destination locked"
        )

        self.lock_all_inputs_until_home(
            reason=(
                f"{RIGHT_BARCODE_TOPIC} == {BARCODE_TRIGGER_VALUE} "
                f"maintained for {stable_elapsed:.3f}s before grip completion"
            )
        )

        return True

    def stop_right_barcode_monitoring_at_grip(self):
        """
        파지 함수가 반환된 직후 감시를 종료하고 현재 사이클 입력을 잠근다.

        barcode R==1이 이미 검출된 경우에는 기존 잠금을 유지한다.
        그렇지 않으면 파지 완료 시점의 최신 값으로 sorter를 최종 확정한다.
        """
        self.right_pregrip_monitoring_active = False

        if self.input_lock_active:
            return

        barcode_r_value = (
            int(self.right_barcode_value)
            if self.right_barcode_received
            else None
        )

        if self.direct_destination_latched:
            self.sorter_switch_value_for_cycle = (
                1
                if (
                    self.direct_route_source == "left_rotate_180"
                    and self.right_barcode_received
                    and barcode_r_value == 0
                )
                else 0
            )
        else:
            # barcode R==0 회전 검사 경로는 목적지 이동 전 sorter를 사용하지 않는다.
            self.sorter_switch_value_for_cycle = None

        self.lock_all_inputs_until_home(
            reason="grip attempt completed; route inputs frozen"
        )

        self.get_logger().info(
            f"pre-grip barcode monitoring stopped; "
            f"source={self.direct_route_source}, "
            f"direct={self.direct_destination_latched}, "
            f"{RIGHT_BARCODE_TOPIC}={barcode_r_value}, "
            f"final sorter={self.sorter_switch_value_for_cycle}"
        )

    def has_cycle_route(self):
        return bool(self.cycle_route_armed)

    def has_external_direct_route(self):
        return bool(self.direct_destination_latched)

    def get_direct_route_source(self):
        return self.direct_route_source

    def get_sorter_switch_value_for_cycle(self):
        return self.sorter_switch_value_for_cycle

    def is_barcode_r_direct_override_latched(self):
        return bool(self.barcode_r_direct_override_latched)

    def arm_left_barcode_inspection(self):
        """
        현재 파지 사이클의 왼쪽 바코드 검사를 활성화한다.

        실제 판정은 각 90도 회전 뒤 1초 검사 창 안에서
        /barcode_exist_L == 1이 연속 LEFT_BARCODE_STABLE_SECONDS 이상
        유지됐을 때만 검출로 인정한다.
        """
        self.left_inspection_enabled = True
        self.reset_left_barcode_window_state()

        self.get_logger().info(
            f"left barcode inspection armed: {LEFT_BARCODE_TOPIC}; "
            f"required continuous 1 duration="
            f"{LEFT_BARCODE_STABLE_SECONDS:.3f}s"
        )

    def reset_left_barcode_window_state(self):
        """현재 검사 창의 barcode_L 연속 유지 상태를 완전히 초기화한다."""
        self.left_barcode_value = 0
        self.left_barcode_received = False
        self.left_barcode_received_time = None
        self.left_barcode_one_since = None
        self.left_barcode_stable_logged = False

    def disarm_left_barcode_inspection(self):
        self.left_inspection_enabled = False
        self.reset_left_barcode_window_state()

    def begin_left_inspection_window(self):
        """
        새 1초 검사 창을 시작한다.

        회전 중이나 이전 검사 창에서 들어온 1은 재사용하지 않는다.
        이 호출 이후 새로 들어온 1부터 0.5초 연속 유지 시간을 측정한다.
        """
        self.reset_left_barcode_window_state()
        self.get_logger().info(
            f"left barcode inspection window started; "
            f"{LEFT_BARCODE_TOPIC}=1 must remain continuous for "
            f"{LEFT_BARCODE_STABLE_SECONDS:.3f}s"
        )

    def get_left_barcode_one_elapsed(self):
        """현재 barcode_L 연속 1 유지 시간을 초 단위로 반환한다."""
        if not (
            self.left_inspection_enabled
            and self.left_barcode_received
            and self.left_barcode_value == BARCODE_TRIGGER_VALUE
            and self.left_barcode_one_since is not None
        ):
            return 0.0

        return max(
            0.0,
            time.monotonic() - float(self.left_barcode_one_since),
        )

    def is_left_barcode_one_stable(self):
        """
        현재 검사 창에서 /barcode_exist_L이 중간의 0 없이
        0.5초 이상 연속으로 1을 유지했는지 반환한다.
        """
        stable = (
            self.left_inspection_enabled
            and self.left_barcode_received
            and self.left_barcode_value == BARCODE_TRIGGER_VALUE
            and self.left_barcode_one_since is not None
            and self.get_left_barcode_one_elapsed()
            >= LEFT_BARCODE_STABLE_SECONDS
        )

        if stable and not self.left_barcode_stable_logged:
            self.left_barcode_stable_logged = True
            self.get_logger().info(
                f"{LEFT_BARCODE_TOPIC} stayed at "
                f"{BARCODE_TRIGGER_VALUE} for "
                f"{self.get_left_barcode_one_elapsed():.3f}s; "
                "left barcode detection confirmed"
            )

        return bool(stable)

    def right_barcode_cb(self, msg):
        if self.input_lock_active:
            return

        # 실제 파지가 끝난 뒤에는 /barcode_exist_R이 현재 사이클의
        # 경로를 변경하지 못하게 한다. 다음 Home 대기에서 다시 활성화된다.
        if not (
            self.accepting_targets
            or self.target is not None
            or self.right_pregrip_monitoring_active
        ):
            return

        previous_value = self.right_barcode_value
        value = int(msg.data)
        now = time.monotonic()

        self.right_barcode_value = value
        self.right_barcode_received = True
        self.right_barcode_received_time = now

        if value == BARCODE_TRIGGER_VALUE:
            # 0 -> 1 전환 시점부터 연속 유지 시간을 측정한다.
            if (
                previous_value != BARCODE_TRIGGER_VALUE
                or self.right_barcode_one_since is None
            ):
                self.right_barcode_one_since = now
                self.right_barcode_stable_logged = False
                self.get_logger().info(
                    f"{RIGHT_BARCODE_TOPIC} == {BARCODE_TRIGGER_VALUE}; "
                    f"{RIGHT_BARCODE_STABLE_SECONDS:.1f}s continuous timer started"
                )
        else:
            # 1초가 되기 전이든 후든 0이 한 번 들어오면 연속 유지 조건을 해제한다.
            if self.right_barcode_one_since is not None:
                elapsed = max(
                    0.0,
                    now - float(self.right_barcode_one_since),
                )
                self.get_logger().info(
                    f"{RIGHT_BARCODE_TOPIC} continuous-1 timer reset "
                    f"after {elapsed:.3f}s because value={value}"
                )

            self.right_barcode_one_since = None
            self.right_barcode_stable_logged = False

        if value != previous_value:
            self.get_logger().info(
                f"{RIGHT_BARCODE_TOPIC} changed: {previous_value} -> {value}"
            )

    def inspection_result_cb(self, msg):
        if self.process_state not in (STATE_WAIT_LEFT, STATE_LEFT_BUSY):
            return

        value = int(msg.data)
        if value not in (0, 1):
            self.get_logger().warning(
                f"invalid {INSPECTION_RESULT_TOPIC} value: {value}"
            )
            return

        self.left_inspection_result = value
        self.left_inspection_result_received = True
        self.left_inspection_result_time = time.monotonic()
        self.get_logger().info(
            f"received {INSPECTION_RESULT_TOPIC}={value}"
        )
        self.try_arm_final_delivery_from_left_topics()

    def left_rotate_180_cb(self, msg):
        if self.process_state not in (STATE_WAIT_LEFT, STATE_LEFT_BUSY):
            return

        value = int(msg.data)
        previous_value = self.left_rotate_180_value
        self.left_rotate_180_value = value
        self.left_rotate_180_received = True
        self.left_rotate_180_received_time = time.monotonic()

        if value == LEFT_ROTATE_TRIGGER_VALUE:
            self.left_rotate_180_pending = True

        if value != previous_value:
            self.get_logger().info(
                f"{LEFT_ROTATE_180_TOPIC} changed: "
                f"{previous_value} -> {value}"
            )

        self.try_arm_final_delivery_from_left_topics()

    def left_rotate_90_cb(self, msg):
        if self.process_state not in (STATE_WAIT_LEFT, STATE_LEFT_BUSY):
            return

        value = int(msg.data)
        previous_value = self.left_rotate_90_value
        self.left_rotate_90_value = value
        self.left_rotate_90_received = True
        self.left_rotate_90_received_time = time.monotonic()

        if value == LEFT_ROTATE_TRIGGER_VALUE:
            self.left_rotate_90_pending = True

        if value != previous_value:
            self.get_logger().info(
                f"{LEFT_ROTATE_90_TOPIC} changed: "
                f"{previous_value} -> {value}"
            )

        self.try_arm_final_delivery_from_left_topics()

    def left_barcode_cb(self, msg):
        # 일반 잠금 중에는 무시한다.
        # 단, barcode_R==0 회전 검사 구간에서는 /barcode_exist_L만 허용한다.
        if self.input_lock_active and not self.left_inspection_enabled:
            return

        # 실제 1초 검사 창이 활성화되지 않은 동안의 값은 판정에 사용하지 않는다.
        if not self.left_inspection_enabled:
            return

        previous_value = self.left_barcode_value
        value = int(msg.data)
        now = time.monotonic()

        self.left_barcode_value = value
        self.left_barcode_received = True
        self.left_barcode_received_time = now

        if value == BARCODE_TRIGGER_VALUE:
            if (
                previous_value != BARCODE_TRIGGER_VALUE
                or self.left_barcode_one_since is None
            ):
                self.left_barcode_one_since = now
                self.left_barcode_stable_logged = False
                self.get_logger().info(
                    f"{LEFT_BARCODE_TOPIC} == {BARCODE_TRIGGER_VALUE}; "
                    f"{LEFT_BARCODE_STABLE_SECONDS:.3f}s continuous timer started"
                )
        else:
            if self.left_barcode_one_since is not None:
                elapsed = max(
                    0.0,
                    now - float(self.left_barcode_one_since),
                )
                self.get_logger().info(
                    f"{LEFT_BARCODE_TOPIC} continuous-1 timer reset "
                    f"after {elapsed:.3f}s because value={value}"
                )

            self.left_barcode_one_since = None
            self.left_barcode_stable_logged = False

        if value != previous_value:
            self.get_logger().info(
                f"{LEFT_BARCODE_TOPIC} changed: {previous_value} -> {value}"
            )

    def center_cb(self, msg):
        # /barcode_exist_R == 1 직행 사이클에서는 Home 복귀까지 좌표도 무시한다.
        if self.input_lock_active:
            return

        # 로봇 동작 중에는 새 좌표를 무시한다.
        if not self.accepting_targets:
            return

        point = np.array(
            [float(msg.x), float(msg.y), float(msg.z)],
            dtype=float,
        )
        now = time.monotonic()

        if self.candidate_anchor is None:
            self.candidate_anchor = point.copy()
            self.latest_candidate = point.copy()
            self.candidate_since = now
            self.get_logger().info(f"candidate started: {point}")
            return

        delta = float(np.linalg.norm(point - self.candidate_anchor))

        if delta > TARGET_STABLE_TOLERANCE:
            self.candidate_anchor = point.copy()
            self.latest_candidate = point.copy()
            self.candidate_since = now
            last_log = getattr(self, "_last_candidate_log_time", 0.0)
            if now - last_log >= 0.5:
                self.get_logger().info(
                    f"candidate changed by {delta:.6f} m; "
                    f"stability timer reset: {point}"
                )
                self._last_candidate_log_time = now
            return


        self.latest_candidate = point.copy()
        stable_elapsed = now - float(self.candidate_since)

        if stable_elapsed >= TARGET_STABLE_SECONDS:
            self.target = self.latest_candidate.copy()
            self.accepting_targets = False
            self.get_logger().info(
                f"stable target locked after "
                f"{stable_elapsed:.2f}s: {self.target}"
            )

class L_LeftCenterSubscriber(Node):
    def __init__(self):
        super().__init__("left_robot_center_subscriber")

        self.target = None
        self.candidate_anchor = None
        self.latest_candidate = None
        self.candidate_since = None
        self.accepting_targets = False

        # 오른팔이 박스를 내려놓고 phase 완료 신호를 보낸 뒤에만 왼팔 좌표 수신을 허용한다.
        self.accepting_start = False
        self.start_requested = False
        self.start_msg_count = 0
        self.last_start_time = None
        self.start_event_topic = None
        self.left_rotate_step_limit = None

        # /barcode_exist_R 검사 상태
        # 90도 회전이 끝난 뒤 reset_barcode_check()로 초기화하고,
        # 1초 정지 중 새 메시지를 받았는지 확인한다.
        self.barcode_exist = 0
        self.barcode_msg_received = False
        self.barcode_msg_count = 0
        self.last_barcode_time = None

        self.subscription = self.create_subscription(
            Point,
            L_CENTER_TOPIC,
            self.center_cb,
            10,
        )

        self.barcode_subscription = self.create_subscription(
            Int32,
            L_BARCODE_TOPIC,
            self.barcode_cb,
            10,
        )

        self.start_subscription = self.create_subscription(
            Int32,
            L_RIGHT_PHASE_DONE_TOPIC,
            self.right_phase_done_cb,
            10,
        )

        self.extra_done_subscription = self.create_subscription(
            Int32,
            L_RIGHT_EXTRA_DONE_TOPIC,
            self.right_extra_done_cb,
            10,
        )

        self.left_rotate_180_pub = self.create_publisher(
            Int32,
            L_LEFT_ROTATE_180_TOPIC,
            10,
        )
        self.left_rotate_90_pub = self.create_publisher(
            Int32,
            L_LEFT_ROTATE_90_TOPIC,
            10,
        )

        self.left_success_pub = self.create_publisher(
            Int32,
            L_LEFT_SCAN_SUCCESS_TOPIC,
            10,
        )
        self.left_failed_pub = self.create_publisher(
            Int32,
            L_LEFT_SCAN_FAILED_TOPIC,
            10,
        )
        self.inspection_result_pub = self.create_publisher(
            Int32,
            L_INSPECTION_RESULT_TOPIC,
            10,
        )

        self.get_logger().info(
            f"subscribed: {L_CENTER_TOPIC}"
        )
        self.get_logger().info(
            f"subscribed: {L_BARCODE_TOPIC}"
        )
        self.get_logger().info(
            f"subscribed: {L_RIGHT_PHASE_DONE_TOPIC}"
        )
        self.get_logger().info(
            f"subscribed: {L_RIGHT_EXTRA_DONE_TOPIC}"
        )
        self.get_logger().info(
            f"publishing: {L_LEFT_ROTATE_180_TOPIC}, {L_LEFT_ROTATE_90_TOPIC}, {L_LEFT_SCAN_SUCCESS_TOPIC}, {L_LEFT_SCAN_FAILED_TOPIC}, {L_INSPECTION_RESULT_TOPIC}"
        )

    def arm_for_right_phase_done_start(self):
        """오른팔이 박스를 내려놓고 보낸 완료 이벤트가 들어온 뒤에만 왼팔 phase를 시작한다."""
        self.target = None
        self.candidate_anchor = None
        self.latest_candidate = None
        self.candidate_since = None
        self.accepting_targets = False
        self.accepting_start = True
        self.start_requested = False
        self.start_msg_count = 0
        self.last_start_time = None
        self.start_event_topic = None
        self.left_rotate_step_limit = None
        self.get_logger().info(
            f"waiting for {L_RIGHT_PHASE_DONE_TOPIC} or {L_RIGHT_EXTRA_DONE_TOPIC} == {L_TRIGGER_VALUE} before accepting left target"
        )

    def _accept_start_event(self, topic_name, value, left_rotate_steps):
        self.start_msg_count += 1
        self.last_start_time = time.monotonic()

        self.get_logger().info(
            f"right event from {topic_name}: {value}"
        )

        if not self.accepting_start:
            return

        if int(value) != L_TRIGGER_VALUE:
            return

        self.start_requested = True
        self.accepting_start = False
        self.accepting_targets = True
        self.start_event_topic = topic_name
        self.left_rotate_step_limit = int(left_rotate_steps)
        self.target = None
        self.candidate_anchor = None
        self.latest_candidate = None
        self.candidate_since = None

        self.get_logger().info(
            f"start accepted from {topic_name}; left rotate steps = {self.left_rotate_step_limit}; left target reception is now armed"
        )

    def right_phase_done_cb(self, msg):
        # 오른팔이 360도 완료 후 박스를 내려놓은 경우: 왼팔은 90도 x 2 = 180도 모드.
        self._accept_start_event(
            topic_name=L_RIGHT_PHASE_DONE_TOPIC,
            value=int(msg.data),
            left_rotate_steps=2,
        )

    def right_extra_done_cb(self, msg):
        # 오른팔이 barcode_L 감지 후 추가 90도까지 완료하고 박스를 내려놓은 경우: 왼팔은 90도 x 1 모드.
        self._accept_start_event(
            topic_name=L_RIGHT_EXTRA_DONE_TOPIC,
            value=int(msg.data),
            left_rotate_steps=1,
        )

    def publish_int(self, publisher, value):
        msg = Int32()
        msg.data = int(value)
        publisher.publish(msg)

    def publish_left_rotate_180(self):
        """왼팔이 총 180도 회전을 끝냈을 때 발행한다."""
        self.publish_int(self.left_rotate_180_pub, 1)
        self.get_logger().info(
            f"published left 180 rotation done: {L_LEFT_ROTATE_180_TOPIC}=1"
        )

    def publish_left_rotate_90(self):
        """왼팔이 90도 회전을 끝냈을 때 발행한다."""
        self.publish_int(self.left_rotate_90_pub, 1)
        self.get_logger().info(
            f"published left 90 rotation done: {L_LEFT_ROTATE_90_TOPIC}=1"
        )

    def publish_left_scan_result(self, detected):
        """왼팔 phase 최종 barcode 판정 결과를 이벤트 토픽으로 발행한다."""
        if detected:
            self.publish_int(self.left_success_pub, 1)
            self.publish_int(self.left_failed_pub, 0)
            self.publish_int(self.inspection_result_pub, L_RESULT_OK)
            self.get_logger().info(
                f"published OK: {L_LEFT_SCAN_SUCCESS_TOPIC}=1, {L_INSPECTION_RESULT_TOPIC}={L_RESULT_OK}"
            )
        else:
            self.publish_int(self.left_success_pub, 0)
            self.publish_int(self.left_failed_pub, 1)
            self.publish_int(self.inspection_result_pub, L_RESULT_FAIL)
            self.get_logger().info(
                f"published FAIL: {L_LEFT_SCAN_FAILED_TOPIC}=1, {L_INSPECTION_RESULT_TOPIC}={L_RESULT_FAIL}"
            )

    def reset_barcode_check(self):
        """이번 1초 검사 구간에서 새 barcode 메시지를 받았는지 확인하기 위해 초기화한다."""
        self.barcode_exist = 0
        self.barcode_msg_received = False
        self.barcode_msg_count = 0
        self.last_barcode_time = None

    def barcode_cb(self, msg):
        new_value = int(msg.data)
        changed = (new_value != self.barcode_exist)
        self.barcode_exist = new_value
        self.barcode_msg_received = True
        self.barcode_msg_count += 1
        self.last_barcode_time = time.monotonic()

        if changed:
            self.get_logger().info(
                f"barcode from {L_BARCODE_TOPIC}: {self.barcode_exist}"
            )

    def barcode_check_result(self):
        """현재 검사 구간의 오른팔 카메라 barcode 확인 결과를 문자열과 함께 반환한다."""
        if not self.barcode_msg_received:
            return "no_message", False

        if self.barcode_exist == L_BARCODE_DETECTED_VALUE:
            return "detected", True

        return "not_detected", False


    def consume_target(self):
        if self.target is None:
            return None

        target = self.target.copy()

        self.target = None
        self.accepting_targets = False

        return target

    def center_cb(self, msg):
        # 로봇 동작 중에는 들어오는 좌표를 무시한다.
        if not self.accepting_targets:
            return

        point = np.array(
            [float(msg.x), float(msg.y), float(msg.z)],
            dtype=float,
        )
        now = time.monotonic()

        if self.candidate_anchor is None:
            self.candidate_anchor = point.copy()
            self.latest_candidate = point.copy()
            self.candidate_since = now

            self.get_logger().info(
                f"candidate started: {point}"
            )
            return

        # 최초 후보 좌표로부터의 누적 이동량을 검사한다.
        delta = float(
            np.linalg.norm(point - self.candidate_anchor)
        )

        if delta > L_TARGET_STABLE_TOLERANCE:
            self.candidate_anchor = point.copy()
            self.latest_candidate = point.copy()
            self.candidate_since = now
            last_log = getattr(self, "_last_candidate_log_time", 0.0)
            if now - last_log >= 0.5:
                self.get_logger().info(
                    f"candidate changed by {delta:.6f} m; "
                    f"stability timer reset: {point}"
                )
                self._last_candidate_log_time = now
            return


        # 작은 측정 노이즈는 허용하고 최신 좌표를 저장한다.
        self.latest_candidate = point.copy()
        stable_elapsed = now - float(self.candidate_since)

        if stable_elapsed >= L_TARGET_STABLE_SECONDS:
            self.target = self.latest_candidate.copy()
            self.accepting_targets = False

            self.get_logger().info(
                f"stable target locked after "
                f"{stable_elapsed:.2f}s: {self.target}"
            )
