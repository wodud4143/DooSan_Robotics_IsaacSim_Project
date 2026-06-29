"""Non-blocking background home-return scheduler for both arms."""

import numpy as np

from .config import *
from .config import _dlog
from .left_arm import *
from .right_arm import *
from .simulation import update_simulation

_BACKGROUND_RIGHT_HOME_TASK = None
_BACKGROUND_LEFT_HOME_TASK = None

def _make_background_home_task(
    arm_name,
    ctrl,
    start_q,
    home_q,
    delta_q,
    move_frames,
    tolerance_deg,
    stable_frames,
):
    return {
        "arm": str(arm_name),
        "ctrl": ctrl,
        "start_q": np.asarray(start_q, dtype=float).copy(),
        "home_q": np.asarray(home_q, dtype=float).copy(),
        "delta_q": np.asarray(delta_q, dtype=float).copy(),
        "commanded_home_q": (
            np.asarray(start_q, dtype=float)
            + np.asarray(delta_q, dtype=float)
        ),
        "move_frames": max(1, int(move_frames)),
        "frame": 0,
        "phase": "move",
        "stable_count": 0,
        "settle_frame": 0,
        "tolerance_deg": float(tolerance_deg),
        "stable_frames": max(1, int(stable_frames)),
    }


def _configure_background_right_joint_home_phase(task):
    ctrl = task["ctrl"]
    start_q = get_robot_joint_positions(ctrl)
    home_q = np.asarray(task["home_q"], dtype=float).copy()

    # wrapped_joint_delta 대신 직접 차이 사용
    # → wrist가 -270도 돌아간 상태에서 home(0도)으로 갈 때
    #   +90 순방향이 아니라 -270 역방향 경로를 유지한다
    delta_q = home_q - start_q

    max_delta_deg = float(np.max(np.abs(np.rad2deg(delta_q))))
    adaptive_frames = int(np.ceil(
        max_delta_deg
        / max(float(HOME_MAX_JOINT_SPEED_DEG_PER_SEC), 1.0e-6)
        / PHYSICS_DT
    ))
    move_frames = max(int(HOME_JOINT_FRAMES), adaptive_frames, 1)

    task["start_q"] = start_q
    task["delta_q"] = delta_q
    task["commanded_home_q"] = home_q.copy()
    task["move_frames"] = move_frames
    task["frame"] = 0
    task["phase"] = "move"
    task["stable_count"] = 0
    task["settle_frame"] = 0

    print("[PARALLEL HOME] RIGHT joint-Home phase configured")
    print("current joint deg:", np.rad2deg(start_q))
    print("home joint deg:", np.rad2deg(home_q))
    print("delta deg (no wrap):", np.rad2deg(delta_q))
    print("move frames:", move_frames)


def start_background_right_home(ctrl, home_q):
    """
    오른팔 Home 관절 복귀를 비블로킹으로 예약한다.

    어떤 회전 경로(90·180·270도)로 끝났든 현재 관절각에서
    저장된 home_q로 바로 이동한다. 역회전 없음.
    """
    global _BACKGROUND_RIGHT_HOME_TASK

    home_q = np.asarray(home_q, dtype=float).copy()

    task = {
        "arm": "RIGHT",
        "ctrl": ctrl,
        "home_q": home_q,
        "commanded_home_q": home_q.copy(),
        "move_frames": 0,
        "frame": 0,
        "phase": "move",
        "stable_count": 0,
        "settle_frame": 0,
        "tolerance_deg": float(HOME_JOINT_TOLERANCE_DEG),
        "stable_frames": max(1, int(HOME_STABLE_FRAMES)),
        "unwind_total_frames": 0,       # wait_for_background_right_home_completion 호환용
        "unwind_settle_limit_frames": 0, # 동일
    }

    _configure_background_right_joint_home_phase(task)
    _BACKGROUND_RIGHT_HOME_TASK = task

    print("[PARALLEL HOME] RIGHT joint-Home scheduled (direct, no unwind)")




def start_background_left_home(ctrl):
    """왼팔 Home 복귀를 블로킹하지 않고 예약한다."""
    global _BACKGROUND_LEFT_HOME_TASK

    start_q = L_get_robot_joint_positions(ctrl)
    home_q = np.asarray(ctrl["home_q"], dtype=float).copy()
    delta_q = L_wrapped_joint_delta(home_q, start_q)

    max_delta_deg = float(np.max(np.abs(np.rad2deg(delta_q))))
    adaptive_frames = int(np.ceil(
        max_delta_deg
        / max(float(HOME_MAX_JOINT_SPEED_DEG_PER_SEC), 1.0e-6)
        / L_PHYSICS_DT
    ))
    move_frames = max(int(L_RETURN_HOME_FRAMES), adaptive_frames, 1)

    _BACKGROUND_LEFT_HOME_TASK = _make_background_home_task(
        arm_name="LEFT",
        ctrl=ctrl,
        start_q=start_q,
        home_q=home_q,
        delta_q=delta_q,
        move_frames=move_frames,
        tolerance_deg=L_HOME_JOINT_TOLERANCE_DEG,
        stable_frames=max(4, int(L_MOVE_STABLE_FRAMES)),
    )

    print("")
    print("[PARALLEL HOME] LEFT return scheduled")
    _dlog(DEBUG_JOINT, "current joint deg:", np.rad2deg(start_q))
    _dlog(DEBUG_JOINT, "home joint deg:", np.rad2deg(home_q))
    _dlog(DEBUG_JOINT, "move frames:", move_frames)


def cancel_background_right_home(reason="right arm becomes active"):
    global _BACKGROUND_RIGHT_HOME_TASK
    if _BACKGROUND_RIGHT_HOME_TASK is not None:
        print("[PARALLEL HOME] RIGHT return cancelled:", reason)
    _BACKGROUND_RIGHT_HOME_TASK = None


def cancel_background_left_home(reason="left arm becomes active"):
    global _BACKGROUND_LEFT_HOME_TASK
    if _BACKGROUND_LEFT_HOME_TASK is not None:
        print("[PARALLEL HOME] LEFT return cancelled:", reason)
    _BACKGROUND_LEFT_HOME_TASK = None


def is_background_right_home_active():
    return _BACKGROUND_RIGHT_HOME_TASK is not None


def is_background_left_home_active():
    return _BACKGROUND_LEFT_HOME_TASK is not None


def wait_for_background_right_home_completion(reason):
    """
    동일한 오른팔의 다음 작업을 시작하기 전에 진행 중인 병렬 Home 복귀가
    실제 관절 오차 기준으로 완료될 때까지 기다린다.

    반대 팔 작업은 이미 병렬로 실행됐으므로 정상적인 경우 남은 프레임만
    짧게 기다린다. 수렴하지 않으면 Home 작업을 취소하지 않고 오류로 중단한다.
    """
    global _BACKGROUND_RIGHT_HOME_TASK

    if _BACKGROUND_RIGHT_HOME_TASK is None:
        return True

    task = _BACKGROUND_RIGHT_HOME_TASK
    max_wait_frames = (
        max(0, int(task.get("unwind_total_frames", 0)))
        + max(0, int(task.get("unwind_settle_limit_frames", 0)))
        # 역회전 완료 후 실제 관절 상태로 move_frames를 다시 계산하므로,
        # 아직 0일 수 있다. 최대 관절 복귀 여유를 추가로 확보한다.
        + max(360, int(task.get("move_frames", 0)))
        + max(1, int(HOME_SETTLE_FRAMES))
        + 30
    )

    print(
        "[PARALLEL HOME] RIGHT completion required before:",
        reason,
    )

    for wait_frame in range(max_wait_frames):
        if _BACKGROUND_RIGHT_HOME_TASK is None:
            print(
                "[PARALLEL HOME] RIGHT completion confirmed before:",
                reason,
            )
            return True

        update_simulation()

        if wait_frame == 0 or (wait_frame + 1) % 60 == 0:
            print(
                f"[PARALLEL HOME WAIT RIGHT "
                f"{wait_frame + 1:03d}/{max_wait_frames}]"
            )

    task = _BACKGROUND_RIGHT_HOME_TASK
    if task is None:
        return True

    actual_q = get_robot_joint_positions(task["ctrl"])
    target_q = np.asarray(task["commanded_home_q"], dtype=float)
    error_deg = np.abs(
        np.rad2deg(wrapped_joint_delta(target_q, actual_q))
    )
    max_error_deg = float(np.max(error_deg))

    raise RuntimeError(
        "RIGHT background Home did not converge before the next RIGHT cycle. "
        f"reason={reason}, max_joint_error_deg={max_error_deg:.3f}, "
        f"joint_error_deg={error_deg}"
    )


def wait_for_background_left_home_completion(reason):
    """
    동일한 왼팔의 다음 작업을 시작하기 전에 진행 중인 병렬 Home 복귀가
    실제 관절 오차 기준으로 완료될 때까지 기다린다.
    """
    global _BACKGROUND_LEFT_HOME_TASK

    if _BACKGROUND_LEFT_HOME_TASK is None:
        return True

    task = _BACKGROUND_LEFT_HOME_TASK
    max_wait_frames = (
        max(1, int(task.get("move_frames", 1)))
        + max(1, int(L_HOME_SETTLE_FRAMES))
        + 10
    )

    print(
        "[PARALLEL HOME] LEFT completion required before:",
        reason,
    )

    for wait_frame in range(max_wait_frames):
        if _BACKGROUND_LEFT_HOME_TASK is None:
            print(
                "[PARALLEL HOME] LEFT completion confirmed before:",
                reason,
            )
            return True

        update_simulation()

        if wait_frame == 0 or (wait_frame + 1) % 60 == 0:
            print(
                f"[PARALLEL HOME WAIT LEFT "
                f"{wait_frame + 1:03d}/{max_wait_frames}]"
            )

    task = _BACKGROUND_LEFT_HOME_TASK
    if task is None:
        return True

    actual_q = L_get_robot_joint_positions(task["ctrl"])
    target_q = np.asarray(task["commanded_home_q"], dtype=float)
    error_deg = np.abs(
        np.rad2deg(L_wrapped_joint_delta(target_q, actual_q))
    )
    max_error_deg = float(np.max(error_deg))

    raise RuntimeError(
        "LEFT background Home did not converge before the next LEFT cycle. "
        f"reason={reason}, max_joint_error_deg={max_error_deg:.3f}, "
        f"joint_error_deg={error_deg}"
    )


def _step_background_right_home(task):
    """
    매 프레임 호출되는 배경 Home 복귀 스텝 함수.

    phase == "move"  : 현재 관절각 → home_q 보간
    phase == "settle": 수렴 대기

    반환: True = 완료, False = 진행 중
    """
    ctrl = task["ctrl"]
    phase = task["phase"]

    # ------------------------------------------------------------------
    # move 단계: 관절 보간
    # ------------------------------------------------------------------
    if phase == "move":
        frame = int(task["frame"])
        move_frames = int(task["move_frames"])
        ratio = frame / float(max(move_frames, 1))
        alpha = smoothstep01(ratio)
        q_now = task["start_q"] + task["delta_q"] * alpha
        if frame >= move_frames:
            q_now = task["commanded_home_q"].copy()

        set_drive_targets_from_rad(
            RIGHT_ROBOT,
            ctrl["active_joints"],
            q_now,
        )
        ctrl["q"] = q_now.copy()
        ctrl["qd"] = np.zeros_like(q_now)

        if DEBUG_JOINT and (frame == 0 or frame % 30 == 0 or frame >= move_frames):
            _dlog(DEBUG_JOINT,
                f"[PARALLEL RIGHT HOME {min(frame, move_frames):03d}/{move_frames}]"
            )

        task["frame"] = frame + 1
        if frame >= move_frames:
            task["phase"] = "settle"
        return False

    # ------------------------------------------------------------------
    # settle 단계: 실제 관절 수렴 대기
    # ------------------------------------------------------------------
    target_q = task["commanded_home_q"]
    set_drive_targets_from_rad(
        RIGHT_ROBOT,
        ctrl["active_joints"],
        target_q,
    )
    ctrl["q"] = target_q.copy()
    ctrl["qd"] = np.zeros_like(target_q)

    actual_q = get_robot_joint_positions(ctrl)
    error_deg = np.abs(np.rad2deg(wrapped_joint_delta(target_q, actual_q)))
    max_error_deg = float(np.max(error_deg))

    if max_error_deg <= task["tolerance_deg"]:
        task["stable_count"] += 1
    else:
        task["stable_count"] = 0

    task["settle_frame"] += 1
    if DEBUG_JOINT and (task["settle_frame"] == 1 or task["settle_frame"] % 60 == 0):
            _dlog(DEBUG_JOINT,
                "[PARALLEL RIGHT HOME SETTLE] max error deg:",
                round(max_error_deg, 4),
                "stable:", task["stable_count"],
            )

    if task["stable_count"] >= task["stable_frames"]:
        final_pos, _ = get_world_pose(RIGHT_EE_PATH)
        print("[PARALLEL HOME] RIGHT reached Home; EE:", final_pos)
        return True

    return False



def _step_background_left_home(task):
    ctrl = task["ctrl"]

    if task["phase"] == "move":
        frame = int(task["frame"])
        move_frames = int(task["move_frames"])
        ratio = frame / float(max(move_frames, 1))
        alpha = L_smoothstep01(ratio)
        q_now = task["start_q"] + task["delta_q"] * alpha
        if frame >= move_frames:
            q_now = task["commanded_home_q"].copy()

        L_set_drive_targets_from_rad(
            L_LEFT_ROBOT,
            ctrl["active_joints"],
            q_now,
        )
        ctrl["q"] = q_now.copy()
        ctrl["qd"] = np.zeros_like(q_now)

        if DEBUG_JOINT and (frame == 0 or frame % 30 == 0 or frame >= move_frames):
            _dlog(DEBUG_JOINT,
                f"[PARALLEL LEFT HOME {min(frame, move_frames):03d}/{move_frames}]"
            )

        task["frame"] = frame + 1
        if frame >= move_frames:
            task["phase"] = "settle"
        return False

    target_q = task["commanded_home_q"]
    L_set_drive_targets_from_rad(
        L_LEFT_ROBOT,
        ctrl["active_joints"],
        target_q,
    )
    ctrl["q"] = target_q.copy()
    ctrl["qd"] = np.zeros_like(target_q)

    actual_q = L_get_robot_joint_positions(ctrl)
    error_deg = np.abs(np.rad2deg(L_wrapped_joint_delta(target_q, actual_q)))
    max_error_deg = float(np.max(error_deg))

    if max_error_deg <= task["tolerance_deg"]:
        task["stable_count"] += 1
    else:
        task["stable_count"] = 0

    task["settle_frame"] += 1
    if DEBUG_JOINT and (task["settle_frame"] == 1 or task["settle_frame"] % 60 == 0):
        _dlog(DEBUG_JOINT,
            "[PARALLEL LEFT HOME SETTLE] max error deg:",
            round(max_error_deg, 4),
            "stable:", task["stable_count"],
        )

    if task["stable_count"] >= task["stable_frames"]:
        final_pos, _ = L_get_world_pose(L_LEFT_EE_PATH)
        print("[PARALLEL HOME] LEFT reached Home; EE:", final_pos)
        return True

    return False


def process_background_home_tasks():
    """한 프레임마다 예약된 양팔 Home 복귀를 각각 한 스텝 진행한다."""
    global _BACKGROUND_RIGHT_HOME_TASK, _BACKGROUND_LEFT_HOME_TASK

    if _BACKGROUND_RIGHT_HOME_TASK is not None:
        if _step_background_right_home(_BACKGROUND_RIGHT_HOME_TASK):
            _BACKGROUND_RIGHT_HOME_TASK = None

    if _BACKGROUND_LEFT_HOME_TASK is not None:
        if _step_background_left_home(_BACKGROUND_LEFT_HOME_TASK):
            _BACKGROUND_LEFT_HOME_TASK = None
