"""Left-arm side grasp, scan rotation, placement, and home control."""

import numpy as np
import carb
from isaacsim.core.prims import Articulation
from isaacsim.robot_motion.motion_generation import RmpFlow, interface_config_loader
import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from . import runtime
from .config import *
from .config import _dlog
from .right_arm import _hit_attr, _normalized_attr_name, _vec3_to_numpy
from .simulation import L_update_simulation

stage = runtime.stage
physx_query = runtime.physx_query
_L_ATTACHMENT_JOINT_PATHS_CACHE = None

def L_wait_frames(frame_count):
    for _ in range(int(frame_count)):
        L_update_simulation()


# ============================================================
# Quaternion
# ============================================================

def L_normalize_quat(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))

    if norm < 1.0e-12:
        raise ValueError("Quaternion norm is zero")

    return quaternion / norm




def L_quat_from_axis_angle(axis, degree):
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1.0e-12:
        raise ValueError("Axis norm is zero")

    axis = axis / axis_norm
    half_angle = np.deg2rad(float(degree)) * 0.5
    sin_half = np.sin(half_angle)

    return L_normalize_quat([
        np.cos(half_angle),
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
    ])


def L_quat_mul(q1, q2):
    w1, x1, y1, z1 = L_normalize_quat(q1)
    w2, x2, y2, z2 = L_normalize_quat(q2)

    return L_normalize_quat([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def L_quat_slerp(q0, q1, alpha):
    q0 = L_normalize_quat(q0)
    q1 = L_normalize_quat(q1)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    dot = float(np.dot(q0, q1))

    # q와 -q는 같은 회전이다. 더 짧은 회전 경로를 선택한다.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return L_normalize_quat(q0 + alpha * (q1 - q0))

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0

    return L_normalize_quat(s0 * q0 + s1 * q1)


def L_smoothstep01(value):
    value = float(np.clip(value, 0.0, 1.0))

    return value * value * value * (
        value * (value * 6.0 - 15.0) + 10.0
    )


def L_quat_conjugate(quaternion):
    w, x, y, z = L_normalize_quat(quaternion)
    return np.array([w, -x, -y, -z], dtype=float)


def L_quat_rotate_vector(quaternion, vector):
    quaternion = L_normalize_quat(quaternion)
    vector = np.asarray(vector, dtype=float)
    pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=float)

    def raw_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], dtype=float)

    rotated = raw_mul(
        raw_mul(quaternion, pure),
        L_quat_conjugate(quaternion),
    )
    return rotated[1:4]


def L_quat_from_rotation_matrix(matrix):
    r = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    return L_normalize_quat([w, x, y, z])


def L_quat_angle_error_deg(q0, q1):
    q0 = L_normalize_quat(q0)
    q1 = L_normalize_quat(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def L_make_vertical_side_grasp_orientation(reference_orientation):
    """
    왼팔 EE 로컬 +X(흡착면 법선)를 월드 -Y 접근 방향에 맞춘다.

    남는 roll 자유도는 EE 로컬 +Z가 월드 XY 평면과 평행하도록 정한다.
    이 로컬 +Z를 wrist_3 수평 기준축으로 사용한다. 현재 자세와 가까운
    +X/-X 수평 방향을 선택해 불필요한 180도 회전을 줄인다.
    """
    reference_orientation = L_normalize_quat(reference_orientation)
    x_world = L_LEFT_APPROACH_DIRECTION.copy()
    x_world /= np.linalg.norm(x_world)

    current_z_world = L_quat_rotate_vector(
        reference_orientation,
        L_WRIST3_HORIZONTAL_LOCAL_AXIS,
    )

    # 지면에 평행하면서 흡착 법선과 직교하는 성분만 남긴다.
    z_world = current_z_world.copy()
    z_world -= L_WORLD_UP_AXIS * float(np.dot(z_world, L_WORLD_UP_AXIS))
    z_world -= x_world * float(np.dot(z_world, x_world))

    if np.linalg.norm(z_world) < 1.0e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        fallback -= x_world * float(np.dot(fallback, x_world))
        if np.linalg.norm(fallback) < 1.0e-8:
            fallback = np.array([-1.0, 0.0, 0.0], dtype=float)
        z_world = fallback

    z_world /= np.linalg.norm(z_world)
    y_world = np.cross(z_world, x_world)
    y_world /= np.linalg.norm(y_world)
    z_world = np.cross(x_world, y_world)
    z_world /= np.linalg.norm(z_world)

    rotation_matrix = np.column_stack((x_world, y_world, z_world))
    return L_quat_from_rotation_matrix(rotation_matrix)


def L_make_vertical_side_grasp_orientation_from_tangent(
    reference_orientation,
    surface_tangent_world,
):
    """
    두 Attachment Ray의 실제 hit 지점 차이로 박스 측면 접선 방향을 구하고,
    EE 로컬 +Z(두 컵 배열 방향)를 그 접선에 맞춘다. 로컬 +X 흡착 법선은
    접선과 월드 +Z에 모두 수직이면서 박스를 향하는 방향으로 정한다.
    """
    reference_orientation = L_normalize_quat(reference_orientation)
    tangent = np.asarray(surface_tangent_world, dtype=float).copy()

    # 측면은 수직이라고 가정하므로 접선의 수평 성분만 사용한다.
    tangent -= L_WORLD_UP_AXIS * float(np.dot(tangent, L_WORLD_UP_AXIS))
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1.0e-8:
        return reference_orientation
    z_world = tangent / tangent_norm

    # 현재 wrist3 기준축과 같은 부호를 선택해 불필요한 180도 반전을 막는다.
    current_z = L_get_wrist3_reference_axis_world(reference_orientation)
    current_z -= L_WORLD_UP_AXIS * float(np.dot(current_z, L_WORLD_UP_AXIS))
    if np.linalg.norm(current_z) > 1.0e-8 and float(np.dot(z_world, current_z)) < 0.0:
        z_world = -z_world

    # z_world가 측면의 수평 접선이면 cross(z, up)가 박스 측면 법선이다.
    x_world = np.cross(z_world, L_WORLD_UP_AXIS)
    x_world /= np.linalg.norm(x_world)

    expected = L_LEFT_APPROACH_DIRECTION.copy()
    expected /= np.linalg.norm(expected)
    if float(np.dot(x_world, expected)) < 0.0:
        z_world = -z_world
        x_world = -x_world

    y_world = np.cross(z_world, x_world)
    y_world /= np.linalg.norm(y_world)
    z_world = np.cross(x_world, y_world)
    z_world /= np.linalg.norm(z_world)

    return L_quat_from_rotation_matrix(
        np.column_stack((x_world, y_world, z_world))
    )


def L_angle_between_vectors_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def L_get_suction_axis_world(orientation):
    return L_quat_rotate_vector(
        orientation,
        L_EE_SUCTION_LOCAL_AXIS,
    )


def L_get_wrist3_reference_axis_world(orientation):
    return L_quat_rotate_vector(
        orientation,
        L_WRIST3_HORIZONTAL_LOCAL_AXIS,
    )


def L_get_side_normal_error_deg(orientation):
    actual = L_get_suction_axis_world(orientation)
    actual /= np.linalg.norm(actual)
    expected = L_LEFT_APPROACH_DIRECTION.copy()
    expected /= np.linalg.norm(expected)
    dot = float(np.clip(np.dot(actual, expected), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def L_get_wrist3_horizontal_error_deg(orientation):
    axis = L_get_wrist3_reference_axis_world(orientation)
    axis /= np.linalg.norm(axis)
    vertical_component = abs(float(np.dot(axis, L_WORLD_UP_AXIS)))
    vertical_component = float(np.clip(vertical_component, 0.0, 1.0))
    return float(np.rad2deg(np.arcsin(vertical_component)))

def L_get_world_pose(path):
    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        raise RuntimeError(
            "Prim not found: " + path
        )

    matrix = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()

    rotation = transform.GetRotation()
    gf_quat = rotation.GetQuat()
    imaginary = gf_quat.GetImaginary()

    position = np.array([
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    ], dtype=float)

    orientation = L_normalize_quat([
        float(gf_quat.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    ])

    return position, orientation


def L_get_world_pos(path):
    return L_get_world_pose(path)[0]


def L_validate_required_prims():
    required_paths = [
        L_LEFT_ROBOT,
        L_LEFT_EE_PATH,
        *L_LEFT_GRIPPERS,
    ]

    print("")
    print("=== VALIDATE LEFT PRIMS ===")

    for path in required_paths:
        valid = stage.GetPrimAtPath(path).IsValid()
        print(path, "valid:", valid)

        if not valid:
            raise RuntimeError(
                "Required prim not found: " + path
            )


def L_get_ur10_rmpflow():
    config = (
        interface_config_loader
        .load_supported_motion_policy_config(
            "UR10",
            "RMPflow",
        )
    )

    return RmpFlow(**config)


def L_make_robot(robot_path, name):
    robot = Articulation(
        prim_paths_expr=robot_path,
        name=name,
    )
    robot.initialize()

    return robot


def L_get_base_pose(robot):
    positions, orientations = robot.get_world_poses()

    position = np.asarray(
        positions
    ).reshape(-1, 3)[0]

    orientation = np.asarray(
        orientations
    ).reshape(-1, 4)[0]

    return position, orientation


def L_find_joint_prim_by_name(robot_root, joint_name):
    root = stage.GetPrimAtPath(robot_root)

    if not root.IsValid():
        raise RuntimeError(
            "Robot root not found: " + robot_root
        )

    for prim in Usd.PrimRange(root):
        if prim.GetName() == joint_name:
            return prim

    return None


def L_get_current_joint_rad(robot_root, joint_name):
    prim = L_find_joint_prim_by_name(
        robot_root,
        joint_name,
    )

    if prim is None:
        print("[WARN] joint not found:", joint_name)
        return 0.0

    position_attr = prim.GetAttribute(
        "state:angular:physics:position"
    )

    if position_attr.IsValid():
        value = position_attr.Get()

        if value is not None:
            return np.deg2rad(float(value))

    drive = UsdPhysics.DriveAPI.Get(
        prim,
        "angular",
    )

    if drive:
        target_attr = drive.GetTargetPositionAttr()

        if target_attr.IsValid():
            value = target_attr.Get()

            if value is not None:
                return np.deg2rad(float(value))

    return 0.0


def L_set_drive_targets_from_rad(
    robot_root,
    joint_names,
    q_rad,
):
    for joint_name, q_value in zip(
        joint_names,
        q_rad,
    ):
        prim = L_find_joint_prim_by_name(
            robot_root,
            joint_name,
        )

        if prim is None:
            continue

        target_degree = float(
            np.rad2deg(q_value)
        )

        drive = UsdPhysics.DriveAPI.Get(
            prim,
            "angular",
        )

        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(
                prim,
                "angular",
            )

        if not drive.GetTargetPositionAttr().IsValid():
            drive.CreateTargetPositionAttr()
        drive.GetTargetPositionAttr().Set(
            target_degree
        )

        if not drive.GetStiffnessAttr().IsValid():
            drive.CreateStiffnessAttr()
        drive.GetStiffnessAttr().Set(
            float(L_STIFFNESS)
        )

        if not drive.GetDampingAttr().IsValid():
            drive.CreateDampingAttr()
        drive.GetDampingAttr().Set(
            float(L_DAMPING)
        )

        if not drive.GetMaxForceAttr().IsValid():
            drive.CreateMaxForceAttr()
        drive.GetMaxForceAttr().Set(
            float(L_MAX_FORCE)
        )


def L_setup_rmp_controller():
    robot = L_make_robot(
        L_LEFT_ROBOT,
        "left_robot_rmp",
    )

    rmpflow = L_get_ur10_rmpflow()

    base_position, base_orientation = L_get_base_pose(
        robot
    )
    rmpflow.set_robot_base_pose(
        base_position,
        base_orientation,
    )

    active_joints = list(
        rmpflow.get_active_joints()
    )

    q = np.array([
        L_get_current_joint_rad(
            L_LEFT_ROBOT,
            joint_name,
        )
        for joint_name in active_joints
    ], dtype=float)

    qd = np.zeros_like(q)

    initial_ee_position, initial_ee_orientation = (
        L_get_world_pose(L_LEFT_EE_PATH)
    )

    print("")
    print("=== SETUP LEFT CONTROLLER ===")
    print("robot root:", L_LEFT_ROBOT)
    print("EE path:", L_LEFT_EE_PATH)
    print("active joints:", active_joints)
    print("initial joint deg:", np.rad2deg(q))
    print("initial EE position:", initial_ee_position)
    print(
        "initial EE orientation wxyz:",
        initial_ee_orientation,
    )

    return {
        "robot": robot,
        "rmpflow": rmpflow,
        "active_joints": active_joints,
        "home_q": q.copy(),
        "home_ee_position": initial_ee_position.copy(),
        "home_ee_orientation": initial_ee_orientation.copy(),
        "q": q.copy(),
        "qd": qd.copy(),
        "watched_q": np.array([], dtype=float),
        "watched_qd": np.array([], dtype=float),
    }


def L_policy_target_from_world(robot, world_target):
    base_position, _ = L_get_base_pose(robot)

    return base_position + (
        np.asarray(world_target, dtype=float)
        - base_position
    ) / L_ROBOT_SCALE_FOR_POLICY


def L_step_rmp_controller(
    ctrl,
    policy_target,
    target_orientation,
):
    robot = ctrl["robot"]
    rmpflow = ctrl["rmpflow"]

    base_position, base_orientation = L_get_base_pose(
        robot
    )

    rmpflow.set_robot_base_pose(
        base_position,
        base_orientation,
    )

    rmpflow.set_end_effector_target(
        target_position=np.asarray(
            policy_target,
            dtype=float,
        ),
        target_orientation=L_normalize_quat(
            target_orientation
        ),
    )

    q_target, qd_target = (
        rmpflow.compute_joint_targets(
            ctrl["q"],
            ctrl["qd"],
            ctrl["watched_q"],
            ctrl["watched_qd"],
            L_PHYSICS_DT,
        )
    )

    ctrl["q"] = np.asarray(
        q_target,
        dtype=float,
    )
    ctrl["qd"] = np.asarray(
        qd_target,
        dtype=float,
    )

    L_set_drive_targets_from_rad(
        L_LEFT_ROBOT,
        ctrl["active_joints"],
        ctrl["q"],
    )


# ============================================================
# EE 이동
# ============================================================

def L_move_left_ee(
    ctrl,
    start_position,
    target_position,
    start_orientation,
    target_orientation,
    frames,
    label,
    convergence_frames=None,
    stable_frames=None,
):
    start_position = np.asarray(
        start_position,
        dtype=float,
    )
    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    start_orientation = L_normalize_quat(
        start_orientation
    )
    target_orientation = L_normalize_quat(
        target_orientation
    )

    print("")
    print(f"=== {label} START ===")
    print("start position:", start_position)
    print("target position:", target_position)
    print(
        "start orientation wxyz:",
        start_orientation,
    )
    print(
        "target orientation wxyz:",
        target_orientation,
    )
    print("frames:", frames)

    convergence_frames = int(
        L_MOVE_CONVERGENCE_FRAMES
        if convergence_frames is None
        else convergence_frames
    )
    stable_frames = int(
        L_MOVE_STABLE_FRAMES
        if stable_frames is None
        else stable_frames
    )
    convergence_frames = max(1, convergence_frames)
    stable_frames = max(1, stable_frames)

    policy_start = L_policy_target_from_world(
        ctrl["robot"],
        start_position,
    )
    policy_end = L_policy_target_from_world(
        ctrl["robot"],
        target_position,
    )

    for frame in range(int(frames) + 1):
        ratio = frame / float(
            max(int(frames), 1)
        )
        alpha = L_smoothstep01(ratio)

        policy_now = policy_start + (
            policy_end - policy_start
        ) * alpha

        orientation_now = L_quat_slerp(
            start_orientation,
            target_orientation,
            alpha,
        )

        L_step_rmp_controller(
            ctrl,
            policy_now,
            orientation_now,
        )

        if DEBUG_MOTION and (frame % 15 == 0 or frame == int(frames)):
            ee_now = L_get_world_pos(L_LEFT_EE_PATH)
            _dlog(DEBUG_MOTION, f"[{label} FRAME {frame:03d}/{frames}]")
            _dlog(DEBUG_MOTION, "alpha:", alpha)
            _dlog(DEBUG_MOTION, "EE now:", ee_now)

        L_update_simulation()

    # 보간 종료 후 실제 EE가 목표에 수렴할 때까지 최종 목표를 유지한다.
    stable_count = 0
    reached = False

    for settle_frame in range(convergence_frames):
        L_step_rmp_controller(
            ctrl,
            policy_end,
            target_orientation,
        )

        L_update_simulation()

        ee_now = L_get_world_pos(
            L_LEFT_EE_PATH
        )
        error = float(
            np.linalg.norm(
                ee_now - target_position
            )
        )

        if error <= L_MOVE_POSITION_TOLERANCE:
            stable_count += 1
        else:
            stable_count = 0

        if settle_frame % 30 == 0:
            print(
                f"[{label} CONVERGENCE "
                f"{settle_frame:03d}/"
                f"{convergence_frames}]",
                "EE=",
                ee_now,
                "error=",
                round(error, 6),
                "stable=",
                stable_count,
            )

        if stable_count >= stable_frames:
            reached = True
            break

    final_ee = L_get_world_pos(
        L_LEFT_EE_PATH
    )
    final_error = float(
        np.linalg.norm(
            final_ee - target_position
        )
    )

    print("final EE:", final_ee)
    print("final target error:", final_error)
    print("target reached:", reached)
    print(f"=== {label} DONE ===")

    return reached, final_ee


# ============================================================
# 왼팔 TCP / Attachment / Raycast 측면 파지
# ============================================================

def L_get_attachment_joint_paths():
    global _L_ATTACHMENT_JOINT_PATHS_CACHE

    if _L_ATTACHMENT_JOINT_PATHS_CACHE is not None:
        return list(_L_ATTACHMENT_JOINT_PATHS_CACHE)

    paths = []
    for gripper_path in L_LEFT_GRIPPERS:
        prim = stage.GetPrimAtPath(gripper_path)
        if not prim.IsValid():
            raise RuntimeError("Left gripper prim not found: " + gripper_path)

        found_relation = False
        for relation in prim.GetRelationships():
            normalized = _normalized_attr_name(relation.GetName())
            if "attachmentpoints" not in normalized:
                continue

            found_relation = True
            for target in relation.GetTargets():
                path = str(target)
                if path not in paths:
                    paths.append(path)

        if not found_relation:
            print("[WARN] LEFT Attachment Points relationship not found:", gripper_path)

    if not paths:
        raise RuntimeError(
            "No LEFT SurfaceGripper attachment D6 joints were found. "
            "Set each SurfaceGripper Prim's Attachment Points relationship."
        )

    for path in paths:
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError("Left attachment joint prim not found: " + path)

    _L_ATTACHMENT_JOINT_PATHS_CACHE = tuple(paths)
    return list(paths)


def L_get_gripper_tcp_points_world():
    return [L_get_world_pos(path) for path in L_get_attachment_joint_paths()]


def L_get_gripper_tcp_world():
    points = L_get_gripper_tcp_points_world()
    return np.mean(np.stack(points, axis=0), axis=0)


def L_print_attachment_diagnostics():
    print("")
    print("=== LEFT SURFACE GRIPPER ATTACHMENT DIAGNOSTICS ===")
    for path in L_get_attachment_joint_paths():
        position, orientation = L_get_world_pose(path)
        print("attachment joint:", path)
        print("  world position:", position)
        print("  world orientation wxyz:", orientation)


def L_configure_surface_grippers():
    print("")
    print("=== CONFIGURE LEFT SURFACE GRIPPERS ===")
    for gripper_path in L_LEFT_GRIPPERS:
        prim = stage.GetPrimAtPath(gripper_path)
        if not prim.IsValid():
            raise RuntimeError("Left gripper prim not found: " + gripper_path)

        max_distance_set = False
        retry_interval_set = False
        for attr in prim.GetAttributes():
            normalized = _normalized_attr_name(attr.GetName())
            try:
                if "maxgripdistance" in normalized:
                    attr.Set(float(L_SURFACE_MAX_GRIP_DISTANCE))
                    max_distance_set = True
                    print(gripper_path, attr.GetName(), "=", L_SURFACE_MAX_GRIP_DISTANCE)
                elif "retryinterval" in normalized:
                    attr.Set(float(L_SURFACE_RETRY_INTERVAL))
                    retry_interval_set = True
                    print(gripper_path, attr.GetName(), "=", L_SURFACE_RETRY_INTERVAL)
            except Exception as exc:
                print("[WARN] left gripper property set failed:", gripper_path, attr.GetName(), repr(exc))

        if not max_distance_set:
            print("[WARN] LEFT Max Grip Distance attribute not found:", gripper_path)
        if not retry_interval_set:
            print("[WARN] LEFT Retry Interval attribute not found:", gripper_path)


def L_get_tcp_offset_in_ee_frame():
    ee_position, ee_orientation = L_get_world_pose(L_LEFT_EE_PATH)
    tcp_world = L_get_gripper_tcp_world()
    offset_world = tcp_world - ee_position
    offset_local = L_quat_rotate_vector(
        L_quat_conjugate(ee_orientation),
        offset_world,
    )

    print("")
    print("=== LEFT TCP CALIBRATION ===")
    print("EE world position:", ee_position)
    print("TCP world midpoint:", tcp_world)
    print("EE -> TCP world offset:", offset_world)
    print("EE -> TCP local offset:", offset_local)
    return offset_local


def L_ee_target_from_tcp_target(tcp_world_target, ee_orientation, tcp_offset_local):
    tcp_offset_world = L_quat_rotate_vector(
        ee_orientation,
        tcp_offset_local,
    )
    return np.asarray(tcp_world_target, dtype=float) - tcp_offset_world


def L_get_gripped_objects_map(gripper_interface):
    result = {}
    for gripper_path in L_LEFT_GRIPPERS:
        try:
            result[gripper_path] = list(
                gripper_interface.get_gripped_objects(gripper_path)
            )
        except Exception as exc:
            print("[WARN] LEFT get_gripped_objects failed:", gripper_path, repr(exc))
            result[gripper_path] = []
    return result


def L_has_any_gripper_attached(gripped_map):
    """upper/lower 중 하나 이상의 흡착컵이 물체를 잡았는지 반환한다."""
    return any(
        len(objects) > 0 for objects in gripped_map.values()
    )


def L_have_all_grippers_attached(gripped_map):
    """upper/lower 두 흡착컵이 모두 물체를 잡았을 때만 True를 반환한다."""
    return bool(gripped_map) and all(
        len(objects) > 0 for objects in gripped_map.values()
    )


def L_command_left_grippers_close():
    gripper_interface = surface_gripper.acquire_surface_gripper_interface()
    for gripper_path in L_LEFT_GRIPPERS:
        try:
            gripper_interface.close_gripper(gripper_path)
        except Exception as exc:
            print("[WARN] LEFT close_gripper failed:", gripper_path, repr(exc))
    return gripper_interface


def L_raycast_non_robot_hits(origin, direction, distance):
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    hits = []

    def report(hit):
        body = str(_hit_attr(hit, "rigid_body", "rigidBody", default=""))
        collision = str(_hit_attr(hit, "collision", default=""))
        if body.startswith(L_LEFT_ROBOT) or collision.startswith(L_LEFT_ROBOT):
            return True

        hits.append({
            "rigidBody": body,
            "collision": collision,
            "distance": float(_hit_attr(hit, "distance", default=np.inf)),
            "position": _vec3_to_numpy(_hit_attr(hit, "position", default=None)),
            "normal": _vec3_to_numpy(_hit_attr(hit, "normal", default=None)),
        })
        return True

    o = carb.Float3(float(origin[0]), float(origin[1]), float(origin[2]))
    d = carb.Float3(float(direction[0]), float(direction[1]), float(direction[2]))

    try:
        physx_query.raycast_all(o, d, float(distance), report)
    except Exception as exc:
        print("[WARN] LEFT raycast_all failed; falling back to raycast_closest:", repr(exc))
        closest = physx_query.raycast_closest(o, d, float(distance))
        if closest.get("hit", False):
            body = str(closest.get("rigidBody", ""))
            collision = str(closest.get("collision", ""))
            if not body.startswith(L_LEFT_ROBOT) and not collision.startswith(L_LEFT_ROBOT):
                hits.append({
                    "rigidBody": body,
                    "collision": collision,
                    "distance": float(closest.get("distance", np.inf)),
                    "position": _vec3_to_numpy(closest.get("position")),
                    "normal": _vec3_to_numpy(closest.get("normal")),
                })

    hits.sort(key=lambda item: item["distance"])
    return hits


def L_detect_side_surface_with_rays(surface_target, ray_direction=None):
    """
    왼팔 측면 Collider를 검사한다.

    기존에는 두 Attachment Joint 위치에서만 Ray를 쐈다. 두 흡착컵의 X 간격이
    박스 폭과 비슷하면 두 Ray가 박스 양쪽 가장자리를 스쳐 모두 miss할 수 있다.
    따라서 실제 제어 기준은 두 Joint의 중점인 TCP center Ray로 사용하고,
    각 Joint Ray와 중간 보조 Ray는 진단 및 보조 검출에 사용한다.
    """
    surface_target = np.asarray(surface_target, dtype=float)
    if ray_direction is None:
        direction = L_LEFT_APPROACH_DIRECTION.copy()
    else:
        direction = np.asarray(ray_direction, dtype=float).copy()
    direction /= np.linalg.norm(direction)

    joint_paths = L_get_attachment_joint_paths()
    joint_points = L_get_gripper_tcp_points_world()
    center_point = np.mean(np.stack(joint_points, axis=0), axis=0)

    ray_specs = [("TCP_CENTER", center_point)]

    for joint_path, point in zip(joint_paths, joint_points):
        ray_specs.append((joint_path, point))

        # Joint와 중심 사이의 보조 Ray. 박스 가장자리 수치 오차를 피한다.
        inner_point = center_point + 0.65 * (point - center_point)
        ray_specs.append((joint_path + "::INNER", inner_point))

    detections = []

    for ray_name, point in ray_specs:
        origin = point + direction * L_RAY_ORIGIN_OFFSET
        hits = L_raycast_non_robot_hits(origin, direction, L_RAYCAST_RANGE)

        accepted = None
        for hit in hits:
            plane_error = abs(float(np.dot(
                hit["position"] - surface_target,
                direction,
            )))
            if plane_error <= L_RAY_SURFACE_PLANE_TOLERANCE:
                accepted = hit
                break

        detections.append({
            "joint": ray_name,
            "origin": origin,
            "hit": accepted,
            "all_hits": hits,
            "is_center": ray_name == "TCP_CENTER",
        })

    # 중심 Ray를 최우선으로 사용한다. 중심 Ray가 없을 때만 보조 Ray를 사용한다.
    center_hits = [
        item["hit"]
        for item in detections
        if item.get("is_center") and item["hit"] is not None
    ]
    if center_hits:
        return center_hits[0], detections

    valid_hits = [
        item["hit"]
        for item in detections
        if item["hit"] is not None
    ]
    nearest = min(valid_hits, key=lambda item: item["distance"]) if valid_hits else None
    return nearest, detections


def L_approach_until_side_surface_gripped(
    ctrl,
    surface_target,
    target_orientation,
    tcp_offset_local,
):
    """
    왼팔 TCP를 박스 측면으로 접근시키고 두 SurfaceGripper를 동시에 붙인다.

    핵심 규칙:
      1. 접근 중에는 gripper를 열어 둔다.
      2. upper/lower 실제 Attachment Joint Ray가 모두 접촉 가능 거리 안에
         들어온 뒤에만 두 gripper를 동시에 close한다.
      3. 한쪽만 붙으면 즉시 두 gripper를 다시 열어 박스가 한쪽 컵에
         끌려가는 것을 막고 동시 close를 재시도한다.
      4. command travel이 최대값에 도달해도 현재 EE를 hold하지 않고,
         계산된 Hard Stop EE 목표를 계속 명령해 실제 TCP가 수렴하게 한다.
      5. 두 컵이 연속 L_DUAL_GRIP_STABLE_FRAMES 동안 붙어야 성공이다.
    """
    surface_target = np.asarray(surface_target, dtype=float)
    # 자동 자세 교정이 반영된 실제 흡착 법선 방향으로 접근한다.
    direction = L_get_suction_axis_world(target_orientation)
    direction /= np.linalg.norm(direction)

    # 접근 시작 시 이전 흡착 상태를 반드시 해제한다.
    L_command_left_grippers_open(settle_frames=3)
    gripper_interface = surface_gripper.acquire_surface_gripper_interface()

    start_tcp = L_get_gripper_tcp_world()
    start_ee, _ = L_get_world_pose(L_LEFT_EE_PATH)
    hard_stop_tcp = surface_target - direction * L_HARD_SURFACE_GAP
    max_command_travel = max(
        0.0,
        float(np.dot(hard_stop_tcp - start_tcp, direction)),
    )

    # 실제 Hard Stop TCP에 대응하는 최종 EE 목표를 미리 계산한다.
    hard_stop_ee_target = L_ee_target_from_tcp_target(
        hard_stop_tcp,
        target_orientation,
        tcp_offset_local,
    )
    hard_stop_policy_target = L_policy_target_from_world(
        ctrl["robot"],
        hard_stop_ee_target,
    )

    commanded_travel = 0.0
    max_frames = max(
        1,
        int(np.ceil(max_command_travel / min(
            L_CONTACT_APPROACH_STEP,
            L_DUAL_GRIP_PRESS_STEP,
        )))
        + L_MOVE_CONVERGENCE_FRAMES
        + L_RAY_CONTACT_SETTLE_FRAMES
        + 120,
    )

    ray_miss_count = 0
    sync_close_retry_count = 0
    all_grip_stable_count = 0
    attachment_joint_paths = set(L_get_attachment_joint_paths())
    # SurfaceGripper max distance는 Attachment 원점 기준이고 Ray는 이미
    # L_RAY_ORIGIN_OFFSET만큼 앞에서 시작하므로 그 차이를 반드시 뺀다.
    effective_sync_close_distance = min(
        L_DUAL_GRIP_SYNC_CLOSE_DISTANCE,
        max(
            0.0,
            L_SURFACE_MAX_GRIP_DISTANCE
            - L_RAY_ORIGIN_OFFSET
            - 0.001,
        ),
    )

    print("")
    print("=== LEFT SYNCHRONIZED DUAL-GRIP APPROACH START ===")
    print("start EE:", start_ee)
    print("start attachment TCP:", start_tcp)
    print("vision side surface:", surface_target)
    print("approach direction:", direction)
    print("hard-stop TCP plane point:", hard_stop_tcp)
    print("hard-stop EE target:", hard_stop_ee_target)
    print("max command travel:", max_command_travel)
    print("configured sync close distance:", L_DUAL_GRIP_SYNC_CLOSE_DISTANCE)
    print("effective sync close distance:", effective_sync_close_distance)
    print("required stable frames:", L_DUAL_GRIP_STABLE_FRAMES)

    def read_actual_joint_hits(ray_details):
        """center/inner ray를 제외하고 실제 upper/lower joint ray만 반환한다."""
        result = {}
        for item in ray_details:
            joint_name = item.get("joint")
            if joint_name in attachment_joint_paths:
                result[joint_name] = item.get("hit")
        return result

    def command_tcp_target(tcp_target):
        ee_target = L_ee_target_from_tcp_target(
            tcp_target,
            target_orientation,
            tcp_offset_local,
        )
        policy_target = L_policy_target_from_world(
            ctrl["robot"],
            ee_target,
        )
        L_step_rmp_controller(ctrl, policy_target, target_orientation)

    def attempt_synchronized_close(hold_policy, reason):
        """두 컵을 동시에 닫고 all-grip 안정 상태를 확인한다."""
        nonlocal sync_close_retry_count, all_grip_stable_count

        sync_close_retry_count += 1
        all_grip_stable_count = 0

        print("")
        print("=== LEFT SYNCHRONIZED CLOSE ATTEMPT ===")
        print("reason:", reason)
        print(
            "attempt:",
            sync_close_retry_count,
            "/",
            L_DUAL_GRIP_SYNC_CLOSE_MAX_RETRIES,
        )

        # 같은 프레임 구간에서 upper/lower 모두 close 명령을 반복한다.
        for settle_frame in range(L_DUAL_GRIP_SYNC_CLOSE_SETTLE_FRAMES):
            L_step_rmp_controller(ctrl, hold_policy, target_orientation)

            if settle_frame % L_GRIP_RETRY_EVERY_FRAMES == 0:
                gripper_now = L_command_left_grippers_close()
            else:
                gripper_now = gripper_interface

            L_update_simulation()
            gripped_map = L_get_gripped_objects_map(gripper_now)
            all_attached = L_have_all_grippers_attached(gripped_map)
            any_attached = L_has_any_gripper_attached(gripped_map)

            if all_attached:
                all_grip_stable_count += 1
            else:
                all_grip_stable_count = 0

            if (
                settle_frame == 0
                or settle_frame % 5 == 0
                or any_attached
            ):
                print(
                    f"[LEFT SYNC CLOSE {settle_frame:03d}/"
                    f"{L_DUAL_GRIP_SYNC_CLOSE_SETTLE_FRAMES}]",
                    "any=", any_attached,
                    "all=", all_attached,
                    "stable=", all_grip_stable_count,
                    "objects=", gripped_map,
                )

            if all_grip_stable_count >= L_DUAL_GRIP_STABLE_FRAMES:
                final_ee, _ = L_get_world_pose(L_LEFT_EE_PATH)
                final_tcp = L_get_gripper_tcp_world()
                print("[SUCCESS] BOTH LEFT grippers attached simultaneously")
                print("final EE:", final_ee)
                print("final TCP:", final_tcp)
                print("gripped:", gripped_map)
                return True, final_ee, final_tcp, gripped_map

        # 한쪽만 붙은 채 끝났다면 그대로 밀지 않는다.
        # 첫 번째 D6 attachment가 박스를 같이 끌고 가기 때문이다.
        final_map = L_get_gripped_objects_map(gripper_interface)
        if L_has_any_gripper_attached(final_map) and not L_have_all_grippers_attached(final_map):
            print("[LEFT SINGLE ATTACH RESET] Only one cup attached.")
            print("Opening both cups before the next synchronized retry.")
            print("objects:", final_map)
            L_command_left_grippers_open(settle_frames=2)

        return False, None, None, final_map

    for frame in range(max_frames + 1):
        nearest_hit, ray_details = L_detect_side_surface_with_rays(surface_target, direction)
        current_ee, current_orientation = L_get_world_pose(L_LEFT_EE_PATH)
        current_tcp = L_get_gripper_tcp_world()

        remaining_to_hard_stop = float(np.dot(
            hard_stop_tcp - current_tcp,
            direction,
        ))
        actual_joint_hits = read_actual_joint_hits(ray_details)
        valid_joint_hits = {
            path: hit
            for path, hit in actual_joint_hits.items()
            if hit is not None
        }
        both_joint_rays_valid = (
            len(valid_joint_hits) == len(attachment_joint_paths)
            and len(attachment_joint_paths) > 0
        )

        joint_distances = {
            path: float(hit["distance"])
            for path, hit in valid_joint_hits.items()
        }
        farthest_joint_distance = (
            max(joint_distances.values())
            if joint_distances
            else float("inf")
        )

        # 두 실제 컵의 ray가 모두 grip 가능 거리 안에 있을 때만 close한다.
        sync_close_ready = (
            both_joint_rays_valid
            and farthest_joint_distance
            <= effective_sync_close_distance
        )

        if sync_close_ready:
            hold_policy = L_policy_target_from_world(ctrl["robot"], current_ee)
            success, final_ee, final_tcp, final_map = attempt_synchronized_close(
                hold_policy,
                reason=(
                    "both attachment rays are within close distance; "
                    f"distances={joint_distances}"
                ),
            )
            if success:
                return True, final_ee, final_tcp, final_map

            if sync_close_retry_count >= L_DUAL_GRIP_SYNC_CLOSE_MAX_RETRIES:
                print("[FAIL] Synchronized close retry limit reached.")
                print("joint ray distances:", joint_distances)
                return False, current_ee, current_tcp, final_map

        # 실제 TCP가 Hard Stop 목표에 도달하면 더 이상 전진값은 늘리지 않는다.
        # 단, 최종 EE 목표는 계속 명령하여 RMPflow 추종 오차를 줄인다.
        if remaining_to_hard_stop <= L_HARD_STOP_TRACKING_TOLERANCE:
            L_step_rmp_controller(
                ctrl,
                hard_stop_policy_target,
                target_orientation,
            )
            L_update_simulation()

            if frame % 5 == 0:
                print(
                    f"[LEFT HARD STOP TRACK {frame:03d}/{max_frames}]",
                    "remaining=", round(remaining_to_hard_stop, 6),
                    "joint hits=", joint_distances,
                    "both rays=", both_joint_rays_valid,
                )

            # Hard Stop인데 두 joint ray가 모두 보이지 않으면 전진으로 해결할 수 없다.
            if not both_joint_rays_valid:
                print("[FAIL] Hard Stop reached, but both attachment rays do not hit the box.")
                print("This is an X/Z alignment or cup-spacing problem, not a Y approach-distance problem.")
                print("actual attachment ray hits:", actual_joint_hits)
                print("surface target:", surface_target)
                print("current TCP:", current_tcp)
                return False, current_ee, current_tcp, L_get_gripped_objects_map(gripper_interface)

            # 둘 다 보이지만 아직 close 거리 밖이면 비전 surface와 실제 collider 면이 다르다.
            if farthest_joint_distance > effective_sync_close_distance:
                print("[FAIL] Hard Stop reached before both cups entered close distance.")
                print("joint distances:", joint_distances)
                print("Increase/decrease L_HARD_SURFACE_GAP only after checking the actual collider plane.")
                return False, current_ee, current_tcp, L_get_gripped_objects_map(gripper_interface)

            continue

        if nearest_hit is None:
            ray_miss_count += 1
            if frame == 0 or frame % 5 == 0:
                print(f"[LEFT RAY MISS {ray_miss_count}/{L_RAY_MISS_LIMIT_FRAMES}]")
                for item in ray_details:
                    print(" joint:", item["joint"])
                    print(" origin:", item["origin"])
                    print(" all non-left-robot hits:", item["all_hits"][:3])

            if ray_miss_count >= L_RAY_MISS_LIMIT_FRAMES:
                stop_policy = L_policy_target_from_world(ctrl["robot"], current_ee)
                for _ in range(L_CONTACT_HOLD_FRAMES):
                    L_step_rmp_controller(ctrl, stop_policy, current_orientation)
                    L_update_simulation()
                print("[FAIL] LEFT raycast cannot see the requested box side surface.")
                return False, current_ee, current_tcp, L_get_gripped_objects_map(gripper_interface)
        else:
            ray_miss_count = 0

        center_ray_distance = (
            float(nearest_hit["distance"])
            if nearest_hit is not None
            else float("inf")
        )

        # 아직 두 cup ray가 close 거리에 없으면 open 상태로 계속 접근한다.
        if both_joint_rays_valid:
            remaining_by_ray = max(
                0.0,
                farthest_joint_distance - effective_sync_close_distance,
            )
        elif nearest_hit is not None:
            remaining_by_ray = max(
                0.0,
                center_ray_distance - effective_sync_close_distance,
            )
        else:
            remaining_by_ray = L_CONTACT_APPROACH_STEP

        safe_increment = min(
            L_CONTACT_APPROACH_STEP,
            remaining_by_ray if remaining_by_ray > 0.0 else L_DUAL_GRIP_PRESS_STEP,
            max(0.0, remaining_to_hard_stop),
            max(0.0, max_command_travel - commanded_travel),
        )

        if safe_increment > 1.0e-6:
            commanded_travel += safe_increment

        # 중요: increment가 0이더라도 현재 EE를 hold하지 않는다.
        # 마지막으로 계산된 TCP command를 계속 명령해 실제 TCP가 따라오게 한다.
        tcp_target = start_tcp + direction * commanded_travel
        command_tcp_target(tcp_target)

        if DEBUG_GRIPPER and frame % 10 == 0:
            _dlog(DEBUG_GRIPPER,
                f"[LEFT OPEN APPROACH {frame:03d}/{max_frames}]",
                "center_ray=", round(center_ray_distance, 6),
                "joint_distances=", joint_distances,
                "both_joint_rays=", both_joint_rays_valid,
                "TCP=", current_tcp,
            )

        L_update_simulation()

    stop_ee, stop_orientation = L_get_world_pose(L_LEFT_EE_PATH)
    stop_tcp = L_get_gripper_tcp_world()
    stop_policy = L_policy_target_from_world(ctrl["robot"], stop_ee)
    for _ in range(L_CONTACT_HOLD_FRAMES):
        L_step_rmp_controller(ctrl, stop_policy, stop_orientation)
        L_update_simulation()

    final_map = L_get_gripped_objects_map(gripper_interface)
    print("[FAIL] LEFT synchronized dual-grip approach timed out.")
    print("final gripped objects:", final_map)
    return False, stop_ee, stop_tcp, final_map

def L_auto_align_left_tool_to_side_plane(
    ctrl,
    surface_target,
    current_target_orientation,
    tcp_offset_local,
):
    """
    upper/lower 실제 Attachment Ray의 hit 두 점으로 측면 접선을 계산한다.
    두 Ray 거리 차이가 크면 TCP 중심 위치를 고정한 채 흡착면을 실제
    박스 측면과 평행하게 교정한다.
    """
    current_target_orientation = L_normalize_quat(current_target_orientation)
    current_direction = L_get_suction_axis_world(current_target_orientation)
    current_direction /= np.linalg.norm(current_direction)

    _, details = L_detect_side_surface_with_rays(
        surface_target,
        current_direction,
    )
    joint_order = L_get_attachment_joint_paths()
    hit_by_joint = {
        item.get("joint"): item.get("hit")
        for item in details
        if item.get("joint") in joint_order
    }

    if any(hit_by_joint.get(path) is None for path in joint_order) or len(joint_order) < 2:
        print("[LEFT SIDE ALIGN] Both attachment rays are not available; fixed -Y orientation retained.")
        return current_target_orientation, False

    hit0 = hit_by_joint[joint_order[0]]
    hit1 = hit_by_joint[joint_order[1]]
    d0 = float(hit0["distance"])
    d1 = float(hit1["distance"])
    spread = abs(d0 - d1)

    print("")
    print("=== LEFT SIDE PLANE ALIGNMENT CHECK ===")
    print("joint 0:", joint_order[0], "distance:", d0, "hit:", hit0["position"])
    print("joint 1:", joint_order[1], "distance:", d1, "hit:", hit1["position"])
    print("distance spread:", spread)
    print("spread tolerance:", L_SIDE_PLANE_DISTANCE_SPREAD_TOLERANCE)

    if spread <= L_SIDE_PLANE_DISTANCE_SPREAD_TOLERANCE:
        print("[LEFT SIDE ALIGN] Tool face is already parallel enough.")
        return current_target_orientation, False

    tangent = np.asarray(hit1["position"] - hit0["position"], dtype=float)
    aligned_orientation = L_make_vertical_side_grasp_orientation_from_tangent(
        current_target_orientation,
        tangent,
    )
    aligned_normal = L_get_suction_axis_world(aligned_orientation)
    correction_deg = L_angle_between_vectors_deg(current_direction, aligned_normal)

    print("estimated surface tangent:", tangent)
    print("current suction normal:", current_direction)
    print("aligned suction normal:", aligned_normal)
    print("auto-align correction deg:", correction_deg)

    if correction_deg > L_SIDE_PLANE_AUTO_ALIGN_MAX_DEG:
        print("[FAIL] Estimated side-plane correction is too large; alignment cancelled.")
        return current_target_orientation, False

    fixed_tcp = L_get_gripper_tcp_world().copy()
    start_orientation = L_get_world_pose(L_LEFT_EE_PATH)[1]

    for frame in range(L_SIDE_PLANE_AUTO_ALIGN_FRAMES + 1):
        alpha = L_smoothstep01(
            frame / float(max(L_SIDE_PLANE_AUTO_ALIGN_FRAMES, 1))
        )
        orientation_now = L_quat_slerp(
            start_orientation,
            aligned_orientation,
            alpha,
        )
        ee_target = L_ee_target_from_tcp_target(
            fixed_tcp,
            orientation_now,
            tcp_offset_local,
        )
        L_step_rmp_controller(
            ctrl,
            L_policy_target_from_world(ctrl["robot"], ee_target),
            orientation_now,
        )
        L_update_simulation()

        if frame % 10 == 0 or frame == L_SIDE_PLANE_AUTO_ALIGN_FRAMES:
            tcp_now = L_get_gripper_tcp_world()
            print(
                f"[LEFT SIDE ALIGN {frame:03d}/{L_SIDE_PLANE_AUTO_ALIGN_FRAMES}]",
                "TCP error=", round(float(np.linalg.norm(tcp_now - fixed_tcp)), 6),
            )

    print("=== LEFT SIDE PLANE ALIGNMENT DONE ===")
    return aligned_orientation, True


def L_approach_left_box_surface(
    ctrl,
    surface_target,
    target_orientation,
):
    surface_target = np.asarray(surface_target, dtype=float)
    direction = L_LEFT_APPROACH_DIRECTION.copy()
    direction /= np.linalg.norm(direction)
    tcp_offset_local = L_get_tcp_offset_in_ee_frame()

    pregrasp_tcp = surface_target - direction * L_PREGRASP_DISTANCE
    safe_tcp = pregrasp_tcp - direction * L_SAFE_OUTSIDE_DISTANCE

    pregrasp_ee = L_ee_target_from_tcp_target(
        pregrasp_tcp,
        target_orientation,
        tcp_offset_local,
    )
    safe_ee = L_ee_target_from_tcp_target(
        safe_tcp,
        target_orientation,
        tcp_offset_local,
    )

    current_ee, current_orientation = L_get_world_pose(L_LEFT_EE_PATH)

    print("")
    print("=== LEFT SIDE SURFACE APPROACH PLAN ===")
    print("surface target:", surface_target)
    print("approach direction:", direction)
    print("safe TCP target:", safe_tcp)
    print("safe EE target:", safe_ee)
    print("pregrasp TCP target:", pregrasp_tcp)
    print("pregrasp EE target:", pregrasp_ee)

    left_safe_frames = adaptive_grasp_waypoint_frames(
        current_ee, safe_ee, L_SAFE_MOVE_FRAMES
    )
    print("adaptive LEFT safe move frames:", left_safe_frames)

    safe_reached, _ = L_move_left_ee(
        ctrl=ctrl,
        start_position=current_ee,
        target_position=safe_ee,
        start_orientation=current_orientation,
        target_orientation=target_orientation,
        frames=left_safe_frames,
        label="LEFT MOVE OUTSIDE SIDE SURFACE",
        convergence_frames=GRASP_MOVE_CONVERGENCE_FRAMES,
        stable_frames=GRASP_MOVE_STABLE_FRAMES,
    )
    if not safe_reached:
        print("[WARN] LEFT safe side waypoint did not fully converge.")

    descend_start, descend_orientation = L_get_world_pose(L_LEFT_EE_PATH)
    left_pregrasp_frames = adaptive_grasp_waypoint_frames(
        descend_start, pregrasp_ee, L_DESCEND_FRAMES
    )
    print("adaptive LEFT pregrasp frames:", left_pregrasp_frames)

    pregrasp_reached, _ = L_move_left_ee(
        ctrl=ctrl,
        start_position=descend_start,
        target_position=pregrasp_ee,
        start_orientation=descend_orientation,
        target_orientation=target_orientation,
        frames=left_pregrasp_frames,
        label="LEFT APPROACH TO SIDE PREGRASP",
        convergence_frames=GRASP_MOVE_CONVERGENCE_FRAMES,
        stable_frames=GRASP_MOVE_STABLE_FRAMES,
    )

    actual_tcp = L_get_gripper_tcp_world()
    _, actual_orientation = L_get_world_pose(L_LEFT_EE_PATH)
    delta = actual_tcp - surface_target
    lateral_delta = delta - direction * float(np.dot(delta, direction))
    lateral_error = float(np.linalg.norm(lateral_delta))
    normal_error = L_get_side_normal_error_deg(actual_orientation)
    wrist3_horizontal_error = L_get_wrist3_horizontal_error_deg(actual_orientation)

    print("actual LEFT pregrasp TCP:", actual_tcp)
    print("side-surface lateral X/Z error:", lateral_error)
    print("suction normal world:", L_get_suction_axis_world(actual_orientation))
    print("side normal error deg:", normal_error)
    print("wrist3 reference axis world:", L_get_wrist3_reference_axis_world(actual_orientation))
    print("wrist3 horizontal error deg:", wrist3_horizontal_error)
    print("pregrasp reached:", pregrasp_reached)

    if not pregrasp_reached:
        print("[FAIL] LEFT pregrasp target did not converge; Raycast contact approach will not start.")
        return False, L_get_world_pos(L_LEFT_EE_PATH), actual_tcp

    if normal_error > L_SIDE_NORMAL_ABORT_TOLERANCE_DEG:
        print("[FAIL] LEFT suction normal is not aligned with the side approach direction.")
        return False, L_get_world_pos(L_LEFT_EE_PATH), actual_tcp

    if wrist3_horizontal_error > L_WRIST3_HORIZONTAL_ABORT_TOLERANCE_DEG:
        print("[FAIL] LEFT wrist3 reference axis is not parallel to the ground.")
        return False, L_get_world_pos(L_LEFT_EE_PATH), actual_tcp

    if lateral_error > L_SIDE_LATERAL_TOLERANCE:
        print("[FAIL] LEFT TCP is not centered on the requested side surface.")
        return False, L_get_world_pos(L_LEFT_EE_PATH), actual_tcp

    # 두 컵의 실제 Ray 거리 차이로 박스 측면 yaw를 추정하고, 필요하면
    # TCP 중심을 고정한 채 흡착면을 실제 측면과 평행하게 교정한다.
    target_orientation, side_aligned = L_auto_align_left_tool_to_side_plane(
        ctrl=ctrl,
        surface_target=surface_target,
        current_target_orientation=target_orientation,
        tcp_offset_local=tcp_offset_local,
    )
    print("side-plane auto aligned:", side_aligned)

    grip_ok, final_ee, final_tcp, gripped_map = (
        L_approach_until_side_surface_gripped(
            ctrl=ctrl,
            surface_target=surface_target,
            target_orientation=target_orientation,
            tcp_offset_local=tcp_offset_local,
        )
    )

    print("")
    print("=== LEFT SIDE SURFACE APPROACH RESULT ===")
    print("grip ok:", grip_ok)
    print("final EE:", final_ee)
    print("final TCP:", final_tcp)
    print("surface target:", surface_target)
    print("gripped objects:", gripped_map)
    return grip_ok, final_ee, final_tcp


# ============================================================
# Surface Gripper
# ============================================================

def L_command_left_grippers_open(
    settle_frames=3,
):
    gripper_interface = (
        surface_gripper
        .acquire_surface_gripper_interface()
    )

    for gripper_path in L_LEFT_GRIPPERS:
        try:
            gripper_interface.open_gripper(
                gripper_path
            )
        except Exception as exc:
            print(
                "[WARN] open_gripper failed:",
                gripper_path,
                repr(exc),
            )

    L_wait_frames(settle_frames)




def L_open_left_grippers(settle_frames=L_RELEASE_SETTLE_FRAMES):
    """왼팔 그리퍼를 열고 지정한 프레임만큼 해제 안정화를 기다린다.

    settle_frames=0이면 open 명령 직후 반환하여 완료 신호를 같은 공정 단계에서
    즉시 발행할 수 있다.
    """
    print("")
    print("=== LEFT GRIPPERS OPEN ===")

    settle_frames = max(0, int(settle_frames))
    L_command_left_grippers_open(
        settle_frames=settle_frames
    )

    print(
        "=== LEFT GRIPPERS OPEN COMMAND SENT ==="
        if settle_frames == 0
        else "=== LEFT GRIPPERS OPEN DONE ==="
    )


# ============================================================
# 상승 / 회전 / 배치 / 이탈
# ============================================================

def L_lift_left_robot(
    ctrl,
    orientation,
):
    start_position, start_orientation = (
        L_get_world_pose(L_LEFT_EE_PATH)
    )

    target_position = start_position.copy()
    target_position[2] += L_LIFT_HEIGHT

    reached, final_position = L_move_left_ee(
        ctrl=ctrl,
        start_position=start_position,
        target_position=target_position,
        start_orientation=start_orientation,
        target_orientation=orientation,
        frames=L_LIFT_FRAMES,
        label="LEFT LIFT",
    )

    return (
        start_position,
        target_position,
        reached,
        final_position,
    )

def L_rotate_left_ee_orientation(
    ctrl,
    fixed_position,
    start_orientation,
    rotation_degree=None,
):
    """
    왼팔 EE 자세를 rotation_degree만큼 회전하되 실제 흡착 TCP 월드 위치를 고정한다.
 
    rotation_degree:
        None 또는 생략 시 L_EE_ROTATE_DEG(90도) 사용.
        180.0 지정 시 0→180도 단일 smoothstep으로 중간 정지 없이 회전.
 
    회전 프레임과 Hold 프레임 모두 rotation_degree에 비례해 자동 계산한다.
        90도  → rotate_frames = L_EE_ROTATE_FRAMES,  hold_frames = L_ROTATE_HOLD_FRAMES
        180도 → rotate_frames = L_EE_ROTATE_FRAMES*2, hold_frames = L_ROTATE_HOLD_FRAMES*2
 
    Hold 단계에서 자세 오차가 수렴할 때까지 확인 후 반환한다.
 
    반환:
        final_orientation, rotation_ok
    """
    if rotation_degree is None:
        rotation_degree = float(L_EE_ROTATE_DEG)
    else:
        rotation_degree = float(rotation_degree)
 
    scale = rotation_degree / max(float(L_EE_ROTATE_DEG), 1.0)
 
    # rotation_degree에 비례한 실제 회전 프레임 수
    rotate_frames = max(
        L_EE_ROTATE_FRAMES,
        int(round(L_EE_ROTATE_FRAMES * scale)),
    )
 
    # rotation_degree에 비례한 Hold 프레임 수
    hold_frames = L_ROTATE_HOLD_FRAMES
 
    # 180도 단일 회전 시 TCP가 더 많이 움직이므로 허용치를 넓힌다.
    tcp_abort_tolerance = (
        L_ROTATE_TCP_ABORT_TOLERANCE * 2.0
        if rotation_degree > 91.0
        else L_ROTATE_TCP_ABORT_TOLERANCE
    )
 
    # Hold 단계 자세 수렴 판정 기준
    ORI_CONV_THRESHOLD_DEG = 3.0
    ORI_STABLE_REQUIRED = 4
 
    requested_fixed_ee = np.asarray(fixed_position, dtype=float)
    start_orientation = L_normalize_quat(start_orientation)
 
    fixed_tcp = L_get_gripper_tcp_world().copy()
    tcp_offset_local = L_get_tcp_offset_in_ee_frame()
    gripper_interface = surface_gripper.acquire_surface_gripper_interface()
 
    def rotation_grip_is_secure():
        gripped = L_get_gripped_objects_map(gripper_interface)
        if L_ROTATE_REQUIRE_ALL_GRIPPERS:
            secure = bool(gripped) and all(
                len(objects) > 0 for objects in gripped.values()
            )
        else:
            secure = L_has_any_gripper_attached(gripped)
        return secure, gripped
 
    print("")
    print("=== LEFT TCP-LOCKED ORIENTATION ROTATE START ===")
    print("requested fixed EE position:", requested_fixed_ee)
    print("fixed TCP rotation center:", fixed_tcp)
    print("rotate degree:", rotation_degree)
    print("rotate frames:", rotate_frames)
    print("hold frames:", hold_frames)
    print("rotation sign:", L_LEFT_ROTATE_SIGN)
    print("tcp abort tolerance:", tcp_abort_tolerance)
    print("require all grippers:", L_ROTATE_REQUIRE_ALL_GRIPPERS)
 
    # 회전 전 그리퍼 확인 (아직 붙으려는 구간 — 무조건 close 유지)
    grip_secure, gripped_map = rotation_grip_is_secure()
    for confirm_frame in range(L_ROTATE_GRIP_CONFIRM_FRAMES):
        if grip_secure:
            break
 
        if confirm_frame % L_GRIP_RETRY_EVERY_FRAMES == 0:
            L_command_left_grippers_close()
 
        confirm_ee = L_ee_target_from_tcp_target(
            fixed_tcp,
            start_orientation,
            tcp_offset_local,
        )
        L_step_rmp_controller(
            ctrl,
            L_policy_target_from_world(ctrl["robot"], confirm_ee),
            start_orientation,
        )
        L_update_simulation()
        grip_secure, gripped_map = rotation_grip_is_secure()
 
    if not grip_secure:
        print("[FAIL] LEFT rotation cancelled: required grippers are not attached.")
        print("gripped objects:", gripped_map)
        return start_orientation.copy(), False
 
    final_orientation = start_orientation.copy()
    grip_loss_count = 0
 
    # ── 회전 루프 ──────────────────────────────────────────────────
    for frame in range(rotate_frames + 1):
        ratio = frame / float(max(rotate_frames, 1))
        alpha = L_smoothstep01(ratio)
 
        current_degree = L_LEFT_ROTATE_SIGN * rotation_degree * alpha
 
        q_delta = L_quat_from_axis_angle(
            [1.0, 0.0, 0.0],
            current_degree,
        )
        current_orientation = L_quat_mul(
            start_orientation,
            q_delta,
        )
        final_orientation = current_orientation.copy()
 
        ee_target = L_ee_target_from_tcp_target(
            fixed_tcp,
            current_orientation,
            tcp_offset_local,
        )
        policy_target = L_policy_target_from_world(
            ctrl["robot"],
            ee_target,
        )
 
        L_step_rmp_controller(ctrl, policy_target, current_orientation)
        L_update_simulation()
 
        grip_secure, gripped_map = rotation_grip_is_secure()
        actual_tcp = L_get_gripper_tcp_world()
        tcp_error = float(np.linalg.norm(actual_tcp - fixed_tcp))
 
        if grip_secure:
            grip_loss_count = 0
        else:
            grip_loss_count += 1
            if frame % L_GRIP_RETRY_EVERY_FRAMES == 0:
                L_command_left_grippers_close()
 
            print(
                "[WARN] LEFT grip temporarily unavailable during rotation:",
                grip_loss_count, "/", L_ROTATE_GRIP_LOSS_CONFIRM_FRAMES,
            )
 
            if grip_loss_count >= L_ROTATE_GRIP_LOSS_CONFIRM_FRAMES:
                print("[FAIL] LEFT grip lost continuously during TCP-locked rotation.")
                print("frame:", frame)
                print("gripped objects:", gripped_map)
                return current_orientation.copy(), False
 
        if tcp_error > tcp_abort_tolerance:
            print("[FAIL] LEFT TCP drift exceeded rotation tolerance.")
            print("frame:", frame)
            print("fixed TCP:", fixed_tcp)
            print("actual TCP:", actual_tcp)
            print("TCP error:", tcp_error)
            return current_orientation.copy(), False
 
    # ── Hold 단계: 자세 수렴 확인 ──────────────────────────────────
    final_ee_target = L_ee_target_from_tcp_target(
        fixed_tcp,
        final_orientation,
        tcp_offset_local,
    )
    final_policy_target = L_policy_target_from_world(
        ctrl["robot"],
        final_ee_target,
    )
 
    grip_loss_count = 0
    ori_stable_count = 0
 
    for hold_frame in range(hold_frames):
        L_step_rmp_controller(ctrl, final_policy_target, final_orientation)
        L_update_simulation()
 
        grip_secure, gripped_map = rotation_grip_is_secure()
        _, actual_ori = L_get_world_pose(L_LEFT_EE_PATH)
        ori_error = L_quat_angle_error_deg(actual_ori, final_orientation)
 
        if grip_secure:
            grip_loss_count = 0
        else:
            grip_loss_count += 1
            if hold_frame % L_GRIP_RETRY_EVERY_FRAMES == 0:
                L_command_left_grippers_close()
 
            if grip_loss_count >= L_ROTATE_GRIP_LOSS_CONFIRM_FRAMES:
                print("[FAIL] LEFT grip lost continuously during post-rotation hold.")
                print("gripped objects:", gripped_map)
                return final_orientation, False
 
        if grip_secure and ori_error < ORI_CONV_THRESHOLD_DEG:
            ori_stable_count += 1
            if ori_stable_count >= ORI_STABLE_REQUIRED:
                print(
                    f"[LEFT ROTATE] orientation converged:"
                    f" hold_frame={hold_frame}, ori_error={ori_error:.2f}deg"
                )
                break
        else:
            ori_stable_count = 0
 
    final_tcp = L_get_gripper_tcp_world()
    final_tcp_error = float(np.linalg.norm(final_tcp - fixed_tcp))
 
    print("final fixed TCP:", fixed_tcp)
    print("final actual TCP:", final_tcp)
    print("final TCP error:", final_tcp_error)
    print("=== LEFT TCP-LOCKED ORIENTATION ROTATE DONE ===")
 
    return final_orientation, (
        final_tcp_error <= tcp_abort_tolerance
    )

def L_wait_and_check_right_camera_barcode(node, label):
    """오른팔 카메라 /barcode_exist_R을 1초간 확인한다.

    회전 완료 토픽은 여기서 발행하지 않는다. 왼팔이 상자를 내려놓고
    Home 복귀까지 끝낸 뒤 execute_left_cycle()에서 발행한다.
    """
    print("")
    print("============================================")
    print(f"=== {label}: RIGHT CAMERA BARCODE CHECK ===")
    print("barcode topic:", L_BARCODE_TOPIC)
    print("wait seconds:", L_BARCODE_CHECK_SECONDS)
    print("wait frames:", L_BARCODE_CHECK_FRAMES)
    print("rotate-complete publish deferred until LEFT Home")
    print("============================================")

    node.reset_barcode_check()
    L_wait_frames(L_BARCODE_CHECK_FRAMES)

    status, detected = node.barcode_check_result()

    print("[RIGHT CAMERA CHECK] status:", status)
    print("[RIGHT CAMERA CHECK] msg received:", node.barcode_msg_received)
    print("[RIGHT CAMERA CHECK] msg count:", node.barcode_msg_count)
    print("[RIGHT CAMERA CHECK] last value:", node.barcode_exist)
    print("[RIGHT CAMERA CHECK] last msg time:", node.last_barcode_time)

    if status == "no_message":
        print("[RIGHT CAMERA CHECK] no message; treated as not detected")
    elif detected:
        print("[RIGHT CAMERA CHECK] /barcode_exist_R == 1")
    else:
        print("[RIGHT CAMERA CHECK] /barcode_exist_R == 0")

    return detected, status


def L_scan_left_two_faces_with_right_camera(
    ctrl,
    fixed_position,
    start_orientation,
    barcode_node,
    scan_steps,
):
    """
    왼팔 기준 최신 로직:

    1) /right_phase_done 모드(scan_steps=2)
       - 왼팔이 0→180도 단일 연속 회전한다 (중간 90도 정지 없음).
       - 180도 완료 후에만 /left_rotate_180 = 1을 발행한다.
       - 그 뒤 1초 정지하며 오른팔 카메라 /barcode_exist_R을 확인한다.
       - detected=True면 OK, 아니면 FAIL.

    2) /right_extra_done 모드(scan_steps=1)
       - 이미 왼팔 카메라 barcode_L 감지 후 오른팔 extra 90까지 끝난 상태다.
       - 왼팔은 시계방향 90도만 회전한다.
       - 이 경우 /left_rotate_180은 발행하지 않는다.
       - prior barcode 감지 상황이므로 OK로 처리한다.
    """
    fixed_position = np.asarray(fixed_position, dtype=float)
    final_orientation = L_normalize_quat(start_orientation)
    detected = False
    last_status = "not_checked"
    completed_steps = 0
    scan_steps = int(scan_steps)

    print("")
    print("================================================")
    print("=== LEFT ROTATE MODE START ===")
    print("fixed EE position:", fixed_position)
    print("scan steps:", scan_steps)
    print("rotation sign:", L_LEFT_ROTATE_SIGN)
    print("left 180 topic:", L_LEFT_ROTATE_180_TOPIC)
    print("left 90 topic:", L_LEFT_ROTATE_90_TOPIC)
    print("barcode topic:", L_BARCODE_TOPIC)
    print("barcode detected value:", L_BARCODE_DETECTED_VALUE)
    print("================================================")

    if scan_steps <= 1:
        # /right_extra_done 모드: 90도 단일 회전
        print("")
        print("==============================================")
        print("=== LEFT 90 DEG MODE FROM /right_extra_done ===")
        print("/left_rotate_90 will be published after one 90 deg rotation.")
        print("Reason: barcode_L was already detected and right extra 90 was already done.")
        print("==============================================")

        final_orientation, rotation_ok = L_rotate_left_ee_orientation(
            ctrl=ctrl,
            fixed_position=fixed_position,
            start_orientation=final_orientation,
        )
        if not rotation_ok:
            return (
                final_orientation,
                False,
                0,
                "grip_or_tcp_lost_during_rotation",
            )

        completed_steps = 1
        detected = True
        last_status = "right_extra_done_90_mode_prior_barcode_ok"

        print("[LEFT 90 MODE] Completed one 90 deg rotation.")
        print("[LEFT 90 MODE] /left_rotate_90 publish deferred until LEFT Home.")
        print("[LEFT 90 MODE] /left_rotate_180 not published.")
        print("[LEFT 90 MODE] Result treated as OK because /right_extra_done means barcode_L was already detected.")

        print("=== LEFT ROTATE MODE DONE ===")
        return final_orientation, detected, completed_steps, last_status

    # /right_phase_done 모드: 0→180도 단일 연속 회전
    print("")
    print("==============================================")
    print("=== LEFT 180 DEG MODE FROM /right_phase_done ===")
    print("Single continuous 180 deg rotation (no mid-stop at 90 deg)")
    print("==============================================")

    final_orientation, rotation_ok = L_rotate_left_ee_orientation(
        ctrl=ctrl,
        fixed_position=fixed_position,
        start_orientation=final_orientation,
        rotation_degree=180.0,
    )
    if not rotation_ok:
        return (
            final_orientation,
            False,
            completed_steps,
            "grip_or_tcp_lost_during_rotation",
        )

    completed_steps = 2
    print("[LEFT 180 MODE] 180 deg rotation done.")

    detected, last_status = L_wait_and_check_right_camera_barcode(
        barcode_node,
        label="LEFT 180 DEG DONE",
    )

    if detected:
        print("")
        print("==============================================")
        print("[LEFT 180 MODE SUCCESS] Barcode detected by RIGHT camera after left 180 deg.")
        print("==============================================")
    else:
        print("")
        print("==============================================")
        print("[LEFT 180 MODE FAIL] Barcode was not detected by RIGHT camera after left 180 deg.")
        print("[LEFT 180 MODE FAIL] All 6 faces are considered checked.")
        print("last barcode check status:", last_status)
        print("==============================================")

    print("=== LEFT ROTATE MODE DONE ===")
    return final_orientation, detected, completed_steps, last_status

def L_place_left_robot(
    ctrl,
    place_tcp_target,
    place_orientation,
):
    """
    회전 후 박스를 원래 흡착 위치에 내려놓는다.

    기존에는 파지 당시 ee_link 위치를 그대로 배치 목표로 사용했다. 하지만 EE 자세가
    90도/180도 회전한 뒤에는 같은 EE 위치라도 EE→TCP 오프셋의 월드 방향이 달라져
    흡착 TCP와 박스가 옆이나 아래로 이동한다. 이제 파지 당시 실제 TCP를 배치 기준으로
    사용하고, 최종 회전 자세에서 그 TCP에 필요한 EE 위치를 역산한다.
    """
    current_position, current_orientation = L_get_world_pose(L_LEFT_EE_PATH)
    place_orientation = L_normalize_quat(place_orientation)

    # 현재 장착 구조의 EE 로컬 TCP 오프셋. 자세가 변해도 로컬 오프셋은 동일하다.
    tcp_offset_local = L_get_tcp_offset_in_ee_frame()

    target_tcp = np.asarray(place_tcp_target, dtype=float).copy()
    # 양수이면 원래 접촉점보다 조금 위에서 해제한다.
    target_tcp[2] += L_PLACE_Z_OFFSET

    target_ee = L_ee_target_from_tcp_target(
        target_tcp,
        place_orientation,
        tcp_offset_local,
    )

    print("")
    print("=== LEFT TCP-REFERENCED PLACE PLAN ===")
    print("current EE:", current_position)
    print("requested place TCP:", target_tcp)
    print("calculated place EE:", target_ee)
    print("place orientation wxyz:", place_orientation)

    reached, final_position = L_move_left_ee(
        ctrl=ctrl,
        start_position=current_position,
        target_position=target_ee,
        start_orientation=current_orientation,
        target_orientation=place_orientation,
        frames=L_PLACE_FRAMES,
        label="LEFT PLACE DOWN WITH TCP TARGET",
    )

    final_tcp = L_get_gripper_tcp_world()
    tcp_error = float(np.linalg.norm(final_tcp - target_tcp))
    print("actual final LEFT TCP:", final_tcp)
    print("LEFT place TCP error:", tcp_error)

    return reached, final_position


def L_retreat_left_robot(
    ctrl,
    retreat_orientation,
):
    current_position, current_orientation = (
        L_get_world_pose(L_LEFT_EE_PATH)
    )

    target_position = current_position.copy()
    target_position[1] += L_RETREAT_Y
    target_position[2] += L_RETREAT_Z

    reached, final_position = L_move_left_ee(
        ctrl=ctrl,
        start_position=current_position,
        target_position=target_position,
        start_orientation=current_orientation,
        target_orientation=retreat_orientation,
        frames=L_RETREAT_FRAMES,
        label="LEFT RETREAT",
    )

    return reached, final_position


# ============================================================
# 초기 관절 자세 복귀
# ============================================================

def L_get_robot_joint_positions(ctrl):
    return np.array([
        L_get_current_joint_rad(
            L_LEFT_ROBOT,
            joint_name,
        )
        for joint_name in ctrl["active_joints"]
    ], dtype=float)


def L_wrapped_joint_delta(
    target_q,
    current_q,
):
    target_q = np.asarray(
        target_q,
        dtype=float,
    )
    current_q = np.asarray(
        current_q,
        dtype=float,
    )

    return np.arctan2(
        np.sin(target_q - current_q),
        np.cos(target_q - current_q),
    )


def L_initialize_vertical_side_home_pose(ctrl):
    """
    현재 왼팔 EE 위치를 유지하면서 다음 자세로 교정한 뒤 새 Home으로 저장한다.

    - 흡착면 법선(EE local +X): 월드 -Y
    - 흡착면: 지면과 수직
    - wrist3 기준축(EE local +Z): 월드 XY 평면과 평행
    """
    start_position, start_orientation = L_get_world_pose(L_LEFT_EE_PATH)
    target_orientation = L_make_vertical_side_grasp_orientation(start_orientation)
    policy_target = L_policy_target_from_world(ctrl["robot"], start_position)

    print("")
    print("=================================================")
    print("=== INITIAL LEFT VERTICAL SIDE HOME START ===")
    print("=================================================")
    print("fixed initial EE position:", start_position)
    print("start orientation wxyz:", start_orientation)
    print("target orientation wxyz:", target_orientation)
    print("target suction normal:", L_get_suction_axis_world(target_orientation))
    print("target wrist3 reference axis:", L_get_wrist3_reference_axis_world(target_orientation))
    print("start side-normal error deg:", L_get_side_normal_error_deg(start_orientation))
    print("start wrist3 horizontal error deg:", L_get_wrist3_horizontal_error_deg(start_orientation))

    for frame in range(L_INITIAL_SIDE_FRAMES + 1):
        ratio = frame / float(max(L_INITIAL_SIDE_FRAMES, 1))
        alpha = L_smoothstep01(ratio)
        orientation_now = L_quat_slerp(
            start_orientation,
            target_orientation,
            alpha,
        )
        L_step_rmp_controller(ctrl, policy_target, orientation_now)

        if DEBUG_MOTION and (frame % 15 == 0 or frame == L_INITIAL_SIDE_FRAMES):
            actual_position, actual_orientation = L_get_world_pose(L_LEFT_EE_PATH)
            _dlog(DEBUG_MOTION, f"[LEFT SIDE HOME FRAME {frame:03d}/{L_INITIAL_SIDE_FRAMES}]")
            _dlog(DEBUG_MOTION, "EE position:", actual_position)
            _dlog(DEBUG_MOTION, "position error:", np.linalg.norm(actual_position - start_position))
            _dlog(DEBUG_MOTION, "side normal error deg:", L_get_side_normal_error_deg(actual_orientation))
            _dlog(DEBUG_MOTION, "wrist3 horizontal error deg:", L_get_wrist3_horizontal_error_deg(actual_orientation))
            _dlog(DEBUG_MOTION, "orientation target error deg:", L_quat_angle_error_deg(actual_orientation, target_orientation))

        L_update_simulation()

    stable_count = 0
    reached = False
    for settle_frame in range(L_INITIAL_SIDE_CONVERGENCE_FRAMES):
        L_step_rmp_controller(ctrl, policy_target, target_orientation)
        L_update_simulation()

        actual_position, actual_orientation = L_get_world_pose(L_LEFT_EE_PATH)
        position_error = float(np.linalg.norm(actual_position - start_position))
        normal_error = L_get_side_normal_error_deg(actual_orientation)
        wrist_error = L_get_wrist3_horizontal_error_deg(actual_orientation)

        if (
            position_error <= L_INITIAL_SIDE_POSITION_TOLERANCE
            and normal_error <= L_INITIAL_SIDE_NORMAL_TOLERANCE_DEG
            and wrist_error <= L_INITIAL_WRIST3_HORIZONTAL_TOLERANCE_DEG
        ):
            stable_count += 1
        else:
            stable_count = 0

        if settle_frame % 30 == 0:
            print(
                f"[LEFT SIDE HOME CONVERGENCE {settle_frame:03d}/"
                f"{L_INITIAL_SIDE_CONVERGENCE_FRAMES}]"
            )
            print("position error:", position_error)
            print("normal error deg:", normal_error)
            print("wrist3 horizontal error deg:", wrist_error)
            print("stable frames:", stable_count)

        if stable_count >= L_INITIAL_SIDE_STABLE_FRAMES:
            reached = True
            break

    home_position, home_orientation = L_get_world_pose(L_LEFT_EE_PATH)
    home_q = L_get_robot_joint_positions(ctrl)
    final_position_error = float(np.linalg.norm(home_position - start_position))
    final_normal_error = L_get_side_normal_error_deg(home_orientation)
    final_wrist_error = L_get_wrist3_horizontal_error_deg(home_orientation)

    print("")
    print("=== INITIAL LEFT VERTICAL SIDE HOME RESULT ===")
    print("home reached:", reached)
    print("home EE position:", home_position)
    print("home EE orientation wxyz:", home_orientation)
    print("home joint deg:", np.rad2deg(home_q))
    print("home suction normal world:", L_get_suction_axis_world(home_orientation))
    print("home wrist3 reference axis world:", L_get_wrist3_reference_axis_world(home_orientation))
    print("home side-normal error deg:", final_normal_error)
    print("home wrist3 horizontal error deg:", final_wrist_error)
    print("home position shift:", final_position_error)

    if not reached:
        raise RuntimeError(
            "Failed to establish LEFT vertical side Home pose. "
            f"position_error={final_position_error:.6f} m, "
            f"normal_error={final_normal_error:.3f} deg, "
            f"wrist3_horizontal_error={final_wrist_error:.3f} deg"
        )

    ctrl["home_q"] = home_q.copy()
    ctrl["home_ee_position"] = home_position.copy()
    ctrl["home_ee_orientation"] = home_orientation.copy()
    ctrl["q"] = home_q.copy()
    ctrl["qd"] = np.zeros_like(home_q)

    print("================================================")
    print("=== INITIAL LEFT VERTICAL SIDE HOME DONE ===")
    print("================================================")
    return home_position, home_orientation, home_q


def L_return_left_robot_home(ctrl):
    start_q = L_get_robot_joint_positions(
        ctrl
    )
    home_q = np.asarray(
        ctrl["home_q"],
        dtype=float,
    ).copy()

    delta_q = L_wrapped_joint_delta(
        home_q,
        start_q,
    )
    commanded_home_q = start_q + delta_q

    print("")
    print("===================================")
    print("=== LEFT ROBOT RETURN HOME START ===")
    print("===================================")
    print(
        "current joint deg:",
        np.rad2deg(start_q),
    )
    print(
        "saved home joint deg:",
        np.rad2deg(home_q),
    )
    print(
        "shortest delta deg:",
        np.rad2deg(delta_q),
    )
    print("frames:", L_RETURN_HOME_FRAMES)

    for frame in range(
        L_RETURN_HOME_FRAMES + 1
    ):
        ratio = frame / float(
            max(L_RETURN_HOME_FRAMES, 1)
        )
        alpha = L_smoothstep01(ratio)

        q_now = start_q + delta_q * alpha

        if frame == L_RETURN_HOME_FRAMES:
            q_now = commanded_home_q.copy()

        L_set_drive_targets_from_rad(
            L_LEFT_ROBOT,
            ctrl["active_joints"],
            q_now,
        )

        ctrl["q"] = q_now.copy()
        ctrl["qd"] = np.zeros_like(q_now)

        if (
            frame % 15 == 0
            or frame == L_RETURN_HOME_FRAMES
        ):
            actual_q = L_get_robot_joint_positions(
                ctrl
            )
            error_q = L_wrapped_joint_delta(
                commanded_home_q,
                actual_q,
            )

            _dlog(DEBUG_JOINT, f"[LEFT HOME FRAME {frame:03d}/{L_RETURN_HOME_FRAMES}]")
            _dlog(DEBUG_JOINT, "command joint deg:", np.rad2deg(q_now))
            _dlog(DEBUG_JOINT, "actual joint deg:", np.rad2deg(actual_q))
            _dlog(DEBUG_JOINT, "max error deg:", np.max(np.abs(np.rad2deg(error_q))))

        L_update_simulation()

    reached = False

    for settle_frame in range(
        L_HOME_SETTLE_FRAMES
    ):
        L_set_drive_targets_from_rad(
            L_LEFT_ROBOT,
            ctrl["active_joints"],
            commanded_home_q,
        )

        ctrl["q"] = commanded_home_q.copy()
        ctrl["qd"] = np.zeros_like(
            commanded_home_q
        )

        L_update_simulation()

        actual_q = L_get_robot_joint_positions(
            ctrl
        )
        error_degree = np.abs(
            np.rad2deg(
                L_wrapped_joint_delta(
                    commanded_home_q,
                    actual_q,
                )
            )
        )

        if (
            np.max(error_degree)
            <= L_HOME_JOINT_TOLERANCE_DEG
        ):
            reached = True
            break

    final_q = L_get_robot_joint_positions(
        ctrl
    )
    final_error_degree = np.abs(
        np.rad2deg(
            L_wrapped_joint_delta(
                commanded_home_q,
                final_q,
            )
        )
    )

    final_ee_position, final_ee_orientation = (
        L_get_world_pose(L_LEFT_EE_PATH)
    )

    print(
        "final joint deg:",
        np.rad2deg(final_q),
    )
    print(
        "final joint error deg:",
        final_error_degree,
    )
    print(
        "saved home EE position:",
        ctrl["home_ee_position"],
    )
    print(
        "returned EE position:",
        final_ee_position,
    )
    print(
        "EE position error:",
        np.linalg.norm(
            final_ee_position
            - ctrl["home_ee_position"]
        ),
    )
    print(
        "returned EE orientation wxyz:",
        final_ee_orientation,
    )
    print("home reached:", reached)
    print("==================================")
    print("=== LEFT ROBOT RETURN HOME DONE ===")
    print("==================================")

    return reached
