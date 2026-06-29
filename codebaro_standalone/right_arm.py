"""Right-arm top grasp, inspection rotation, placement, and home control."""

import numpy as np
import carb
from isaacsim.core.prims import Articulation
from isaacsim.robot_motion.motion_generation import RmpFlow, interface_config_loader
import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from . import runtime
from .config import *
from .config import _dlog
from .math_utils import *
from .simulation import update_simulation, wait_frames

stage = runtime.stage
physx_query = runtime.physx_query
_ATTACHMENT_JOINT_PATHS_CACHE = None

# 초기 카메라 하향 자세를 사용하므로 아래 고정 자세는 기본적으로 사용하지 않는다.
RIGHT_GRASP_ORIENTATION = quat_from_euler_deg(
    *RIGHT_GRASP_ORIENTATION_EULER_DEG
)

def get_world_pose(path):
    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():
        raise RuntimeError("Prim not found: " + path)

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
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

    orientation = normalize_quat([
        float(gf_quat.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    ])

    return position, orientation


def get_world_pos(path):
    return get_world_pose(path)[0]


def _normalized_attr_name(name):
    return str(name).lower().replace("_", "").replace(":", "")


def get_attachment_joint_paths():
    """SurfaceGripper의 Attachment Points 관계가 가리키는 D6 joint 경로를 반환한다."""
    global _ATTACHMENT_JOINT_PATHS_CACHE

    if _ATTACHMENT_JOINT_PATHS_CACHE is not None:
        return list(_ATTACHMENT_JOINT_PATHS_CACHE)

    paths = []

    for gripper_path in RIGHT_GRIPPERS:
        prim = stage.GetPrimAtPath(gripper_path)
        if not prim.IsValid():
            raise RuntimeError("Gripper prim not found: " + gripper_path)

        found_relation = False
        for rel in prim.GetRelationships():
            normalized = _normalized_attr_name(rel.GetName())
            if "attachmentpoints" not in normalized:
                continue

            found_relation = True
            for target in rel.GetTargets():
                path = str(target)
                if path not in paths:
                    paths.append(path)

        if not found_relation:
            print("[WARN] Attachment Points relationship not found:", gripper_path)

    if not paths:
        raise RuntimeError(
            "No SurfaceGripper attachment D6 joints were found. "
            "Set each SurfaceGripper Prim's Attachment Points relationship."
        )

    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError("Attachment joint prim not found: " + path)

    _ATTACHMENT_JOINT_PATHS_CACHE = tuple(paths)
    return list(paths)


def get_gripper_tcp_points_world():
    """실제 흡착점인 Attachment D6 joint들의 월드 위치를 반환한다."""
    return [get_world_pos(path) for path in get_attachment_joint_paths()]


def get_gripper_tcp_world():
    """Attachment D6 joint 월드 위치의 중점을 흡착 TCP로 사용한다."""
    points = get_gripper_tcp_points_world()
    return np.mean(np.stack(points, axis=0), axis=0)


def print_attachment_diagnostics():
    print("")
    print("=== RIGHT SURFACE GRIPPER ATTACHMENT DIAGNOSTICS ===")

    for path in get_attachment_joint_paths():
        pos, orientation = get_world_pose(path)
        forward_value = None
        forward_attr_name = None

        prim = stage.GetPrimAtPath(path)
        for attr in prim.GetAttributes():
            normalized = _normalized_attr_name(attr.GetName())
            if "forwardaxis" in normalized:
                forward_attr_name = attr.GetName()
                forward_value = attr.Get()
                break

        print("attachment joint:", path)
        print("  world position:", pos)
        print("  world orientation wxyz:", orientation)
        print("  forward axis attribute:", forward_attr_name, forward_value)


def _hit_attr(hit, *names, default=None):
    for name in names:
        if hasattr(hit, name):
            return getattr(hit, name)
    return default


def _vec3_to_numpy(value):
    if value is None:
        return np.zeros(3, dtype=float)
    return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)


def raycast_non_robot_hits(origin, direction, distance):
    """로봇 자체 Collider를 제외한 PhysX raycast hit들을 거리순으로 반환한다."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    hits = []

    def report(hit):
        body = str(_hit_attr(hit, "rigid_body", "rigidBody", default=""))
        collision = str(_hit_attr(hit, "collision", default=""))

        # 자신의 로봇 링크/툴 Collider는 표면 후보에서 제외한다.
        if body.startswith(RIGHT_ROBOT) or collision.startswith(RIGHT_ROBOT):
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
        print("[WARN] raycast_all failed; falling back to raycast_closest:", repr(exc))
        closest = physx_query.raycast_closest(o, d, float(distance))
        if closest.get("hit", False):
            body = str(closest.get("rigidBody", ""))
            collision = str(closest.get("collision", ""))
            if not body.startswith(RIGHT_ROBOT) and not collision.startswith(RIGHT_ROBOT):
                hits.append({
                    "rigidBody": body,
                    "collision": collision,
                    "distance": float(closest.get("distance", np.inf)),
                    "position": _vec3_to_numpy(closest.get("position")),
                    "normal": _vec3_to_numpy(closest.get("normal")),
                })

    hits.sort(key=lambda item: item["distance"])
    return hits


def detect_top_surface_with_rays(surface_target, ray_direction=None):
    """
    오른팔 상면 Collider를 다중 Ray로 검사한다.

    실제 upper/lower Attachment Ray와 TCP 중심 Ray 외에 두 컵 배열에
    직교하는 방향으로 보조 Ray를 추가한다. 이 비공선 hit들을 이용하면
    상면의 roll/pitch를 모두 추정할 수 있다.
    """
    surface_target = np.asarray(surface_target, dtype=float)
    if ray_direction is None:
        direction = RIGHT_APPROACH_DIRECTION.copy()
    else:
        direction = np.asarray(ray_direction, dtype=float).copy()
    direction /= np.linalg.norm(direction)

    joint_paths = get_attachment_joint_paths()
    joint_points = get_gripper_tcp_points_world()
    center_point = np.mean(np.stack(joint_points, axis=0), axis=0)

    ray_specs = [("TCP_CENTER", center_point, False)]

    for joint_path, point in zip(joint_paths, joint_points):
        ray_specs.append((joint_path, point, True))
        inner_point = center_point + 0.65 * (point - center_point)
        ray_specs.append((joint_path + "::INNER", inner_point, False))

    # 두 컵 배열 방향을 상면 접선축으로 사용하고, 접근 방향과의 외적으로
    # 두 번째 접선축을 만든다. 이 축의 +/- 보조 Ray가 비공선 평면점을 제공한다.
    if len(joint_points) >= 2:
        baseline = np.asarray(joint_points[1] - joint_points[0], dtype=float)
        baseline -= direction * float(np.dot(baseline, direction))
    else:
        baseline = np.zeros(3, dtype=float)

    if np.linalg.norm(baseline) < 1.0e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(fallback, direction))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        baseline = fallback - direction * float(np.dot(fallback, direction))

    baseline /= np.linalg.norm(baseline)
    orthogonal = np.cross(direction, baseline)
    orthogonal /= np.linalg.norm(orthogonal)

    ray_specs.extend([
        ("TOP_PROBE_ORTHO_POS", center_point + orthogonal * TOP_RAY_PROBE_OFFSET, False),
        ("TOP_PROBE_ORTHO_NEG", center_point - orthogonal * TOP_RAY_PROBE_OFFSET, False),
    ])

    detections = []
    for ray_name, point, is_actual_joint in ray_specs:
        origin = point + direction * RAY_ORIGIN_OFFSET
        hits = raycast_non_robot_hits(origin, direction, RAYCAST_RANGE)

        accepted = None
        for hit in hits:
            plane_error = abs(float(np.dot(
                hit["position"] - surface_target,
                direction,
            )))
            if plane_error <= RAY_VISION_Z_TOLERANCE:
                accepted = hit
                break

        detections.append({
            "joint": ray_name,
            "origin": origin,
            "hit": accepted,
            "all_hits": hits,
            "is_center": ray_name == "TCP_CENTER",
            "is_actual_joint": bool(is_actual_joint),
        })

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


def estimate_top_surface_approach_normal(surface_target, current_orientation):
    """
    다중 Ray hit로 실제 상면 평면을 적합하고 표면을 향하는 법선을 반환한다.

    반환:
        (approach_normal, diagnostics) 또는 (None, diagnostics)
    """
    current_orientation = normalize_quat(current_orientation)
    current_direction = get_ee_suction_axis_world(current_orientation)
    current_direction /= np.linalg.norm(current_direction)

    _, details = detect_top_surface_with_rays(
        surface_target,
        current_direction,
    )

    valid_items = [item for item in details if item.get("hit") is not None]
    diagnostics = {
        "details": details,
        "method": None,
        "fit_residual": None,
        "used_hits": 0,
        "tilt_from_down_deg": None,
    }

    if not valid_items:
        return None, diagnostics

    # 중심 Ray가 본 rigid body를 우선 기준으로 삼아 다른 물체의 hit 혼입을 막는다.
    center_item = next(
        (item for item in valid_items if item.get("is_center")),
        None,
    )
    if center_item is not None:
        center_hit = center_item["hit"]
        center_body = str(center_hit.get("rigidBody", ""))
        center_collision = str(center_hit.get("collision", ""))

        def same_surface(item):
            hit = item["hit"]
            body = str(hit.get("rigidBody", ""))
            collision = str(hit.get("collision", ""))
            if center_body:
                return body == center_body
            if center_collision:
                return collision == center_collision
            return True

        same_items = [item for item in valid_items if same_surface(item)]
        if same_items:
            valid_items = same_items

    points = np.stack(
        [np.asarray(item["hit"]["position"], dtype=float) for item in valid_items],
        axis=0,
    )
    normals = [
        np.asarray(item["hit"]["normal"], dtype=float)
        for item in valid_items
        if np.linalg.norm(np.asarray(item["hit"]["normal"], dtype=float)) > 1.0e-8
    ]

    estimated = None
    fit_residual = None

    if len(points) >= TOP_PLANE_MIN_HITS:
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

        # 두 개 이상의 독립 접선 방향이 있어야 실제 평면 법선을 얻을 수 있다.
        if len(singular_values) >= 2 and singular_values[1] > 1.0e-5:
            estimated = np.asarray(vh[-1], dtype=float)
            estimated /= np.linalg.norm(estimated)
            residuals = np.abs(centered @ estimated)
            fit_residual = float(np.sqrt(np.mean(residuals ** 2)))
            diagnostics["method"] = "multi_ray_plane_svd"

            if fit_residual > TOP_PLANE_FIT_RESIDUAL_TOLERANCE:
                print(
                    "[RIGHT TOP ALIGN] Plane-fit residual is too large:",
                    fit_residual,
                )
                estimated = None

    # 보조 Ray가 상자 경계를 벗어난 경우 PhysX hit normal 평균으로 안전하게 대체한다.
    if estimated is None and normals:
        normal_sum = np.sum(
            [normal / np.linalg.norm(normal) for normal in normals],
            axis=0,
        )
        if np.linalg.norm(normal_sum) > 1.0e-8:
            estimated = normal_sum / np.linalg.norm(normal_sum)
            diagnostics["method"] = "physx_hit_normal_average"

    diagnostics["fit_residual"] = fit_residual
    diagnostics["used_hits"] = len(valid_items)

    if estimated is None:
        return None, diagnostics

    # 평면 법선 부호를 현재 그리퍼의 표면 접근 방향과 같게 맞춘다.
    if float(np.dot(estimated, current_direction)) < 0.0:
        estimated = -estimated

    world_down = EE_WORLD_DOWN_AXIS / np.linalg.norm(EE_WORLD_DOWN_AXIS)
    tilt_from_down_deg = angle_between_vectors_deg(estimated, world_down)
    diagnostics["tilt_from_down_deg"] = tilt_from_down_deg

    if tilt_from_down_deg > TOP_PLANE_MAX_TILT_FROM_WORLD_DOWN_DEG:
        print(
            "[RIGHT TOP ALIGN] Detected normal is not a valid top surface; "
            f"tilt_from_down={tilt_from_down_deg:.3f} deg"
        )
        return None, diagnostics

    diagnostics["approach_normal"] = estimated.copy()
    return estimated, diagnostics


def auto_align_right_tool_to_top_plane(
    ctrl,
    surface_target,
    current_target_orientation,
    tcp_offset_local,
):
    """
    다중 Ray로 추정한 실제 상면 법선에 EE 로컬 +X 흡착축을 맞춘다.
    자세를 변경하는 동안 TCP 중심은 같은 월드 위치에 고정한다.

    반환:
        aligned_orientation, correction_performed, plane_verified
    """
    current_target_orientation = normalize_quat(current_target_orientation)
    current_direction = get_ee_suction_axis_world(current_target_orientation)
    current_direction /= np.linalg.norm(current_direction)

    estimated_direction, diagnostics = estimate_top_surface_approach_normal(
        surface_target,
        current_target_orientation,
    )

    print("")
    print("=== RIGHT TOP PLANE ALIGNMENT CHECK ===")
    print("plane estimation method:", diagnostics.get("method"))
    print("used ray hits:", diagnostics.get("used_hits"))
    print("fit residual:", diagnostics.get("fit_residual"))
    print("tilt from world -Z deg:", diagnostics.get("tilt_from_down_deg"))

    if estimated_direction is None:
        print("[FAIL] RIGHT top-surface plane could not be verified by Raycast.")
        return current_target_orientation, False, False

    aligned_orientation = make_top_grasp_orientation_from_normal(
        current_target_orientation,
        estimated_direction,
    )
    correction_deg = angle_between_vectors_deg(
        current_direction,
        estimated_direction,
    )

    print("current suction normal:", current_direction)
    print("estimated surface approach normal:", estimated_direction)
    print("auto-align correction deg:", correction_deg)

    if correction_deg > TOP_PLANE_AUTO_ALIGN_MAX_DEG:
        print("[FAIL] RIGHT estimated top-plane correction exceeds safety limit.")
        return current_target_orientation, False, False

    if correction_deg <= TOP_PLANE_AUTO_ALIGN_MIN_DEG:
        print("[RIGHT TOP ALIGN] Tool face is already parallel enough.")
        return aligned_orientation, False, True

    fixed_tcp = get_gripper_tcp_world().copy()
    start_orientation = get_world_pose(RIGHT_EE_PATH)[1]

    for frame in range(TOP_PLANE_AUTO_ALIGN_FRAMES + 1):
        alpha = smoothstep01(
            frame / float(max(TOP_PLANE_AUTO_ALIGN_FRAMES, 1))
        )
        orientation_now = quat_slerp(
            start_orientation,
            aligned_orientation,
            alpha,
        )
        ee_target = ee_target_from_tcp_target(
            fixed_tcp,
            orientation_now,
            tcp_offset_local,
        )
        step_rmp_controller(
            ctrl,
            policy_target_from_world(ctrl["robot"], ee_target),
            orientation_now,
        )
        update_simulation()

        if DEBUG_GRIPPER and (frame % 10 == 0 or frame == TOP_PLANE_AUTO_ALIGN_FRAMES):
            tcp_now = get_gripper_tcp_world()
            actual_orientation = get_world_pose(RIGHT_EE_PATH)[1]
            actual_normal = get_ee_suction_axis_world(actual_orientation)
            _dlog(DEBUG_GRIPPER,
                f"[RIGHT TOP ALIGN {frame:03d}/{TOP_PLANE_AUTO_ALIGN_FRAMES}]",
                "TCP error=",
                round(float(np.linalg.norm(tcp_now - fixed_tcp)), 6),
                "normal error deg=",
                round(angle_between_vectors_deg(actual_normal, estimated_direction), 4),
            )

    print("=== RIGHT TOP PLANE ALIGNMENT DONE ===")
    return aligned_orientation, True, True


def configure_surface_grippers():
    """USD SurfaceGripper 속성을 찾아 접촉 감지 거리를 작게 설정한다."""
    print("")
    print("=== CONFIGURE RIGHT SURFACE GRIPPERS ===")

    for gripper_path in RIGHT_GRIPPERS:
        prim = stage.GetPrimAtPath(gripper_path)

        if not prim.IsValid():
            raise RuntimeError("Gripper prim not found: " + gripper_path)

        max_distance_set = False
        retry_interval_set = False

        for attr in prim.GetAttributes():
            normalized = _normalized_attr_name(attr.GetName())

            try:
                if "maxgripdistance" in normalized:
                    attr.Set(float(SURFACE_MAX_GRIP_DISTANCE))
                    max_distance_set = True
                    print(
                        gripper_path,
                        attr.GetName(),
                        "=",
                        SURFACE_MAX_GRIP_DISTANCE,
                    )

                elif "retryinterval" in normalized:
                    attr.Set(float(SURFACE_RETRY_INTERVAL))
                    retry_interval_set = True
                    print(
                        gripper_path,
                        attr.GetName(),
                        "=",
                        SURFACE_RETRY_INTERVAL,
                    )

            except Exception as exc:
                print(
                    "[WARN] gripper property set failed:",
                    gripper_path,
                    attr.GetName(),
                    repr(exc),
                )

        if not max_distance_set:
            print(
                "[WARN] Max Grip Distance attribute not found:",
                gripper_path,
            )

        if not retry_interval_set:
            print(
                "[WARN] Retry Interval attribute not found:",
                gripper_path,
            )


def get_gripped_objects_map(sg):
    result = {}

    for gripper_path in RIGHT_GRIPPERS:
        try:
            result[gripper_path] = list(
                sg.get_gripped_objects(gripper_path)
            )
        except Exception as exc:
            print(
                "[WARN] get_gripped_objects failed:",
                gripper_path,
                repr(exc),
            )
            result[gripper_path] = []

    return result


def has_any_right_gripper_attached(gripped_map):
    """오른팔 upper/lower 중 하나 이상이 물체를 잡았는지 반환한다."""
    return any(
        len(objects) > 0 for objects in gripped_map.values()
    )


def have_all_right_grippers_attached(gripped_map):
    """오른팔 upper/lower가 모두 물체를 잡았을 때만 True를 반환한다."""
    return bool(gripped_map) and all(
        len(objects) > 0 for objects in gripped_map.values()
    )


def has_any_gripped_object(gripped_map):
    """기존 호출 호환용. 오른팔 두 흡착컵이 모두 붙었을 때 True다."""
    return have_all_right_grippers_attached(gripped_map)


def command_right_grippers_open(settle_frames=3):
    sg = surface_gripper.acquire_surface_gripper_interface()

    for gripper_path in RIGHT_GRIPPERS:
        try:
            sg.open_gripper(gripper_path)
        except Exception as exc:
            print("[WARN] open_gripper failed:", gripper_path, repr(exc))

    wait_frames(settle_frames)


def command_right_grippers_close():
    sg = surface_gripper.acquire_surface_gripper_interface()

    for gripper_path in RIGHT_GRIPPERS:
        try:
            sg.close_gripper(gripper_path)
        except Exception as exc:
            print("[WARN] close_gripper failed:", gripper_path, repr(exc))

    return sg


def get_tcp_offset_in_ee_frame():
    """현재 EE 자세에서 EE 원점→흡착 TCP 벡터를 EE 로컬 좌표로 변환한다."""
    ee_pos, ee_orientation = get_world_pose(RIGHT_EE_PATH)
    tcp_world = get_gripper_tcp_world()
    offset_world = tcp_world - ee_pos
    offset_local = quat_rotate_vector(
        quat_conjugate(ee_orientation),
        offset_world,
    )

    print("")
    print("=== RIGHT TCP CALIBRATION ===")
    print("EE world position:", ee_pos)
    print("TCP world midpoint:", tcp_world)
    print("EE -> TCP world offset:", offset_world)
    print("EE -> TCP local offset:", offset_local)

    return offset_local


def ee_target_from_tcp_target(tcp_world_target, ee_orientation, tcp_offset_local):
    """원하는 흡착 TCP 위치로부터 필요한 ee_link 월드 목표를 계산한다."""
    tcp_offset_world = quat_rotate_vector(
        ee_orientation,
        tcp_offset_local,
    )
    return np.asarray(tcp_world_target, dtype=float) - tcp_offset_world


def get_ur10_rmpflow():
    config = interface_config_loader.load_supported_motion_policy_config(
        "UR10",
        "RMPflow",
    )

    return RmpFlow(**config)


def make_robot(robot_path, name):
    robot = Articulation(
        prim_paths_expr=robot_path,
        name=name,
    )

    robot.initialize()
    return robot


def get_base_pose(robot):
    pos, quat = robot.get_world_poses()

    pos = np.asarray(pos).reshape(-1, 3)[0]
    quat = np.asarray(quat).reshape(-1, 4)[0]

    return pos, quat


def find_joint_prim_by_name(robot_root, joint_name):
    root = stage.GetPrimAtPath(robot_root)

    if not root.IsValid():
        raise RuntimeError("Robot root not found: " + robot_root)

    for prim in Usd.PrimRange(root):
        if prim.GetName() == joint_name:
            return prim

    return None


def get_current_joint_rad(robot_root, joint_name):
    prim = find_joint_prim_by_name(robot_root, joint_name)

    if prim is None:
        print("[WARN] joint not found:", joint_name)
        return 0.0

    pos_attr = prim.GetAttribute("state:angular:physics:position")

    if pos_attr.IsValid():
        value = pos_attr.Get()

        if value is not None:
            return np.deg2rad(float(value))

    drive = UsdPhysics.DriveAPI.Get(prim, "angular")

    if drive:
        attr = drive.GetTargetPositionAttr()

        if attr.IsValid():
            value = attr.Get()

            if value is not None:
                return np.deg2rad(float(value))

    return 0.0


def set_drive_targets_from_rad(robot_root, joint_names, q_rad):
    for joint_name, q in zip(joint_names, q_rad):
        prim = find_joint_prim_by_name(robot_root, joint_name)

        if prim is None:
            continue

        target_deg = float(np.rad2deg(q))

        drive = UsdPhysics.DriveAPI.Get(prim, "angular")

        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")

        if not drive.GetTargetPositionAttr().IsValid():
            drive.CreateTargetPositionAttr()
        drive.GetTargetPositionAttr().Set(target_deg)

        if not drive.GetStiffnessAttr().IsValid():
            drive.CreateStiffnessAttr()
        drive.GetStiffnessAttr().Set(float(STIFFNESS))

        if not drive.GetDampingAttr().IsValid():
            drive.CreateDampingAttr()
        drive.GetDampingAttr().Set(float(DAMPING))

        if not drive.GetMaxForceAttr().IsValid():
            drive.CreateMaxForceAttr()
        drive.GetMaxForceAttr().Set(float(MAX_FORCE))


def setup_rmp_controller():
    robot = make_robot(
        RIGHT_ROBOT,
        "right_robot_rmp",
    )

    rmpflow = get_ur10_rmpflow()

    base_pos, base_quat = get_base_pose(robot)
    rmpflow.set_robot_base_pose(base_pos, base_quat)

    active_joints = list(rmpflow.get_active_joints())

    q = np.array([
        get_current_joint_rad(RIGHT_ROBOT, joint_name)
        for joint_name in active_joints
    ], dtype=float)

    qd = np.zeros_like(q)

    print("")
    print("=== SETUP RIGHT CONTROLLER ===")
    print("robot root:", RIGHT_ROBOT)
    print("EE path:", RIGHT_EE_PATH)
    print("active joints:", active_joints)
    print("initial joint deg:", np.rad2deg(q))

    return {
        "robot": robot,
        "rmpflow": rmpflow,
        "active_joints": active_joints,
        "home_q": q.copy(),
        "q": q,
        "qd": qd,
        "watched_q": np.array([], dtype=float),
        "watched_qd": np.array([], dtype=float),
    }


def policy_target_from_world(robot, world_target):
    base_pos, _ = get_base_pose(robot)

    return base_pos + (
        np.asarray(world_target, dtype=float) - base_pos
    ) / ROBOT_SCALE_FOR_POLICY


def step_rmp_controller(ctrl, policy_target, target_orientation):
    robot = ctrl["robot"]
    rmpflow = ctrl["rmpflow"]

    base_pos, base_quat = get_base_pose(robot)
    rmpflow.set_robot_base_pose(base_pos, base_quat)

    rmpflow.set_end_effector_target(
        target_position=np.asarray(policy_target, dtype=float),
        target_orientation=normalize_quat(target_orientation),
    )

    q_target, qd_target = rmpflow.compute_joint_targets(
        ctrl["q"],
        ctrl["qd"],
        ctrl["watched_q"],
        ctrl["watched_qd"],
        PHYSICS_DT,
    )

    ctrl["q"] = np.asarray(q_target, dtype=float)
    ctrl["qd"] = np.asarray(qd_target, dtype=float)

    set_drive_targets_from_rad(
        RIGHT_ROBOT,
        ctrl["active_joints"],
        ctrl["q"],
    )


# ============================================================
# 오른팔 이동
# ============================================================

def move_right_ee(
    ctrl,
    start_pos,
    target_pos,
    start_orientation,
    target_orientation,
    frames,
    label,
    convergence_frames=None,
    stable_frames=None,
):
    print("")
    print(f"=== {label} START ===")
    print("start position:", start_pos)
    print("target position:", target_pos)
    print("start orientation wxyz:", start_orientation)
    print("target orientation wxyz:", target_orientation)
    print("frames:", frames)

    convergence_frames = int(
        MOVE_CONVERGENCE_FRAMES
        if convergence_frames is None
        else convergence_frames
    )
    stable_frames = int(
        MOVE_STABLE_FRAMES
        if stable_frames is None
        else stable_frames
    )
    convergence_frames = max(1, convergence_frames)
    stable_frames = max(1, stable_frames)

    policy_start = policy_target_from_world(
        ctrl["robot"],
        start_pos,
    )

    policy_end = policy_target_from_world(
        ctrl["robot"],
        target_pos,
    )

    for frame in range(int(frames) + 1):
        ratio = frame / float(max(int(frames), 1))
        alpha = smoothstep01(ratio)

        policy_now = policy_start + (
            policy_end - policy_start
        ) * alpha

        orientation_now = quat_slerp(
            start_orientation,
            target_orientation,
            alpha,
        )

        step_rmp_controller(
            ctrl,
            policy_now,
            orientation_now,
        )

        if DEBUG_MOTION and (frame % 15 == 0 or frame == int(frames)):
            ee_now = get_world_pos(RIGHT_EE_PATH)
            _dlog(DEBUG_MOTION, f"[{label} FRAME {frame:03d}/{frames}]")
            _dlog(DEBUG_MOTION, "alpha:", alpha)
            _dlog(DEBUG_MOTION, "EE now:", ee_now)
            _dlog(DEBUG_MOTION, "target:", target_pos)
            _dlog(DEBUG_MOTION, "position error:", np.linalg.norm(ee_now - target_pos))

        update_simulation()

    # 프레임 수가 끝났다는 이유만으로 다음 단계로 넘어가지 않는다.
    # 최종 목표를 계속 명령하면서 실제 EE 오차가 충분히 작아졌는지 확인한다.
    stable_count = 0
    reached = False

    for settle_frame in range(convergence_frames):
        step_rmp_controller(
            ctrl,
            policy_end,
            target_orientation,
        )

        update_simulation()

        ee_now = get_world_pos(RIGHT_EE_PATH)
        error = float(np.linalg.norm(ee_now - target_pos))

        if error <= MOVE_POSITION_TOLERANCE:
            stable_count += 1
        else:
            stable_count = 0

        if DEBUG_MOTION and settle_frame % 30 == 0:
            _dlog(DEBUG_MOTION,
                f"[{label} CONVERGENCE {settle_frame:03d}/{convergence_frames}]",
                "EE=", ee_now,
                "error=", round(error, 6),
                "stable=", stable_count,
            )

        if stable_count >= stable_frames:
            reached = True
            break

    final_ee = get_world_pos(RIGHT_EE_PATH)
    final_error = float(np.linalg.norm(final_ee - target_pos))

    print("final EE:", final_ee)
    print("final target error:", final_error)
    print("target reached:", reached)
    print(f"=== {label} DONE ===")

    return reached, final_ee


# ============================================================
# 접촉 기반 최종 접근
# ============================================================

def approach_until_surface_gripped(
    ctrl,
    surface_target,
    target_orientation,
    tcp_offset_local,
):
    """
    Raycast로 추정·정렬된 실제 상면 법선 방향으로 접근한다.

    접근 중에는 두 gripper를 열어 두고, upper/lower 실제 Attachment Ray가
    모두 흡착 가능 거리 안에 들어온 뒤에만 동시에 close한다. 한쪽만 붙으면
    두 컵을 즉시 열고 재시도하여 기울어진 단일 흡착을 방지한다.
    """
    surface_target = np.asarray(surface_target, dtype=float)
    target_orientation = normalize_quat(target_orientation)
    direction = get_ee_suction_axis_world(target_orientation)
    direction /= np.linalg.norm(direction)

    command_right_grippers_open(settle_frames=3)
    gripper_interface = surface_gripper.acquire_surface_gripper_interface()

    start_tcp = get_gripper_tcp_world()
    start_ee, _ = get_world_pose(RIGHT_EE_PATH)

    # surface_target을 지나는 실제 상면과 평행한 안전 offset plane까지
    # 현재 TCP에서 direction 직선으로 이동하는 교점을 Hard Stop으로 사용한다.
    # 이렇게 하면 미세한 프리그라스프 위치 오차가 있어도 불필요한 횡이동이 없다.
    start_plane_coordinate = float(np.dot(
        start_tcp - surface_target,
        direction,
    ))
    max_command_travel = max(
        0.0,
        -HARD_SURFACE_GAP - start_plane_coordinate,
    )
    hard_stop_tcp = start_tcp + direction * max_command_travel

    hard_stop_ee_target = ee_target_from_tcp_target(
        hard_stop_tcp,
        target_orientation,
        tcp_offset_local,
    )
    hard_stop_policy_target = policy_target_from_world(
        ctrl["robot"],
        hard_stop_ee_target,
    )

    commanded_travel = 0.0
    max_frames = max(
        1,
        int(np.ceil(max_command_travel / min(
            CONTACT_APPROACH_STEP,
            DUAL_GRIP_PRESS_STEP,
        )))
        + MOVE_CONVERGENCE_FRAMES
        + RAY_CONTACT_SETTLE_FRAMES
        + 120,
    )

    ray_miss_count = 0
    sync_close_retry_count = 0
    all_grip_stable_count = 0
    attachment_joint_paths = set(get_attachment_joint_paths())
    effective_sync_close_distance = min(
        DUAL_GRIP_SYNC_CLOSE_DISTANCE,
        max(
            0.0,
            SURFACE_MAX_GRIP_DISTANCE
            - RAY_ORIGIN_OFFSET
            - 0.001,
        ),
    )

    print("")
    print("=== RIGHT SYNCHRONIZED TOP DUAL-GRIP APPROACH START ===")
    print("start EE:", start_ee)
    print("start attachment TCP:", start_tcp)
    print("vision top surface:", surface_target)
    print("Raycast-aligned approach direction:", direction)
    print("hard-stop TCP plane point:", hard_stop_tcp)
    print("hard-stop EE target:", hard_stop_ee_target)
    print("max command travel:", max_command_travel)
    print("configured sync close distance:", DUAL_GRIP_SYNC_CLOSE_DISTANCE)
    print("effective sync close distance:", effective_sync_close_distance)
    print("required stable frames:", DUAL_GRIP_STABLE_FRAMES)

    def read_actual_joint_hits(ray_details):
        result = {}
        for item in ray_details:
            joint_name = item.get("joint")
            if joint_name in attachment_joint_paths:
                result[joint_name] = item.get("hit")
        return result

    def command_tcp_target(tcp_target):
        ee_target = ee_target_from_tcp_target(
            tcp_target,
            target_orientation,
            tcp_offset_local,
        )
        step_rmp_controller(
            ctrl,
            policy_target_from_world(ctrl["robot"], ee_target),
            target_orientation,
        )

    def attempt_synchronized_close(hold_policy, reason):
        nonlocal sync_close_retry_count, all_grip_stable_count

        sync_close_retry_count += 1
        all_grip_stable_count = 0

        print("")
        print("=== RIGHT SYNCHRONIZED CLOSE ATTEMPT ===")
        print("reason:", reason)
        print(
            "attempt:",
            sync_close_retry_count,
            "/",
            DUAL_GRIP_SYNC_CLOSE_MAX_RETRIES,
        )

        for settle_frame in range(DUAL_GRIP_SYNC_CLOSE_SETTLE_FRAMES):
            step_rmp_controller(ctrl, hold_policy, target_orientation)

            if settle_frame % GRIP_RETRY_EVERY_FRAMES == 0:
                gripper_now = command_right_grippers_close()
            else:
                gripper_now = gripper_interface

            update_simulation()
            gripped_map = get_gripped_objects_map(gripper_now)
            all_attached = have_all_right_grippers_attached(gripped_map)
            any_attached = has_any_right_gripper_attached(gripped_map)

            if all_attached:
                all_grip_stable_count += 1
            else:
                all_grip_stable_count = 0

            if DEBUG_GRIPPER and (settle_frame == 0 or settle_frame % 5 == 0 or any_attached):
                _dlog(DEBUG_GRIPPER,
                    f"[RIGHT SYNC CLOSE {settle_frame:03d}/"
                    f"{DUAL_GRIP_SYNC_CLOSE_SETTLE_FRAMES}]",
                    "any=", any_attached,
                    "all=", all_attached,
                    "stable=", all_grip_stable_count,
                    "objects=", gripped_map,
                )

            if all_grip_stable_count >= DUAL_GRIP_STABLE_FRAMES:
                final_ee, _ = get_world_pose(RIGHT_EE_PATH)
                final_tcp = get_gripper_tcp_world()
                print("[SUCCESS] BOTH RIGHT grippers attached simultaneously")
                print("final EE:", final_ee)
                print("final TCP:", final_tcp)
                print("gripped:", gripped_map)
                return True, final_ee, final_tcp, gripped_map

        final_map = get_gripped_objects_map(gripper_interface)
        if (
            has_any_right_gripper_attached(final_map)
            and not have_all_right_grippers_attached(final_map)
        ):
            print("[RIGHT SINGLE ATTACH RESET] Only one cup attached.")
            print("Opening both cups before the next synchronized retry.")
            print("objects:", final_map)
            command_right_grippers_open(settle_frames=2)

        return False, None, None, final_map

    for frame in range(max_frames + 1):
        nearest_hit, ray_details = detect_top_surface_with_rays(
            surface_target,
            direction,
        )
        current_ee, current_orientation = get_world_pose(RIGHT_EE_PATH)
        current_tcp = get_gripper_tcp_world()

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

        sync_close_ready = (
            both_joint_rays_valid
            and farthest_joint_distance <= effective_sync_close_distance
        )

        if sync_close_ready:
            hold_policy = policy_target_from_world(ctrl["robot"], current_ee)
            success, final_ee, final_tcp, final_map = attempt_synchronized_close(
                hold_policy,
                reason=(
                    "both attachment rays are within close distance; "
                    f"distances={joint_distances}"
                ),
            )
            if success:
                return True, final_ee, final_tcp, final_map

            if sync_close_retry_count >= DUAL_GRIP_SYNC_CLOSE_MAX_RETRIES:
                print("[FAIL] RIGHT synchronized close retry limit reached.")
                print("joint ray distances:", joint_distances)
                return False, current_ee, current_tcp, final_map

        if remaining_to_hard_stop <= HARD_STOP_TRACKING_TOLERANCE:
            step_rmp_controller(
                ctrl,
                hard_stop_policy_target,
                target_orientation,
            )
            update_simulation()

            if DEBUG_GRIPPER and frame % 5 == 0:
                _dlog(DEBUG_GRIPPER,
                    f"[RIGHT HARD STOP TRACK {frame:03d}/{max_frames}]",
                    "remaining=", round(remaining_to_hard_stop, 6),
                    "joint hits=", joint_distances,
                    "both rays=", both_joint_rays_valid,
                )

            if not both_joint_rays_valid:
                print("[FAIL] RIGHT Hard Stop reached, but both attachment rays do not hit the box.")
                print("This is an XY alignment, cup-spacing, or surface-angle problem.")
                print("actual attachment ray hits:", actual_joint_hits)
                print("surface target:", surface_target)
                print("current TCP:", current_tcp)
                return False, current_ee, current_tcp, get_gripped_objects_map(gripper_interface)

            if farthest_joint_distance > effective_sync_close_distance:
                print("[FAIL] RIGHT Hard Stop reached before both cups entered close distance.")
                print("joint distances:", joint_distances)
                print("Check the actual top collider plane before tuning HARD_SURFACE_GAP.")
                return False, current_ee, current_tcp, get_gripped_objects_map(gripper_interface)

            continue

        if nearest_hit is None:
            ray_miss_count += 1
            if DEBUG_GRIPPER and (frame == 0 or frame % 5 == 0):
                _dlog(DEBUG_GRIPPER, f"[RIGHT RAY MISS {ray_miss_count}/{RAY_MISS_LIMIT_FRAMES}]")
                for item in ray_details:
                    _dlog(DEBUG_GRIPPER, " ray:", item["joint"])
                    _dlog(DEBUG_GRIPPER, " origin:", item["origin"])
                    _dlog(DEBUG_GRIPPER, " all non-right-robot hits:", item["all_hits"][:3])

            if ray_miss_count >= RAY_MISS_LIMIT_FRAMES:
                stop_policy = policy_target_from_world(ctrl["robot"], current_ee)
                for _ in range(CONTACT_HOLD_FRAMES):
                    step_rmp_controller(ctrl, stop_policy, current_orientation)
                    update_simulation()
                print("[FAIL] RIGHT raycast cannot see the requested box top surface.")
                return False, current_ee, current_tcp, get_gripped_objects_map(gripper_interface)
        else:
            ray_miss_count = 0

        center_ray_distance = (
            float(nearest_hit["distance"])
            if nearest_hit is not None
            else float("inf")
        )

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
            remaining_by_ray = CONTACT_APPROACH_STEP

        safe_increment = min(
            CONTACT_APPROACH_STEP,
            remaining_by_ray if remaining_by_ray > 0.0 else DUAL_GRIP_PRESS_STEP,
            max(0.0, remaining_to_hard_stop),
            max(0.0, max_command_travel - commanded_travel),
        )

        if safe_increment > 1.0e-6:
            commanded_travel += safe_increment

        tcp_target = start_tcp + direction * commanded_travel
        command_tcp_target(tcp_target)

        if DEBUG_GRIPPER and frame % 10 == 0:
            _dlog(DEBUG_GRIPPER,
                f"[RIGHT OPEN APPROACH {frame:03d}/{max_frames}]",
                "center_ray=",
                round(center_ray_distance, 6)
                if np.isfinite(center_ray_distance)
                else "NONE",
                "joint_distances=", joint_distances,
                "both_joint_rays=", both_joint_rays_valid,
                "remaining_hard_stop=", round(remaining_to_hard_stop, 6),
                "TCP=", current_tcp,
                "cmd_travel=", round(commanded_travel, 6),
            )

        update_simulation()

    stop_ee, stop_orientation = get_world_pose(RIGHT_EE_PATH)
    stop_tcp = get_gripper_tcp_world()
    stop_policy = policy_target_from_world(ctrl["robot"], stop_ee)
    for _ in range(CONTACT_HOLD_FRAMES):
        step_rmp_controller(ctrl, stop_policy, stop_orientation)
        update_simulation()

    final_map = get_gripped_objects_map(gripper_interface)
    print("[FAIL] RIGHT synchronized dual-grip approach timed out.")
    print("final gripped objects:", final_map)
    return False, stop_ee, stop_tcp, final_map


# ============================================================
# 오른팔 안전 접근: 위쪽 안전점 → 프리그라스프 → 접촉 감지 접근
# ============================================================

def approach_right_box_surface(
    ctrl,
    surface_target,
    initial_orientation,
    target_orientation,
):
    surface_target = np.asarray(surface_target, dtype=float)
    approach = RIGHT_APPROACH_DIRECTION / np.linalg.norm(
        RIGHT_APPROACH_DIRECTION
    )

    tcp_offset_local = get_tcp_offset_in_ee_frame()

    # 초기 접근은 안전한 월드 -Z 자세로 수행하고, 프리그라스프에서 실제 상면
    # 다중 Ray를 측정한 뒤 TCP를 고정한 채 최종 자세를 보정한다.
    pregrasp_tcp = surface_target - approach * PREGRASP_DISTANCE
    pregrasp_ee = ee_target_from_tcp_target(
        pregrasp_tcp,
        target_orientation,
        tcp_offset_local,
    )

    current_ee, current_orientation = get_world_pose(RIGHT_EE_PATH)
    safe_ee = pregrasp_ee.copy()
    safe_ee[2] = max(
        float(current_ee[2]),
        float(pregrasp_ee[2] + SAFE_ABOVE_Z),
    )

    print("")
    print("=== RIGHT SURFACE APPROACH PLAN ===")
    print("surface target:", surface_target)
    print("nominal approach direction:", approach)
    print("pregrasp TCP target:", pregrasp_tcp)
    print("pregrasp EE target:", pregrasp_ee)
    print("safe EE waypoint:", safe_ee)

    safe_move_frames = adaptive_grasp_waypoint_frames(
        current_ee, safe_ee, SAFE_MOVE_FRAMES
    )
    print("adaptive safe move frames:", safe_move_frames)

    safe_reached, _ = move_right_ee(
        ctrl=ctrl,
        start_pos=current_ee,
        target_pos=safe_ee,
        start_orientation=current_orientation,
        target_orientation=target_orientation,
        frames=safe_move_frames,
        label="RIGHT MOVE ABOVE TOP",
        convergence_frames=GRASP_MOVE_CONVERGENCE_FRAMES,
        stable_frames=GRASP_MOVE_STABLE_FRAMES,
    )
    if not safe_reached:
        print("[WARN] safe waypoint was not fully reached; continuing with actual pose.")

    descend_start, descend_orientation = get_world_pose(RIGHT_EE_PATH)
    descend_move_frames = adaptive_grasp_waypoint_frames(
        descend_start, pregrasp_ee, DESCEND_FRAMES
    )
    print("adaptive pregrasp descend frames:", descend_move_frames)

    pregrasp_reached, _ = move_right_ee(
        ctrl=ctrl,
        start_pos=descend_start,
        target_pos=pregrasp_ee,
        start_orientation=descend_orientation,
        target_orientation=target_orientation,
        frames=descend_move_frames,
        label="RIGHT DESCEND TO TOP PREGRASP",
        convergence_frames=GRASP_MOVE_CONVERGENCE_FRAMES,
        stable_frames=GRASP_MOVE_STABLE_FRAMES,
    )

    actual_pregrasp_tcp = get_gripper_tcp_world()
    _, actual_pregrasp_orientation = get_world_pose(RIGHT_EE_PATH)
    xy_error = float(np.linalg.norm(
        actual_pregrasp_tcp[:2] - surface_target[:2]
    ))
    z_gap = float(actual_pregrasp_tcp[2] - surface_target[2])
    level_error_deg = get_ee_level_error_deg(actual_pregrasp_orientation)

    print("actual pregrasp TCP:", actual_pregrasp_tcp)
    print("top-surface XY error:", xy_error)
    print("top-surface vertical gap:", z_gap)
    print("EE local X world:", get_ee_suction_axis_world(actual_pregrasp_orientation))
    print("nominal world-down error deg:", level_error_deg)
    print("pregrasp reached:", pregrasp_reached)

    if not pregrasp_reached:
        print("[FAIL] RIGHT pregrasp target did not converge; Raycast contact approach will not start.")
        return False, get_world_pos(RIGHT_EE_PATH), actual_pregrasp_tcp, target_orientation

    if level_error_deg > EE_LEVEL_ABORT_TOLERANCE_DEG:
        print("[FAIL] EE suction normal is not aligned with nominal world -Z before Ray alignment.")
        print("[FAIL] Contact descent cancelled.")
        return False, get_world_pos(RIGHT_EE_PATH), actual_pregrasp_tcp, target_orientation

    if xy_error > 0.03:
        print("[FAIL] TCP is not aligned over the top surface.")
        print("[FAIL] Contact descent cancelled to avoid hitting the box edge.")
        return False, get_world_pos(RIGHT_EE_PATH), actual_pregrasp_tcp, target_orientation

    target_orientation, top_aligned, plane_verified = (
        auto_align_right_tool_to_top_plane(
            ctrl=ctrl,
            surface_target=surface_target,
            current_target_orientation=target_orientation,
            tcp_offset_local=tcp_offset_local,
        )
    )
    print("top-plane auto aligned:", top_aligned)
    print("top-plane verified:", plane_verified)

    if not plane_verified:
        print("[FAIL] RIGHT top surface could not be verified/aligned safely.")
        return False, get_world_pos(RIGHT_EE_PATH), get_gripper_tcp_world(), target_orientation

    # 자세 정렬 후 실제 법선축 위의 프리그라스프 위치로 TCP를 재배치한다.
    # 이 단계가 없으면 기존 월드 -Z 프리그라스프 위치에서 기울어진 법선으로
    # 내려가면서 비전 중심점에서 횡방향으로 벗어날 수 있다.
    requested_normal = get_ee_suction_axis_world(target_orientation)
    requested_normal /= np.linalg.norm(requested_normal)
    aligned_pregrasp_tcp = surface_target - requested_normal * PREGRASP_DISTANCE
    aligned_pregrasp_ee = ee_target_from_tcp_target(
        aligned_pregrasp_tcp,
        target_orientation,
        tcp_offset_local,
    )
    reposition_start, reposition_orientation = get_world_pose(RIGHT_EE_PATH)
    aligned_move_frames = adaptive_grasp_waypoint_frames(
        reposition_start, aligned_pregrasp_ee, TOP_ALIGNED_PREGRASP_MOVE_FRAMES
    )
    print("adaptive aligned pregrasp frames:", aligned_move_frames)

    aligned_pregrasp_reached, _ = move_right_ee(
        ctrl=ctrl,
        start_pos=reposition_start,
        target_pos=aligned_pregrasp_ee,
        start_orientation=reposition_orientation,
        target_orientation=target_orientation,
        frames=aligned_move_frames,
        label="RIGHT MOVE TO RAY-ALIGNED PREGRASP",
        convergence_frames=GRASP_MOVE_CONVERGENCE_FRAMES,
        stable_frames=GRASP_MOVE_STABLE_FRAMES,
    )

    aligned_actual_tcp = get_gripper_tcp_world()
    _, aligned_actual_orientation = get_world_pose(RIGHT_EE_PATH)
    actual_normal = get_ee_suction_axis_world(aligned_actual_orientation)
    aligned_normal_error = angle_between_vectors_deg(actual_normal, requested_normal)
    aligned_delta = aligned_actual_tcp - surface_target
    aligned_lateral = aligned_delta - requested_normal * float(np.dot(
        aligned_delta,
        requested_normal,
    ))
    aligned_lateral_error = float(np.linalg.norm(aligned_lateral))
    aligned_normal_gap = -float(np.dot(aligned_delta, requested_normal))

    print("requested aligned suction normal:", requested_normal)
    print("actual aligned suction normal:", actual_normal)
    print("post-align normal tracking error deg:", aligned_normal_error)
    print("aligned pregrasp reached:", aligned_pregrasp_reached)
    print("aligned pregrasp TCP:", aligned_actual_tcp)
    print("aligned lateral error:", aligned_lateral_error)
    print("aligned normal gap:", aligned_normal_gap)

    if not aligned_pregrasp_reached:
        print("[FAIL] RIGHT Ray-aligned pregrasp target did not converge.")
        return False, get_world_pos(RIGHT_EE_PATH), aligned_actual_tcp, target_orientation

    if aligned_normal_error > EE_LEVEL_ABORT_TOLERANCE_DEG:
        print("[FAIL] RIGHT tool did not converge to the Raycast-estimated top normal.")
        return False, get_world_pos(RIGHT_EE_PATH), aligned_actual_tcp, target_orientation

    if aligned_lateral_error > 0.03:
        print("[FAIL] RIGHT TCP did not reach the Ray-aligned pregrasp axis.")
        return False, get_world_pos(RIGHT_EE_PATH), aligned_actual_tcp, target_orientation

    grip_ok, final_ee, final_tcp, gripped_map = (
        approach_until_surface_gripped(
            ctrl=ctrl,
            surface_target=surface_target,
            target_orientation=target_orientation,
            tcp_offset_local=tcp_offset_local,
        )
    )

    print("")
    print("=== RIGHT SURFACE APPROACH RESULT ===")
    print("grip ok:", grip_ok)
    print("final EE:", final_ee)
    print("final TCP:", final_tcp)
    print("surface target:", surface_target)
    print("final aligned orientation:", target_orientation)
    print("gripped objects:", gripped_map)

    return grip_ok, final_ee, final_tcp, target_orientation


# ============================================================
# Surface Gripper
# ============================================================

def close_right_grippers():
    sg = surface_gripper.acquire_surface_gripper_interface()

    print("")
    print("=== RIGHT GRIPPERS CLOSE ===")

    for gripper_path in RIGHT_GRIPPERS:
        print("close:", gripper_path)

        try:
            sg.close_gripper(gripper_path)
        except Exception as exc:
            print("[ERROR] close_gripper failed:", repr(exc))

    wait_frames(GRIP_SETTLE_FRAMES)

    print("")
    print("=== RIGHT GRIPPERS OBJECT CHECK ===")

    all_ok = True

    for gripper_path in RIGHT_GRIPPERS:
        try:
            objects = sg.get_gripped_objects(gripper_path)
            print("objects:", gripper_path, objects)

            if len(objects) == 0:
                all_ok = False

        except Exception as exc:
            print("[ERROR] get_gripped_objects failed:", repr(exc))
            all_ok = False

    print("right upper/lower grip ok:", all_ok)

    return all_ok


def open_right_grippers(settle_frames=RELEASE_SETTLE_FRAMES):
    """오른팔 그리퍼를 열고 지정한 프레임만큼 해제 안정화를 기다린다.

    settle_frames=0이면 open 명령만 즉시 내리고 반환한다. 다음 팔 시작 신호를
    그리퍼 open 직후 발행해야 하는 파이프라인 인계 경로에서 사용한다.
    """
    sg = surface_gripper.acquire_surface_gripper_interface()

    print("")
    print("=== RIGHT GRIPPERS OPEN ===")

    for gripper_path in RIGHT_GRIPPERS:
        print("open:", gripper_path)

        try:
            sg.open_gripper(gripper_path)
        except Exception as exc:
            print("[ERROR] open_gripper failed:", repr(exc))

    settle_frames = max(0, int(settle_frames))
    if settle_frames > 0:
        wait_frames(settle_frames)

    print(
        "=== RIGHT GRIPPERS OPEN COMMAND SENT ==="
        if settle_frames == 0
        else "=== RIGHT GRIPPERS OPEN DONE ==="
    )


# ============================================================
# Lift
# ============================================================

def lift_right_robot(ctrl, orientation):
    start_pos, start_orientation = get_world_pose(RIGHT_EE_PATH)

    target_pos = start_pos.copy()
    target_pos[2] += LIFT_HEIGHT

    move_right_ee(
        ctrl=ctrl,
        start_pos=start_pos,
        target_pos=target_pos,
        start_orientation=start_orientation,
        target_orientation=orientation,
        frames=LIFT_FRAMES,
        label="RIGHT LIFT",
    )

    return start_pos, target_pos


# ============================================================
# EE orientation rotate
# ============================================================

def hold_and_check_left_barcode(
    ctrl,
    fixed_position,
    hold_orientation,
    barcode_node,
    hold_index,
):
    """
    현재 자세를 정확히 1초 유지한다.

    이 검사 창이 시작된 뒤 /barcode_exist_L data == 1이 중간의 0 없이
    0.5초 이상 연속 유지돼야 검출로 인정한다. 검출되더라도
    1초 정지는 끝까지 수행한다.
    """
    fixed_position = np.asarray(fixed_position, dtype=float)
    hold_orientation = normalize_quat(hold_orientation)
    policy_target = policy_target_from_world(
        ctrl["robot"],
        fixed_position,
    )

    barcode_node.begin_left_inspection_window()
    detected = False

    print("")
    print(
        f"=== LEFT BARCODE HOLD {hold_index}: "
        f"{EE_ROTATE_PAUSE_SECONDS:.1f}s START ==="
    )
    print("topic:", LEFT_BARCODE_TOPIC)
    print("required value:", BARCODE_TRIGGER_VALUE)
    print("required continuous seconds:", LEFT_BARCODE_STABLE_SECONDS)

    for pause_frame in range(EE_ROTATE_PAUSE_FRAMES):
        step_rmp_controller(
            ctrl,
            policy_target,
            hold_orientation,
        )
        update_simulation()

        if barcode_node.is_left_barcode_one_stable():
            detected = True

        if (
            pause_frame == 0
            or (pause_frame + 1) % 30 == 0
            or pause_frame == EE_ROTATE_PAUSE_FRAMES - 1
        ):
            ee_now = get_world_pos(RIGHT_EE_PATH)
            print(
                f"[LEFT BARCODE HOLD {hold_index} "
                f"{pause_frame + 1:03d}/"
                f"{EE_ROTATE_PAUSE_FRAMES}]"
            )
            print("EE now:", ee_now)
            print(
                "fixed position error:",
                np.linalg.norm(ee_now - fixed_position),
            )
            print(
                "left barcode continuous-1 elapsed:",
                round(barcode_node.get_left_barcode_one_elapsed(), 3),
            )
            print("left barcode detected:", detected)

    print(
        f"=== LEFT BARCODE HOLD {hold_index} DONE; "
        f"detected={detected} ==="
    )
    return detected

def rotate_right_to_cumulative_angle(
    ctrl,
    fixed_position,
    start_orientation,
    start_total_deg,
    target_total_deg,
    label,
):
    """시작 자세 기준 누적각 start_total_deg -> target_total_deg로 회전한다."""
    fixed_position = np.asarray(fixed_position, dtype=float)
    start_orientation = normalize_quat(start_orientation)
    policy_target = policy_target_from_world(
        ctrl["robot"],
        fixed_position,
    )

    final_orientation = start_orientation.copy()

    print("")
    print(f"=== {label} START ===")
    print("fixed position:", fixed_position)
    print("start cumulative degree:", start_total_deg)
    print("target cumulative degree:", target_total_deg)

    for frame in range(EE_ROTATE_STEP_FRAMES + 1):
        ratio = frame / float(max(EE_ROTATE_STEP_FRAMES, 1))
        alpha = smoothstep01(ratio)

        cumulative_deg = (
            float(start_total_deg)
            + (
                float(target_total_deg)
                - float(start_total_deg)
            ) * alpha
        )
        signed_cumulative_deg = RIGHT_ROTATE_SIGN * cumulative_deg

        q_delta = quat_from_axis_angle(
            [1.0, 0.0, 0.0],
            signed_cumulative_deg,
        )
        final_orientation = quat_mul(
            start_orientation,
            q_delta,
        )

        step_rmp_controller(
            ctrl,
            policy_target,
            final_orientation,
        )

        if frame % 15 == 0 or frame == EE_ROTATE_STEP_FRAMES:
            ee_now = get_world_pos(RIGHT_EE_PATH)
            print(
                f"[{label} FRAME {frame:03d}/"
                f"{EE_ROTATE_STEP_FRAMES}]"
            )
            print("signed cumulative degree:", signed_cumulative_deg)
            print("EE now:", ee_now)
            print(
                "fixed position error:",
                np.linalg.norm(ee_now - fixed_position),
            )

        update_simulation()

    print(f"=== {label} DONE ===")
    return final_orientation



def rotate_until_left_barcode_then_extra_90(
    ctrl,
    fixed_position,
    start_orientation,
    barcode_node,
):
    """
    수정된 오른팔 회전 검사 로직.

    변경점:
        - 첫 90도 회전 전에 /barcode_exist_L을 먼저 1초 검사한다.
        - 첫 90도 전에 /barcode_exist_L == 1이 확인되면
          오른팔은 90도만 회전하고 종료한다.
        - 이 경우 기존처럼 90도 회전 후 다시 검사해서 추가 90도 돌지 않는다.

    기존 로직:
        - 첫 90도 전 L 검출이 없으면
          90도 회전 -> 1초 검사 -> 검출 시 추가 90도
          기존 방식 그대로 수행한다.

    반환:
        final_orientation, final_cumulative_deg, left_extra_rotation_performed

        left_extra_rotation_performed=True:
            오른팔이 L 검출 기반으로 90도 완료 경로에 들어갔다는 뜻.
            이후 기존 코드에서 /right_extra_done 및 왼팔 90도 모드로 연결됨.
    """
    fixed_position = np.asarray(fixed_position, dtype=float)
    start_orientation = normalize_quat(start_orientation)

    print("")
    print("=================================================")
    print("=== RIGHT ROTATE / LEFT BARCODE INSPECTION START ===")
    print("=================================================")
    print("fixed position:", fixed_position)
    print("rotation step degree:", EE_ROTATE_STEP_DEG)
    print("configured inspection rotations:", EE_ROTATE_STEPS)
    print("pause seconds:", EE_ROTATE_PAUSE_SECONDS)
    print("left barcode topic:", LEFT_BARCODE_TOPIC)

    current_total_deg = 0.0
    final_orientation = start_orientation.copy()
    hold_index = 0

    # ============================================================
    # 추가된 부분:
    # 첫 90도 회전 전에 /barcode_exist_L을 먼저 검사한다.
    # 이때 L=1이 확인되면 오른팔은 90도만 돌고 바로 종료한다.
    # ============================================================
    print("")
    print("=================================================")
    print("=== PRE-FIRST-ROTATION LEFT BARCODE CHECK START ===")
    print("If left barcode is already visible before first 90 deg,")
    print("right arm will rotate only 90 deg and finish.")
    print("=================================================")

    pre_detected = hold_and_check_left_barcode(
        ctrl=ctrl,
        fixed_position=fixed_position,
        hold_orientation=start_orientation,
        barcode_node=barcode_node,
        hold_index=0,
    )

    if pre_detected:
        first_total_deg = EE_ROTATE_STEP_DEG

        print("")
        print("[LEFT BARCODE PRECHECK] data == 1 confirmed before first rotation.")
        print("[LEFT BARCODE PRECHECK] Performing only one 90-degree rotation.")
        print("[LEFT BARCODE PRECHECK] No additional 90-degree rotation will be executed.")

        final_orientation = rotate_right_to_cumulative_angle(
            ctrl=ctrl,
            fixed_position=fixed_position,
            start_orientation=start_orientation,
            start_total_deg=0.0,
            target_total_deg=first_total_deg,
            label="RIGHT ONLY 90 AFTER PRECHECK LEFT BARCODE",
        )

        print(
            "final cumulative degree:",
            RIGHT_ROTATE_SIGN * first_total_deg,
        )

        return final_orientation, first_total_deg, True

    print("")
    print("[LEFT BARCODE PRECHECK] No confirmed barcode before first rotation.")
    print("[LEFT BARCODE PRECHECK] Continue normal 90-degree inspection loop.")

    # ============================================================
    # 기존 로직:
    # 90도 회전 -> 1초 검사 반복
    # 검사 중 L=1이면 추가 90도 한 번 더 회전 후 종료
    # ============================================================
    for step_index in range(1, EE_ROTATE_STEPS + 1):
        next_total_deg = current_total_deg + EE_ROTATE_STEP_DEG

        final_orientation = rotate_right_to_cumulative_angle(
            ctrl=ctrl,
            fixed_position=fixed_position,
            start_orientation=start_orientation,
            start_total_deg=current_total_deg,
            target_total_deg=next_total_deg,
            label=(
                f"RIGHT INSPECTION ROTATE "
                f"{step_index}/{EE_ROTATE_STEPS}"
            ),
        )

        current_total_deg = next_total_deg
        hold_index += 1

        detected = hold_and_check_left_barcode(
            ctrl=ctrl,
            fixed_position=fixed_position,
            hold_orientation=final_orientation,
            barcode_node=barcode_node,
            hold_index=hold_index,
        )

        if detected:
            extra_total_deg = current_total_deg + EE_ROTATE_STEP_DEG

            print("")
            print("[LEFT BARCODE] data == 1 confirmed during 1-second hold.")
            print(
                "[LEFT BARCODE] Performing exactly one additional "
                "90-degree rotation before replacement."
            )

            final_orientation = rotate_right_to_cumulative_angle(
                ctrl=ctrl,
                fixed_position=fixed_position,
                start_orientation=start_orientation,
                start_total_deg=current_total_deg,
                target_total_deg=extra_total_deg,
                label="RIGHT EXTRA 90 AFTER LEFT BARCODE",
            )

            print(
                "final cumulative degree:",
                RIGHT_ROTATE_SIGN * extra_total_deg,
            )

            return final_orientation, extra_total_deg, True

    # 세 번의 90도 회전이 끝나 누적 270도에 도달하면 신호를 더 기다리지 않는다.
    # 현재 270도 자세를 그대로 유지한 채 원래 파지 위치 배치 단계로 넘어간다.
    print("")
    print("======================================================")
    print("=== 270 DEG REACHED: REPLACE AT ORIGINAL GRASP POSE ===")
    print("======================================================")
    print(
        "final cumulative degree:",
        RIGHT_ROTATE_SIGN * current_total_deg,
    )
    print(
        "No fixed-destination transfer will be executed for this 270-degree path."
    )

    return final_orientation, current_total_deg, False

# ============================================================
# Place / Retreat
# ============================================================

def place_right_robot_fixed_destination(
    ctrl,
    place_orientation,
    grasp_ee_z,
):
    """
    /barcode_exist_R == 1일 때 사용하는 고정 목적지 배치.

    최종 EE 좌표:
      X = FIXED_PLACE_EE_XY[0]
      Y = FIXED_PLACE_EE_XY[1]
      Z = 실제 파지 순간 ee_link Z + PLACE_Z_ABOVE_GRASP

    이동 순서:
      1. 상승 높이를 유지하며 고정 목적지 X/Y 상공으로 수평 이동
      2. 계산된 목적지 Z까지 수직 하강
      3. 실제 최종 EE 위치 반환
    """
    grasp_ee_z = float(grasp_ee_z)

    final_target = np.array([
        float(FIXED_PLACE_EE_XY[0]),
        float(FIXED_PLACE_EE_XY[1]),
        grasp_ee_z + PLACE_Z_ABOVE_GRASP,
    ], dtype=float)

    current_pos, current_orientation = get_world_pose(RIGHT_EE_PATH)

    above_target = np.array([
        final_target[0],
        final_target[1],
        max(float(current_pos[2]), float(final_target[2])),
    ], dtype=float)

    print("")
    print("=== RIGHT DIRECT FIXED-DESTINATION PLACE PLAN ===")
    print("trigger topic:", RIGHT_BARCODE_TOPIC)
    print("current EE:", current_pos)
    print("grasp EE Z:", grasp_ee_z)
    print("fixed destination XY:", FIXED_PLACE_EE_XY)
    print("final destination EE:", final_target)
    print("transfer waypoint:", above_target)
    print("place orientation wxyz:", place_orientation)

    transfer_reached, _ = move_right_ee(
        ctrl=ctrl,
        start_pos=current_pos,
        target_pos=above_target,
        start_orientation=current_orientation,
        target_orientation=place_orientation,
        frames=PLACE_TRANSFER_FRAMES,
        label="RIGHT DIRECT TRANSFER TO FIXED DESTINATION",
    )

    if not transfer_reached:
        print("[WARN] Fixed-destination transfer waypoint was not fully reached.")

    descend_start, descend_orientation = get_world_pose(RIGHT_EE_PATH)

    place_reached, final_ee = move_right_ee(
        ctrl=ctrl,
        start_pos=descend_start,
        target_pos=final_target,
        start_orientation=descend_orientation,
        target_orientation=place_orientation,
        frames=PLACE_DESCEND_FRAMES,
        label="RIGHT DIRECT DESCEND AT FIXED DESTINATION",
    )

    print("fixed destination reached:", place_reached)
    print("requested fixed destination EE:", final_target)
    print("actual final EE:", final_ee)
    print(
        "fixed destination error:",
        np.linalg.norm(final_ee - final_target),
    )

    return final_ee


def place_right_robot_on_original_grasp_axis(
    ctrl,
    place_orientation,
    original_grasp_ee_position,
):
    """
    원래 파지했던 EE X/Y 수직축으로 돌아가 파지 당시 EE Z에 내려놓는다.

    이동 순서:
      1. 현재 상승 높이를 유지한 채 original_grasp_ee_position의 X/Y로 이동
      2. X/Y를 고정하고 original_grasp_ee_position의 Z까지 수직 하강
      3. 실제 최종 EE 위치를 반환
    """
    original_grasp_ee_position = np.asarray(
        original_grasp_ee_position,
        dtype=float,
    ).copy()

    current_pos, current_orientation = get_world_pose(RIGHT_EE_PATH)

    above_original_axis = np.array([
        float(original_grasp_ee_position[0]),
        float(original_grasp_ee_position[1]),
        max(
            float(current_pos[2]),
            float(original_grasp_ee_position[2]),
        ),
    ], dtype=float)

    print("")
    print("=== RIGHT ORIGINAL-GRASP-AXIS PLACE PLAN ===")
    print("current EE:", current_pos)
    print("original grasp EE:", original_grasp_ee_position)
    print("horizontal waypoint:", above_original_axis)
    print("final replacement target:", original_grasp_ee_position)
    print("place orientation wxyz:", place_orientation)

    transfer_reached, _ = move_right_ee(
        ctrl=ctrl,
        start_pos=current_pos,
        target_pos=above_original_axis,
        start_orientation=current_orientation,
        target_orientation=place_orientation,
        frames=PLACE_TRANSFER_FRAMES,
        label="RIGHT RETURN ABOVE ORIGINAL GRASP AXIS",
    )

    if not transfer_reached:
        print(
            "[WARN] Original grasp-axis horizontal waypoint "
            "was not fully reached."
        )

    descend_start, descend_orientation = get_world_pose(RIGHT_EE_PATH)

    place_reached, final_ee = move_right_ee(
        ctrl=ctrl,
        start_pos=descend_start,
        target_pos=original_grasp_ee_position,
        start_orientation=descend_orientation,
        target_orientation=place_orientation,
        frames=PLACE_DESCEND_FRAMES,
        label="RIGHT DESCEND TO ORIGINAL GRASP Z",
    )

    print("original-axis place reached:", place_reached)
    print("requested original grasp EE:", original_grasp_ee_position)
    print("actual final EE:", final_ee)
    print(
        "replacement position error:",
        np.linalg.norm(final_ee - original_grasp_ee_position),
    )

    return final_ee


def retreat_right_robot(
    ctrl,
    start_pos,
    retreat_orientation,
):
    current_pos, current_orientation = get_world_pose(RIGHT_EE_PATH)

    # 윗면에서 놓은 뒤에는 X/Y를 유지하고 수직 위로 빠진다.
    target_pos = current_pos.copy()
    target_pos[2] += RETREAT_Z

    move_right_ee(
        ctrl=ctrl,
        start_pos=current_pos,
        target_pos=target_pos,
        start_orientation=current_orientation,
        target_orientation=retreat_orientation,
        frames=RETREAT_FRAMES,
        label="RIGHT RETREAT UP",
    )


def hold_after_release_and_publish_right_done(
    ctrl,
    node,
    hold_orientation,
    publish_extra_done,
):
    """
    박스를 놓은 뒤 현재 EE 위치/자세를 1초간 유지하고 완료 신호를 발행한다.

    publish_extra_done=True:
        /barcode_exist_L == 1로 추가 90도 회전한 경로이므로
        /right_extra_done에 Int32(data=1)을 한 번 발행한다.

    publish_extra_done=False:
        일반 270도 종료 또는 /barcode_exist_R 직행 경로이므로
        /right_phase_done에 Int32(data=1)을 한 번 발행한다.
    """
    hold_position, actual_orientation = get_world_pose(RIGHT_EE_PATH)
    hold_orientation = normalize_quat(hold_orientation)

    publish_topic = (
        RIGHT_EXTRA_DONE_TOPIC
        if publish_extra_done
        else RIGHT_PHASE_DONE_TOPIC
    )

    hold_policy = policy_target_from_world(
        ctrl["robot"],
        hold_position,
    )

    print("")
    print("================================================")
    print("=== HOLD AFTER RELEASE / ROUTE DONE PUBLISH ===")
    print("================================================")
    print("hold EE position:", hold_position)
    print("actual orientation wxyz:", actual_orientation)
    print("hold orientation wxyz:", hold_orientation)
    print("delay seconds:", RIGHT_PHASE_DONE_DELAY_SECONDS)
    print("delay frames:", RIGHT_PHASE_DONE_DELAY_FRAMES)
    print("publish topic:", publish_topic)
    print("publish data: 1")

    for frame in range(RIGHT_PHASE_DONE_DELAY_FRAMES):
        step_rmp_controller(
            ctrl,
            hold_policy,
            hold_orientation,
        )

        update_simulation()

        if (
            frame == 0
            or (frame + 1) % 30 == 0
            or frame == RIGHT_PHASE_DONE_DELAY_FRAMES - 1
        ):
            ee_now = get_world_pos(RIGHT_EE_PATH)
            print(
                f"[POST RELEASE HOLD "
                f"{frame + 1:03d}/{RIGHT_PHASE_DONE_DELAY_FRAMES}]"
            )
            print(
                "hold position error:",
                np.linalg.norm(ee_now - hold_position),
            )

    if publish_extra_done:
        node.publish_right_extra_done()
    else:
        node.publish_right_phase_done()

    # publish 호출 직후 DDS가 메시지를 처리할 수 있도록 한 프레임 갱신한다.
    step_rmp_controller(
        ctrl,
        hold_policy,
        hold_orientation,
    )
    update_simulation()

    print(
        "=== RIGHT EXTRA DONE PUBLISHED ==="
        if publish_extra_done
        else "=== RIGHT PHASE DONE PUBLISHED ==="
    )


# ============================================================
# 초기 자세 복귀
# ============================================================

def get_robot_joint_positions(ctrl):
    """현재 USD joint 상태를 RMPflow active joint 순서로 읽는다."""
    return np.array([
        get_current_joint_rad(RIGHT_ROBOT, joint_name)
        for joint_name in ctrl["active_joints"]
    ], dtype=float)


def wrapped_joint_delta(target_q, current_q):
    """각 관절을 가장 짧은 회전 경로로 이동시키는 차이를 계산한다."""
    target_q = np.asarray(target_q, dtype=float)
    current_q = np.asarray(current_q, dtype=float)
    return np.arctan2(
        np.sin(target_q - current_q),
        np.cos(target_q - current_q),
    )


def initialize_level_home_pose(ctrl):
    """
    프로그램 시작 시 현재 EE 위치를 유지하고 흡착면만 지면과 평행하게 보정한다.

    보정 완료 후의 실제 관절각과 EE 자세를 새로운 Home으로 저장한다.
    이후 작업 복귀와 다음 박스 대기는 이 수평 Home을 기준으로 수행한다.
    """
    start_pos, start_orientation = get_world_pose(RIGHT_EE_PATH)
    target_orientation = make_level_top_grasp_orientation(start_orientation)
    policy_target = policy_target_from_world(ctrl["robot"], start_pos)

    print("")
    print("=================================================")
    print("=== INITIAL LEVEL HOME ALIGNMENT START ===")
    print("=================================================")
    print("fixed initial EE position:", start_pos)
    print("start orientation wxyz:", start_orientation)
    print("target level orientation wxyz:", target_orientation)
    print("start suction axis world:", get_ee_suction_axis_world(start_orientation))
    print("target suction axis world:", get_ee_suction_axis_world(target_orientation))
    print("start level error deg:", get_ee_level_error_deg(start_orientation))
    print("target level error deg:", get_ee_level_error_deg(target_orientation))

    # 현재 위치를 고정한 채 자세만 수평 자세로 보간한다.
    for frame in range(INITIAL_LEVEL_FRAMES + 1):
        ratio = frame / float(max(INITIAL_LEVEL_FRAMES, 1))
        alpha = smoothstep01(ratio)
        orientation_now = quat_slerp(
            start_orientation,
            target_orientation,
            alpha,
        )

        step_rmp_controller(
            ctrl,
            policy_target,
            orientation_now,
        )

        if frame % 20 == 0 or frame == INITIAL_LEVEL_FRAMES:
            actual_pos, actual_orientation = get_world_pose(RIGHT_EE_PATH)
            print(
                f"[INITIAL LEVEL FRAME {frame:03d}/{INITIAL_LEVEL_FRAMES}]"
            )
            print("EE position:", actual_pos)
            print("position error:", np.linalg.norm(actual_pos - start_pos))
            print(
                "suction axis world:",
                get_ee_suction_axis_world(actual_orientation),
            )
            print(
                "level error deg:",
                get_ee_level_error_deg(actual_orientation),
            )
            print(
                "orientation target error deg:",
                quat_angle_error_deg(actual_orientation, target_orientation),
            )

        update_simulation()

    # 위치와 수평 오차가 모두 안정될 때까지 목표를 유지한다.
    stable_count = 0
    reached = False

    for settle_frame in range(INITIAL_LEVEL_CONVERGENCE_FRAMES):
        step_rmp_controller(
            ctrl,
            policy_target,
            target_orientation,
        )
        update_simulation()

        actual_pos, actual_orientation = get_world_pose(RIGHT_EE_PATH)
        position_error = float(np.linalg.norm(actual_pos - start_pos))
        level_error = get_ee_level_error_deg(actual_orientation)

        if (
            position_error <= INITIAL_LEVEL_POSITION_TOLERANCE
            and level_error <= INITIAL_LEVEL_TOLERANCE_DEG
        ):
            stable_count += 1
        else:
            stable_count = 0

        if settle_frame % 30 == 0:
            print(
                f"[INITIAL LEVEL CONVERGENCE "
                f"{settle_frame:03d}/{INITIAL_LEVEL_CONVERGENCE_FRAMES}]"
            )
            print("position error:", position_error)
            print("level error deg:", level_error)
            print("stable frames:", stable_count)

        if stable_count >= INITIAL_LEVEL_STABLE_FRAMES:
            reached = True
            break

    home_pos, home_orientation = get_world_pose(RIGHT_EE_PATH)
    home_q = get_robot_joint_positions(ctrl)
    final_position_error = float(np.linalg.norm(home_pos - start_pos))
    final_level_error = get_ee_level_error_deg(home_orientation)

    print("")
    print("=== INITIAL LEVEL HOME RESULT ===")
    print("home reached:", reached)
    print("home EE position:", home_pos)
    print("home EE orientation wxyz:", home_orientation)
    print("home joint deg:", np.rad2deg(home_q))
    print("home suction axis world:", get_ee_suction_axis_world(home_orientation))
    print("home level error deg:", final_level_error)
    print("home position shift:", final_position_error)

    if not reached:
        raise RuntimeError(
            "Failed to establish level initial Home pose. "
            f"position_error={final_position_error:.6f} m, "
            f"level_error={final_level_error:.3f} deg"
        )

    # 이후 RMPflow 내부 상태와 복귀 기준을 실제 수평 Home 관절 상태에 맞춘다.
    ctrl["home_q"] = home_q.copy()
    ctrl["q"] = home_q.copy()
    ctrl["qd"] = np.zeros_like(home_q)

    print("================================================")
    print("=== INITIAL LEVEL HOME ALIGNMENT DONE ===")
    print("================================================")

    return home_pos, home_orientation, home_q


def move_right_joints_home(ctrl, home_q):
    start_q = get_robot_joint_positions(ctrl)
    home_q = np.asarray(home_q, dtype=float).copy()

    delta_q = wrapped_joint_delta(home_q, start_q)
    commanded_home_q = start_q + delta_q

    max_delta_deg = float(
        np.max(np.abs(np.rad2deg(delta_q)))
    )

    adaptive_frames = int(np.ceil(
        max_delta_deg
        / HOME_MAX_JOINT_SPEED_DEG_PER_SEC
        / PHYSICS_DT
    ))

    home_move_frames = max(
        HOME_JOINT_FRAMES,
        adaptive_frames,
    )

    print("")
    print("=== RIGHT RETURN HOME JOINT START ===")
    print("current joint deg:", np.rad2deg(start_q))
    print("saved home joint deg:", np.rad2deg(home_q))
    print("shortest delta deg:", np.rad2deg(delta_q))
    print("maximum joint movement deg:", max_delta_deg)
    print("adaptive home frames:", home_move_frames)

    for frame in range(home_move_frames + 1):
        ratio = frame / float(max(home_move_frames, 1))
        alpha = smoothstep01(ratio)

        q_now = start_q + delta_q * alpha

        set_drive_targets_from_rad(
            RIGHT_ROBOT,
            ctrl["active_joints"],
            q_now,
        )

        ctrl["q"] = q_now.copy()
        ctrl["qd"] = np.zeros_like(q_now)

        update_simulation()

        if frame % 30 == 0 or frame == home_move_frames:
            actual_q = get_robot_joint_positions(ctrl)
            error_q = wrapped_joint_delta(
                commanded_home_q,
                actual_q,
            )

            _dlog(DEBUG_JOINT,
                f"[RIGHT HOME FRAME {frame:03d}/{home_move_frames}]"
            )
            _dlog(DEBUG_JOINT, "command joint deg:", np.rad2deg(q_now))
            _dlog(DEBUG_JOINT, "actual joint deg:", np.rad2deg(actual_q))
            _dlog(DEBUG_JOINT, "max error deg:", np.max(np.abs(np.rad2deg(error_q))))

    reached = False
    stable_count = 0

    for settle_frame in range(HOME_SETTLE_FRAMES):
        set_drive_targets_from_rad(
            RIGHT_ROBOT,
            ctrl["active_joints"],
            commanded_home_q,
        )

        ctrl["q"] = commanded_home_q.copy()
        ctrl["qd"] = np.zeros_like(commanded_home_q)

        update_simulation()

        actual_q = get_robot_joint_positions(ctrl)
        error_deg = np.abs(
            np.rad2deg(
                wrapped_joint_delta(
                    commanded_home_q,
                    actual_q,
                )
            )
        )

        max_error_deg = float(np.max(error_deg))

        if max_error_deg <= HOME_JOINT_TOLERANCE_DEG:
            stable_count += 1
        else:
            stable_count = 0

        if DEBUG_JOINT and settle_frame % 30 == 0:
            _dlog(DEBUG_JOINT,
                f"[RIGHT HOME SETTLE {settle_frame:03d}/{HOME_SETTLE_FRAMES}]"
            )
            _dlog(DEBUG_JOINT, "joint error deg:", error_deg)
            _dlog(DEBUG_JOINT, "max error deg:", max_error_deg)
            _dlog(DEBUG_JOINT, "stable frames:", stable_count, "/", HOME_STABLE_FRAMES)

        if stable_count >= HOME_STABLE_FRAMES:
            reached = True
            break

    final_q = get_robot_joint_positions(ctrl)
    final_error_deg = np.abs(
        np.rad2deg(
            wrapped_joint_delta(
                commanded_home_q,
                final_q,
            )
        )
    )

    final_ee_pos, final_ee_orientation = get_world_pose(
        RIGHT_EE_PATH
    )

    print("final joint deg:", np.rad2deg(final_q))
    print("final joint error deg:", final_error_deg)
    print("final EE position:", final_ee_pos)
    print("final EE orientation wxyz:", final_ee_orientation)
    print("home reached:", reached)
    print("=== RIGHT RETURN HOME JOINT DONE ===")

    return reached


def return_right_robot_home(
    ctrl,
    home_q,
    home_ee_pos,
    home_ee_orientation,
    safe_lift_first=True,
):
    """
    상자 근처에서 곧바로 관절 복귀하면 링크가 박스를 스칠 수 있으므로,
    필요하면 현재 위치에서 수직 상승한 뒤 저장된 초기 관절각으로 복귀한다.
    """
    print("")
    print("====================================")
    print("=== RIGHT ROBOT RETURN HOME START ===")
    print("====================================")
    print("saved home EE position:", home_ee_pos)
    print("saved home EE orientation wxyz:", home_ee_orientation)

    if safe_lift_first:
        current_pos, current_orientation = get_world_pose(RIGHT_EE_PATH)
        safe_pos = current_pos.copy()
        safe_pos[2] += HOME_SAFE_LIFT_Z

        print("safe lift start:", current_pos)
        print("safe lift target:", safe_pos)

        move_right_ee(
            ctrl=ctrl,
            start_pos=current_pos,
            target_pos=safe_pos,
            start_orientation=current_orientation,
            target_orientation=current_orientation,
            frames=HOME_SAFE_LIFT_FRAMES,
            label="RIGHT HOME SAFE LIFT",
        )

    reached = move_right_joints_home(
        ctrl=ctrl,
        home_q=home_q,
    )

    final_pos, final_orientation = get_world_pose(RIGHT_EE_PATH)

    print("saved home EE position:", home_ee_pos)
    print("returned EE position:", final_pos)
    print("EE position error:", np.linalg.norm(final_pos - home_ee_pos))
    print("returned EE orientation wxyz:", final_orientation)
    print("===================================")
    print("=== RIGHT ROBOT RETURN HOME DONE ===")
    print("===================================")

    return reached


# ============================================================
# main
# ============================================================


# ============================================================
# 왼팔 ROS 토픽, 이벤트 값, 좌표 안정화 및 관절 Drive 설정
# ============================================================
