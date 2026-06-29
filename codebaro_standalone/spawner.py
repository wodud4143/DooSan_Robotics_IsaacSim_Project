"""Cardbox spawning, barcode placement, and spawn timer management."""

import random
import time

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from . import runtime
from .config import *
from .config import _dlog
from .math_utils import *

stage = runtime.stage
simulation_app = runtime.simulation_app

SPAWN_COUNT = 0
LAST_SPAWN_TIME = None
SPAWNER_STARTED = False

def ensure_spawn_parent():
    parent = stage.GetPrimAtPath(SPAWN_PARENT_PATH)

    if not parent.IsValid():
        parent = stage.DefinePrim(SPAWN_PARENT_PATH, "Xform")
        print("[spawn] parent created:", SPAWN_PARENT_PATH)

    return parent


def apply_spawned_box_physics_safe(root_prim):
    """동적 박스에 RigidBody, 질량 1.5 kg, convexHull collider를 적용한다."""
    try:
        if root_prim.IsValid() and not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(root_prim)
            print("[spawn] RigidBodyAPI applied:", root_prim.GetPath())
    except Exception as exc:
        print("[WARN] RigidBodyAPI apply failed:", repr(exc))

    try:
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_attr = mass_api.GetMassAttr()

        if not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()

        mass_attr.Set(float(SPAWN_BOX_MASS_KG))
        print(
            "[spawn] box mass =",
            SPAWN_BOX_MASS_KG,
            "kg:",
            root_prim.GetPath(),
        )
    except Exception as exc:
        print("[WARN] MassAPI setup failed:", repr(exc))

    try:
        barcode_prefix = str(root_prim.GetPath()) + "/BarcodeMount"

        for prim in Usd.PrimRange(root_prim):
            # 바코드는 박스에 붙는 순수 렌더 형상이다.
            # 바코드 mesh에 CollisionAPI를 다시 적용하면 동적 박스의
            # triangle-mesh simulation shape 오류가 발생하므로 제외한다.
            if str(prim.GetPath()).startswith(barcode_prefix):
                continue

            if not prim.IsA(UsdGeom.Mesh):
                continue

            try:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)

                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                approximation_attr = mesh_collision_api.GetApproximationAttr()

                if not approximation_attr.IsValid():
                    approximation_attr = (
                        mesh_collision_api.CreateApproximationAttr()
                    )

                approximation_attr.Set("convexHull")
                print(
                    "[spawn] Mesh collision approximation = convexHull:",
                    prim.GetPath(),
                )
            except Exception as exc:
                print(
                    "[WARN] mesh collision setup failed:",
                    prim.GetPath(),
                    repr(exc),
                )
    except Exception as exc:
        print("[WARN] CollisionAPI scan failed:", repr(exc))


def _vec3_to_np(value):
    return np.array(
        [float(value[0]), float(value[1]), float(value[2])],
        dtype=float,
    )


def _get_valid_local_bounds(prim, min_nonzero_axes=3):
    """
    Prim 하위 렌더 형상의 로컬 정렬 바운딩 박스를 반환한다.

    min_nonzero_axes:
        박스는 3, 두께가 0일 수 있는 평면형 바코드는 2를 사용한다.

    반환:
        (minimum[3], maximum[3]) 또는 아직 bounds가 없으면 None
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )

    local_bound = bbox_cache.ComputeLocalBound(prim)
    aligned_box = local_bound.ComputeAlignedBox()

    minimum = _vec3_to_np(aligned_box.GetMin())
    maximum = _vec3_to_np(aligned_box.GetMax())
    size = maximum - minimum

    if (
        not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(size < -1.0e-9)
        or int(np.count_nonzero(size > 1.0e-6)) < int(min_nonzero_axes)
    ):
        return None

    return minimum, maximum


def _wait_for_valid_local_bounds(
    prim,
    max_frames=120,
    min_nonzero_axes=3,
):
    """원격 USD reference의 형상이 로드될 때까지 bounds를 확인한다."""
    for _ in range(max(1, int(max_frames))):
        bounds = _get_valid_local_bounds(
            prim,
            min_nonzero_axes=min_nonzero_axes,
        )
        if bounds is not None:
            return bounds

        simulation_app.update()

    return None


def _quat_from_two_vectors(source, target):
    """source 단위벡터를 target 단위벡터로 회전하는 wxyz 쿼터니언."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)

    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 1.0 - 1.0e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    if dot < -1.0 + 1.0e-9:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(source, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)

        axis = np.cross(source, fallback)
        axis /= np.linalg.norm(axis)
        return quat_from_axis_angle(axis, 180.0)

    cross = np.cross(source, target)
    return normalize_quat([
        1.0 + dot,
        cross[0],
        cross[1],
        cross[2],
    ])


def _prepare_barcode_visual_and_physics(barcode_root):
    """
    참조된 바코드 에셋을 박스의 순수 시각 자식으로 정리한다.

    - 루트 resetXformStack을 해제해 BarcodeMount/박스 변환을 반드시 상속한다.
    - 모든 Gprim을 visible + double-sided로 만들어 앞/뒷면 방향 문제를 제거한다.
    - 바코드 내부 collider/rigid body를 비활성화한다.
    """
    try:
        barcode_root.SetInstanceable(False)
    except Exception as exc:
        print("[WARN] barcode SetInstanceable(False) failed:", repr(exc))

    visible_count = 0
    double_sided_count = 0
    collision_disabled_count = 0
    rigid_disabled_count = 0

    try:
        for prim in Usd.PrimRange(barcode_root):
            try:
                # 참조 USD 내부 하위 Prim이 resetXformStack을 사용하면
                # BarcodeMount/Scale/Center 변환을 무시하고 월드 원점 부근에
                # 나타날 수 있다. 모든 Xformable 하위에서 강제로 해제한다.
                if prim.IsA(UsdGeom.Xformable):
                    UsdGeom.Xformable(prim).SetResetXformStack(False)

                if prim.IsA(UsdGeom.Imageable):
                    imageable = UsdGeom.Imageable(prim)
                    visibility_attr = imageable.GetVisibilityAttr()
                    if not visibility_attr.IsValid():
                        visibility_attr = imageable.CreateVisibilityAttr()
                    visibility_attr.Set(UsdGeom.Tokens.inherited)
                    visible_count += 1

                if BARCODE_FORCE_DOUBLE_SIDED and prim.IsA(UsdGeom.Gprim):
                    gprim = UsdGeom.Gprim(prim)
                    double_sided_attr = gprim.GetDoubleSidedAttr()
                    if not double_sided_attr.IsValid():
                        double_sided_attr = gprim.CreateDoubleSidedAttr()
                    double_sided_attr.Set(True)
                    double_sided_count += 1

                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI(prim)
                    enabled_attr = collision_api.GetCollisionEnabledAttr()
                    if not enabled_attr.IsValid():
                        enabled_attr = collision_api.CreateCollisionEnabledAttr()
                    enabled_attr.Set(False)
                    collision_disabled_count += 1

                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_api = UsdPhysics.RigidBodyAPI(prim)
                    enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
                    if not enabled_attr.IsValid():
                        enabled_attr = rigid_api.CreateRigidBodyEnabledAttr()
                    enabled_attr.Set(False)
                    rigid_disabled_count += 1

            except Exception as exc:
                print(
                    "[WARN] barcode visual/physics setup failed:",
                    prim.GetPath(),
                    repr(exc),
                )
    except Exception as exc:
        print("[WARN] barcode subtree scan failed:", repr(exc))

    _dlog(DEBUG_BARCODE,
        "[barcode] visual/physics prepared:",
        "visible=", visible_count,
        "double_sided=", double_sided_count,
        "collision_disabled=", collision_disabled_count,
        "rigid_disabled=", rigid_disabled_count,
    )



# 기존 함수 이름을 호출하는 코드가 남아 있어도 동작하도록 호환 래퍼를 유지한다.
def _disable_barcode_physics(barcode_root):
    _prepare_barcode_visual_and_physics(barcode_root)


def _face_definition(face_name):
    """면 이름을 (법선축 index, 부호, 법선 벡터, 접선축 2개)로 변환한다."""
    table = {
        "+X": (0, +1.0, np.array([+1.0, 0.0, 0.0]), (1, 2)),
        "-X": (0, -1.0, np.array([-1.0, 0.0, 0.0]), (1, 2)),
        "+Y": (1, +1.0, np.array([0.0, +1.0, 0.0]), (0, 2)),
        "-Y": (1, -1.0, np.array([0.0, -1.0, 0.0]), (0, 2)),
        "+Z": (2, +1.0, np.array([0.0, 0.0, +1.0]), (0, 1)),
        "-Z": (2, -1.0, np.array([0.0, 0.0, -1.0]), (0, 1)),
    }
    return table[face_name]


def _get_valid_world_bounds(prim, min_nonzero_axes=2):
    """Prim 하위 렌더 형상의 월드 정렬 bounds를 반환한다."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )
    world_bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = world_bound.ComputeAlignedBox()
    minimum = _vec3_to_np(aligned_box.GetMin())
    maximum = _vec3_to_np(aligned_box.GetMax())
    size = maximum - minimum

    if (
        not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(size < -1.0e-9)
        or int(np.count_nonzero(size > 1.0e-6)) < int(min_nonzero_axes)
    ):
        return None

    return minimum, maximum


def attach_random_barcode_to_box(box_prim, box_prim_path):
    """
    선택한 박스 면에 시각 전용 바코드를 부착한다.

    구조:
        box_prim_path/
          BarcodeMount  ← 위치 + 회전 + 스케일 + 중심 보정 통합 (박스 로컬 좌표)
            Asset       ← USD reference geometry

    중요:
        - 이 함수는 박스에 RigidBodyAPI를 적용하기 전에 호출해야 한다.
        - box_bounds를 박스 순수 로컬 공간으로 변환해 사용한다.
        - mount_position은 박스 로컬 좌표 → BarcodeMount에 직접 설정 가능.
    """

    # ── 박스 월드 transform 읽기 ──────────────────────────────────
    box_xformable = UsdGeom.Xformable(box_prim)
    box_world_matrix = box_xformable.ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    try:
        box_world_matrix_inv = box_world_matrix.GetInverse()
    except Exception:
        box_world_matrix_inv = None

    # ── 박스 로컬 bounds 계산 ─────────────────────────────────────
    # ComputeLocalBound는 부모 공간 기준을 반환하므로
    # 박스 world_matrix 역행렬을 이용해 순수 로컬 좌표로 변환한다.
    raw_bounds = _wait_for_valid_local_bounds(box_prim)
    if raw_bounds is None:
        print("[WARN] box local bounds unavailable; barcode skipped")
        return False

    raw_min, raw_max = raw_bounds

    if box_world_matrix_inv is not None:
        def _transform_point(m_inv, pt):
            p = Gf.Vec4d(float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
            r = p * m_inv
            return np.array([float(r[0]), float(r[1]), float(r[2])], dtype=float)

        corners = np.array([
            _transform_point(box_world_matrix_inv, c)
            for c in [
                raw_min,
                [raw_max[0], raw_min[1], raw_min[2]],
                [raw_min[0], raw_max[1], raw_min[2]],
                [raw_min[0], raw_min[1], raw_max[2]],
                [raw_max[0], raw_max[1], raw_min[2]],
                [raw_max[0], raw_min[1], raw_max[2]],
                [raw_min[0], raw_max[1], raw_max[2]],
                raw_max,
            ]
        ])
        box_min = corners.min(axis=0)
        box_max = corners.max(axis=0)
    else:
        box_min, box_max = raw_min, raw_max

    box_size   = box_max - box_min
    box_center = 0.5 * (box_min + box_max)

    _dlog(DEBUG_BARCODE, "[barcode] box local bounds min:", box_min)
    _dlog(DEBUG_BARCODE, "[barcode] box local bounds max:", box_max)
    _dlog(DEBUG_BARCODE, "[barcode] box local size:", box_size)


    # ── 경로 설정 (2계층) ─────────────────────────────────────────
    mount_path = f"{box_prim_path}/BarcodeMount"
    asset_path = f"{mount_path}/Asset"

    if stage.GetPrimAtPath(mount_path).IsValid():
        stage.RemovePrim(mount_path)
        for _ in range(2):
            simulation_app.update()

    mount_prim = stage.DefinePrim(mount_path, "Xform")
    asset_prim = stage.DefinePrim(asset_path, "Xform")
    asset_prim.SetInstanceable(False)
    asset_prim.GetReferences().AddReference(BARCODE_USD)

    try:
        stage.Load(asset_path)
    except Exception as exc:
        print("[WARN] barcode stage.Load failed:", repr(exc))

    _prepare_barcode_visual_and_physics(asset_prim)

    for _ in range(10):
        simulation_app.update()

    _prepare_barcode_visual_and_physics(asset_prim)

    # ── 바코드 Asset 로컬 bounds ──────────────────────────────────
    barcode_bounds = _wait_for_valid_local_bounds(
        asset_prim,
        max_frames=120,
        min_nonzero_axes=2,
    )
    if barcode_bounds is None:
        print("[WARN] barcode Asset-local bounds unavailable; barcode skipped")
        stage.RemovePrim(mount_path)
        return False

    barcode_min, barcode_max = barcode_bounds
    barcode_center    = 0.5 * (barcode_min + barcode_max)
    barcode_size      = barcode_max - barcode_min
    barcode_half_size = 0.5 * barcode_size

    _dlog(DEBUG_BARCODE, "[barcode] asset local bounds min:", barcode_min)
    _dlog(DEBUG_BARCODE, "[barcode] asset local bounds max:", barcode_max)
    _dlog(DEBUG_BARCODE, "[barcode] asset local size:", barcode_size)
    _dlog(DEBUG_BARCODE, "[barcode] asset local center:", barcode_center)

    if (
        not np.all(np.isfinite(barcode_center))
        or not np.all(np.isfinite(barcode_size))
        or int(np.count_nonzero(barcode_size > 1.0e-6)) < 2
    ):
        print("[WARN] invalid barcode geometry bounds")
        stage.RemovePrim(mount_path)
        return False

    # 바코드 법선 축: 가장 얇은 축
    barcode_normal_axis_index = int(np.argmin(barcode_size))
    barcode_local_normal = np.zeros(3, dtype=float)
    barcode_local_normal[barcode_normal_axis_index] = 1.0

    # ── 면 선택 및 회전 계산 ──────────────────────────────────────
    face_name = random.choice(BARCODE_ALLOWED_FACES)
    normal_axis, normal_sign, face_normal, tangent_axes = (
        _face_definition(face_name)
    )

    q_align = _quat_from_two_vectors(barcode_local_normal, face_normal)
    in_plane_angle_deg = random.uniform(0.0, 360.0)
    q_spin = quat_from_axis_angle(face_normal, in_plane_angle_deg)
    mount_orientation = quat_mul(q_spin, q_align)

    # ── 스케일 계산 ───────────────────────────────────────────────
    rotation_columns = np.column_stack([
        quat_rotate_vector(mount_orientation, np.eye(3, dtype=float)[i])
        for i in range(3)
    ])
    rotated_half_extent_unit = np.abs(rotation_columns) @ barcode_half_size

    scale_limits = []
    for axis_index in tangent_axes:
        denominator = 2.0 * float(rotated_half_extent_unit[axis_index])
        if denominator > 1.0e-9:
            scale_limits.append(
                float(box_size[axis_index])
                * BARCODE_FACE_COVERAGE
                / denominator
            )

    if not scale_limits:
        print("[WARN] invalid barcode scale limits; barcode skipped")
        stage.RemovePrim(mount_path)
        return False

    fit_scale = min(scale_limits)
    uniform_scale = fit_scale * random.uniform(BARCODE_RANDOM_SCALE_MIN, 1.0)
    if not np.isfinite(uniform_scale) or uniform_scale <= 1.0e-6:
        print("[WARN] invalid barcode scale:", uniform_scale)
        stage.RemovePrim(mount_path)
        return False

    rotated_half_extent = rotated_half_extent_unit * uniform_scale

    # geometry 중심 보정 (Asset 로컬 center를 회전+스케일 후 반영)
    rotated_barcode_center = (
        quat_rotate_vector(mount_orientation, barcode_center) * uniform_scale
    )

    # ── 접선 방향 위치 결정 (박스 로컬 좌표) ─────────────────────
    mount_position = box_center.copy()

    for axis_index in tangent_axes:
        face_margin = max(
            BARCODE_SURFACE_GAP,
            float(box_size[axis_index]) * BARCODE_EDGE_MARGIN_RATIO,
        )
        low  = float(box_min[axis_index]) + face_margin + float(rotated_half_extent[axis_index])
        high = float(box_max[axis_index]) - face_margin - float(rotated_half_extent[axis_index])
        mount_position[axis_index] = (
            random.uniform(low, high)
            if low <= high
            else 0.5 * (float(box_min[axis_index]) + float(box_max[axis_index]))
        )

    # 법선 방향 위치 (면에서 gap만큼 띄움)
    if normal_sign > 0.0:
        mount_position[normal_axis] = (
            float(box_max[normal_axis])
            + BARCODE_SURFACE_GAP
            + float(rotated_half_extent[normal_axis])
        )
    else:
        mount_position[normal_axis] = (
            float(box_min[normal_axis])
            - BARCODE_SURFACE_GAP
            - float(rotated_half_extent[normal_axis])
        )

    calculated_surface_gap = abs(
        float(mount_position[normal_axis])
        - float(rotated_half_extent[normal_axis])
        - (float(box_max[normal_axis]) if normal_sign > 0.0 else float(box_min[normal_axis]))
    )

    # Center 보정을 mount_position에 직접 반영
    mount_position_final = mount_position - rotated_barcode_center

    _dlog(DEBUG_BARCODE, "[barcode] mount_position (before center correction):", mount_position)
    _dlog(DEBUG_BARCODE, "[barcode] rotated_barcode_center:", rotated_barcode_center)
    _dlog(DEBUG_BARCODE, "[barcode] mount_position_final (local):", mount_position_final)

    # ── BarcodeMount Xform 설정 (박스 로컬 좌표) ─────────────────
    mount_xform = UsdGeom.Xformable(mount_prim)
    mount_xform.ClearXformOpOrder()
    mount_xform.AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in mount_position_final])
    )
    mount_xform.AddOrientOp().Set(
        Gf.Quatf(
            float(mount_orientation[0]),
            float(mount_orientation[1]),
            float(mount_orientation[2]),
            float(mount_orientation[3]),
        )
    )
    mount_xform.AddScaleOp().Set(
        Gf.Vec3f(float(uniform_scale), float(uniform_scale), float(uniform_scale))
    )

    for _ in range(8):
        simulation_app.update()

    _prepare_barcode_visual_and_physics(asset_prim)

    # ── 검증 ─────────────────────────────────────────────────────
    final_bounds = _get_valid_world_bounds(asset_prim, min_nonzero_axes=2)
    if final_bounds is None:
        print("[WARN] final barcode world bounds unavailable; barcode skipped")
        stage.RemovePrim(mount_path)
        return False

    final_min, final_max    = final_bounds
    final_center_world      = 0.5 * (final_min + final_max)
    final_size_world        = final_max - final_min

    mount_matrix = UsdGeom.Xformable(mount_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    mount_transform      = Gf.Transform(mount_matrix)
    expected_translation = mount_transform.GetTranslation()
    expected_center_world = np.array([
        float(expected_translation[0]),
        float(expected_translation[1]),
        float(expected_translation[2]),
    ], dtype=float)

    center_error_world = float(
        np.linalg.norm(final_center_world - expected_center_world)
    )

    if center_error_world > 0.01:
        print("[WARN] barcode world-placement verification failed")
        print("expected Mount world center:", expected_center_world)
        print("actual render world center:", final_center_world)
        print("world center error:", center_error_world)
        stage.RemovePrim(mount_path)
        return False

    if not (
        np.isfinite(calculated_surface_gap)
        and calculated_surface_gap >= 0.5 * BARCODE_SURFACE_GAP
    ):
        print("[WARN] barcode surface-gap verification failed:", calculated_surface_gap)
        stage.RemovePrim(mount_path)
        return False

    print("")
    print("=== RANDOM BARCODE ATTACHED / VERIFIED ===")
    print("box:", box_prim_path)
    print("face:", face_name)
    print("face normal:", face_normal)
    print("mount position final (box local):", mount_position_final)
    print("in-plane angle deg:", in_plane_angle_deg)
    print("uniform scale:", uniform_scale)
    print("barcode asset-local size:", barcode_size)
    print("barcode asset-local center:", barcode_center)
    print("final barcode world AABB size:", final_size_world)
    print("box local size:", box_size)
    print("surface gap:", calculated_surface_gap)
    print("expected Mount world center:", expected_center_world)
    print("actual render world center:", final_center_world)
    print("world center error:", center_error_world)
    print("double sided:", BARCODE_FORCE_DOUBLE_SIDED)
    print("============================================")

    return True

def spawn_box():
    """
    설정된 위치에 카드박스 한 개를 생성한다.

    - 박스 질량: 1.5 kg
    - 바코드: BARCODE_ATTACH_PROBABILITY 확률
    - 부착 위치: 바닥면을 제외한 5개 면 중 임의 면의 내부 위치
    - 자세: 바코드 판이 선택된 박스 면과 평행
    """
    global SPAWN_COUNT

    ensure_spawn_parent()

    prim_path = f"{SPAWN_PARENT_PATH}/CardBox_{SPAWN_COUNT:04d}"
    SPAWN_COUNT += 1

    attach_barcode = (
        random.random() < BARCODE_ATTACH_PROBABILITY
    )
    # region 박스 크기 수정
    box_width_scale = random.uniform(0.8, 1.1)
    box_height_scale = random.uniform(0.8, 1.1)

    spawn_scale = Gf.Vec3f(
        float(box_width_scale),   # X width: 40~55cm
        float(box_width_scale),   # Y width: 40~55cm
        float(box_height_scale),  # Z height: 40~55cm
    )
    print("")
    print("==============================================")
    print("=== SPAWN BOX REQUEST ===")
    print("prim path:", prim_path)
    print("asset:", BOX_ASSET_URL)
    print("position:", tuple(SPAWN_POSITION))
    # print("scale:", tuple(SPAWN_SCALE))
    """"""""""""""""""
    print("scale:", tuple(spawn_scale))
    print("box width cm:", box_width_scale * 50.0)
    print("box height cm:", box_height_scale * 50.0)
    """"""""""""""""""
    print("mass kg:", SPAWN_BOX_MASS_KG)
    print(
        "barcode selected:",
        attach_barcode,
        f"(probability={BARCODE_ATTACH_PROBABILITY:.2f})",
    )
    print("==============================================")

    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(BOX_ASSET_URL)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(SPAWN_POSITION)
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # xform.AddScaleOp().Set(SPAWN_SCALE)
    xform.AddScaleOp().Set(spawn_scale)

    # reference composition과 box asset 로딩을 위한 갱신 구간.
    for _ in range(10):
        simulation_app.update()

    # 중요: 박스를 동적 RigidBody로 만들기 전에 바코드 reference를 로드하고
    # 바코드 내부 collider를 비활성화한다. 순서가 반대면 S_Barcode의 triangle
    # mesh collider가 동적 박스의 simulation shape로 파싱되어 PhysX 오류가 난다.
    barcode_attached = False
    if attach_barcode:
        for attempt_index in range(1, BARCODE_ATTACH_MAX_ATTEMPTS + 1):
            print(
                f"[spawn] barcode attach attempt "
                f"{attempt_index}/{BARCODE_ATTACH_MAX_ATTEMPTS}"
            )
            barcode_attached = attach_random_barcode_to_box(
                box_prim=prim,
                box_prim_path=prim_path,
            )
            if barcode_attached:
                break

            if stage.GetPrimAtPath(f"{prim_path}/BarcodeMount").IsValid():
                stage.RemovePrim(f"{prim_path}/BarcodeMount")
            for _ in range(5):
                simulation_app.update()
    else:
        print(
            f"[spawn] barcode intentionally omitted by random selection "
            f"(probability={BARCODE_ATTACH_PROBABILITY:.2f})"
        )

    # 바코드 physics가 비활성화된 뒤 박스 본체 mesh에만 collider/RigidBody를 적용한다.
    apply_spawned_box_physics_safe(prim)

    print(
        "[spawn] done:",
        prim_path,
        "| mass kg:",
        SPAWN_BOX_MASS_KG,
        "| barcode:",
        barcode_attached,
    )

def start_box_spawner():
    """자동 소환 타이머를 시작하고 옵션에 따라 첫 박스를 즉시 생성한다."""
    global LAST_SPAWN_TIME, SPAWNER_STARTED

    SPAWNER_STARTED = True
    LAST_SPAWN_TIME = time.monotonic()

    if SPAWN_ON_START:
        spawn_box()
        LAST_SPAWN_TIME = time.monotonic()


def _box_spawn_window_is_safe():
    """
    원격 USD reference 로딩은 여러 simulation update를 소비한다.
    로봇 이동 중 spawn하면 RMPflow 목표 갱신이 끊기므로 공정 대기 상태에서만 허용한다.
    """
    right_node = runtime.ROS_NODE
    if right_node is not None and hasattr(right_node, "get_process_state"):
        state = right_node.get_process_state()
        if state != STATE_WAIT_RIGHT_INITIAL:
            return False

        # WAIT_RIGHT_INITIAL이어도 기존 박스 좌표가 들어오는 중이거나
        # 이미 목표가 고정된 상태에서는 새 박스를 추가하지 않는다.
        if getattr(right_node, "target", None) is not None:
            return False
        if getattr(right_node, "candidate_anchor", None) is not None:
            return False
        if bool(getattr(right_node, "cycle_route_armed", False)):
            return False

    from .background_home import (
        is_background_left_home_active,
        is_background_right_home_active,
    )

    if is_background_right_home_active():
        return False
    if is_background_left_home_active():
        return False

    return True

def tick_box_spawner():
    """설정 간격이 지나도 로봇이 움직이는 동안에는 생성을 안전하게 연기한다."""
    global LAST_SPAWN_TIME

    if not SPAWNER_STARTED:
        return

    now = time.monotonic()

    if LAST_SPAWN_TIME is None:
        LAST_SPAWN_TIME = now
        return

    if now - LAST_SPAWN_TIME < SPAWN_INTERVAL_SEC:
        return

    if not _box_spawn_window_is_safe():
        return

    spawn_box()
    LAST_SPAWN_TIME = time.monotonic()
