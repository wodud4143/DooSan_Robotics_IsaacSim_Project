"""Quaternion, vector, and easing helpers shared by arm controllers."""

import numpy as np

from .config import EE_SUCTION_LOCAL_AXIS, EE_WORLD_DOWN_AXIS

def normalize_quat(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)

    if norm < 1.0e-12:
        raise ValueError("Quaternion norm is zero")

    return q / norm


def quat_from_euler_deg(rx, ry, rz):
    rx = np.deg2rad(rx)
    ry = np.deg2rad(ry)
    rz = np.deg2rad(rz)

    cx = np.cos(rx * 0.5)
    sx = np.sin(rx * 0.5)
    cy = np.cos(ry * 0.5)
    sy = np.sin(ry * 0.5)
    cz = np.cos(rz * 0.5)
    sz = np.sin(rz * 0.5)

    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz

    return normalize_quat([w, x, y, z])


def quat_from_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1.0e-12:
        raise ValueError("Axis norm is zero")

    axis = axis / axis_norm

    rad = np.deg2rad(float(deg))
    half = rad * 0.5
    s = np.sin(half)

    return normalize_quat([
        np.cos(half),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
    ])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = normalize_quat(q1)
    w2, x2, y2, z2 = normalize_quat(q2)

    return normalize_quat([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    w, x, y, z = normalize_quat(q)
    return np.array([w, -x, -y, -z], dtype=float)


def quat_rotate_vector(q, vector):
    """wxyz 쿼터니언으로 3차원 벡터를 회전한다."""
    q = normalize_quat(q)
    vector = np.asarray(vector, dtype=float)

    pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=float)

    # 벡터 쿼터니언은 정규화하면 안 되므로 직접 곱한다.
    def raw_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], dtype=float)

    rotated = raw_mul(raw_mul(q, pure), quat_conjugate(q))
    return rotated[1:4]


def quat_slerp(q0, q1, alpha):
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    dot = float(np.dot(q0, q1))

    # q와 -q는 같은 회전이다. 최단 경로를 사용한다.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return normalize_quat(q0 + alpha * (q1 - q0))

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * alpha
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0

    return normalize_quat(s0 * q0 + s1 * q1)


def quat_from_rotation_matrix(matrix):
    """로컬축→월드축 3x3 회전행렬을 wxyz 쿼터니언으로 변환한다."""
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

    return normalize_quat([w, x, y, z])


def make_level_top_grasp_orientation(reference_orientation):
    """
    EE 로컬 +X축을 월드 -Z축에 맞춘다.

    로컬 X축이 흡착면 법선이므로 이 자세에서 흡착면의 로컬 Y-Z 평면은
    월드 XY 평면, 즉 지면과 정확히 평행하다. 수평면 내 방향은 초기 EE의
    로컬 Y축 투영을 사용해 불필요한 손목 회전을 줄인다.
    """
    reference_orientation = normalize_quat(reference_orientation)

    x_world = EE_WORLD_DOWN_AXIS / np.linalg.norm(EE_WORLD_DOWN_AXIS)

    # 초기 로컬 Y축의 수평 투영으로 yaw를 최대한 보존한다.
    y_reference = quat_rotate_vector(
        reference_orientation,
        np.array([0.0, 1.0, 0.0], dtype=float),
    )
    y_world = y_reference - x_world * float(np.dot(y_reference, x_world))

    if np.linalg.norm(y_world) < 1.0e-8:
        z_reference = quat_rotate_vector(
            reference_orientation,
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        y_world = z_reference - x_world * float(np.dot(z_reference, x_world))

    if np.linalg.norm(y_world) < 1.0e-8:
        y_world = np.array([0.0, 1.0, 0.0], dtype=float)

    y_world = y_world / np.linalg.norm(y_world)
    z_world = np.cross(x_world, y_world)
    z_world = z_world / np.linalg.norm(z_world)
    y_world = np.cross(z_world, x_world)
    y_world = y_world / np.linalg.norm(y_world)

    # 열 벡터가 각각 로컬 X/Y/Z축의 월드 방향이다.
    rotation_matrix = np.column_stack((x_world, y_world, z_world))
    return quat_from_rotation_matrix(rotation_matrix)


def make_top_grasp_orientation_from_normal(
    reference_orientation,
    surface_approach_normal_world,
):
    """
    EE 로컬 +X 흡착 법선을 Raycast로 추정한 실제 상면 접근 법선에 맞춘다.

    surface_approach_normal_world는 그리퍼에서 상자 표면을 향하는 방향이다.
    남는 회전 자유도는 reference_orientation의 로컬 +Y를 새 평면에 투영해
    기존 카메라/손목 yaw를 최대한 보존한다.
    """
    reference_orientation = normalize_quat(reference_orientation)
    x_world = np.asarray(surface_approach_normal_world, dtype=float).copy()
    x_norm = float(np.linalg.norm(x_world))
    if x_norm < 1.0e-8:
        raise ValueError("Top-surface approach normal is zero")
    x_world /= x_norm

    y_reference = quat_rotate_vector(
        reference_orientation,
        np.array([0.0, 1.0, 0.0], dtype=float),
    )
    y_world = y_reference - x_world * float(np.dot(y_reference, x_world))

    if np.linalg.norm(y_world) < 1.0e-8:
        z_reference = quat_rotate_vector(
            reference_orientation,
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        y_world = z_reference - x_world * float(np.dot(z_reference, x_world))

    if np.linalg.norm(y_world) < 1.0e-8:
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(fallback, x_world))) > 0.9:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        y_world = fallback - x_world * float(np.dot(fallback, x_world))

    y_world /= np.linalg.norm(y_world)
    z_world = np.cross(x_world, y_world)
    z_world /= np.linalg.norm(z_world)
    y_world = np.cross(z_world, x_world)
    y_world /= np.linalg.norm(y_world)

    return quat_from_rotation_matrix(
        np.column_stack((x_world, y_world, z_world))
    )


def angle_between_vectors_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1.0e-12 or b_norm < 1.0e-12:
        return float("inf")
    dot = float(np.clip(np.dot(a / a_norm, b / b_norm), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def get_ee_suction_axis_world(orientation):
    return quat_rotate_vector(
        orientation,
        EE_SUCTION_LOCAL_AXIS,
    )


def get_ee_level_error_deg(orientation):
    actual = get_ee_suction_axis_world(orientation)
    actual = actual / np.linalg.norm(actual)
    expected = EE_WORLD_DOWN_AXIS / np.linalg.norm(EE_WORLD_DOWN_AXIS)
    dot = float(np.clip(np.dot(actual, expected), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def quat_angle_error_deg(q0, q1):
    """두 자세 사이의 최소 회전각 오차를 degree로 반환한다."""
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(dot)))


def smoothstep01(value):
    value = float(np.clip(value, 0.0, 1.0))

    return value * value * value * (
        value * (value * 6.0 - 15.0) + 10.0
    )
