"""Configuration constants for the standalone dual-arm workflow."""

import numpy as np
from pxr import Gf

from . import runtime

WORLD_USD_PATH = runtime.WORLD_USD_PATH
SORTER_ROS_TOPIC = runtime.SORTER_ROS_TOPIC
SPAWN_INTERVAL_SEC = runtime.SPAWN_INTERVAL_SEC
SPAWN_ON_START = runtime.SPAWN_ON_START

SORTER_GRAPH_PATH = "/World/ConveyorBelt_A44/Sorter_01/ActionGraph"
SORTER_SWITCH_ATTR = SORTER_GRAPH_PATH + "/binary_switch.inputs:value"

BOX_ASSET_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/"
    "SM_CardBoxB_01_290.usd"
)

BARCODE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props/"
    "S_Barcode.usd"
)

# 생성되는 박스의 물리 질량. UsdPhysics MassAPI의 mass 단위는 kg이다.
SPAWN_BOX_MASS_KG = 1.5

# 박스마다 독립적으로 70% 확률로 바코드를 부착한다.
# 공정 시험 중 모든 박스에 바코드를 붙이려면 1.0으로 설정한다.
BARCODE_ATTACH_PROBABILITY = 0.60

# 바코드는 선택된 면 크기의 최대 약 58% 안에 들어오도록 자동 축소한다.
BARCODE_FACE_COVERAGE = 0.40
BARCODE_RANDOM_SCALE_MIN = 0.78

# 박스 표면과 바코드 사이의 간격이다.
# 기존 1.5 mm는 렌더링 시 Z-fighting이 발생할 수 있어 3 mm로 확대한다.
BARCODE_SURFACE_GAP = 0.003
BARCODE_EDGE_MARGIN_RATIO = 0.04

# 참조 에셋의 앞/뒷면 방향과 관계없이 카메라에서 보이도록 양면 렌더링한다.
BARCODE_FORCE_DOUBLE_SIDED = True

# Center 보정 후 바코드 형상 중심이 Mount 원점에서 허용되는 최대 오차이다.
BARCODE_CENTER_TOLERANCE = 0.003

# 원격 USD 로딩이 일시적으로 늦을 경우 부착을 다시 시도한다.
BARCODE_ATTACH_MAX_ATTEMPTS = 2

# 바닥면(-Z)은 컨베이어에 가려지고 충돌할 수 있으므로 제외한다.
# 바코드는 네 측면과 윗면 중 하나에만 부착한다.
BARCODE_ALLOWED_FACES = (
    "+X",
    "-X",
    "+Y",
    "-Y",
    "+Z",
    "-Z",
)

SPAWN_PARENT_PATH = "/World/SpawnedBoxes"
SPAWN_POSITION = Gf.Vec3d(15.3, 0.0, 1.9)
SPAWN_SCALE = Gf.Vec3f(1.0, 1.0, 1.0)

DEBUG_MOTION  = False   # 이동/회전 매 프레임 EE, alpha, error
DEBUG_GRIPPER = False   # raycast 거리, joint distances, 흡착 상태 반복
DEBUG_BARCODE = False   # 바코드 bounds, mount 계산값
DEBUG_JOINT   = False   # Home 복귀 매 프레임 관절각


def _dlog(flag, *args):
    """DEBUG 플래그가 True일 때만 출력한다."""
    if flag:
        print(*args)

# ============================================================
# 오른팔 설정
# ============================================================

CENTER_TOPIC = "/box_coordinate_center_R"

# /barcode_exist_R:
#   좌표 고정 시 최신 값이 1이면 정상 파지/상승 후 회전 없이
#   기존 고정 목적지로 바로 이동하여 내려놓는다.
# /barcode_exist_L:
#   /barcode_exist_R != 1인 작업에서만 사용한다.
#   각 90도 회전 뒤의 1초 검사 구간에서 1이 0.5초 연속 유지되면
#   추가 90도 회전 후
#   원래 파지 위치의 수직축으로 복귀하여 내려놓는다.
RIGHT_BARCODE_TOPIC = "/barcode_exist_R"
LEFT_BARCODE_TOPIC = "/barcode_exist_L"
LEFT_ROTATE_180_TOPIC = "/left_rotate_180"
LEFT_ROTATE_90_TOPIC = "/left_rotate_90"

RIGHT_PHASE_DONE_TOPIC = "/right_phase_done"
RIGHT_EXTRA_DONE_TOPIC = "/right_extra_done"
INSPECTION_RESULT_TOPIC = "/inspection_result"
SORTER_SWITCH_TOPIC = SORTER_ROS_TOPIC

# ============================================================
# 공정 상태: 모든 새 상자는 반드시 오른팔 초기 검사부터 시작한다.
# ============================================================
STATE_WAIT_RIGHT_INITIAL = "WAIT_RIGHT_INITIAL"
STATE_RIGHT_INITIAL_BUSY = "RIGHT_INITIAL_BUSY"
STATE_WAIT_LEFT = "WAIT_LEFT"
STATE_LEFT_BUSY = "LEFT_BUSY"
STATE_WAIT_RIGHT_FINAL = "WAIT_RIGHT_FINAL"
STATE_RIGHT_FINAL_BUSY = "RIGHT_FINAL_BUSY"


# ============================================================
# 공통 입력/시간 설정
# ============================================================

# 오른팔 barcode 토픽에서 목적지 직행으로 판단할 값이다.
BARCODE_TRIGGER_VALUE = 1

# 왼팔 회전 완료 토픽에서 완료로 판단할 값이다.
LEFT_ROTATE_TRIGGER_VALUE = 1

# /barcode_exist_R이 이 시간 동안 연속으로 1을 유지해야
# 오른팔의 고정 목적지 직행 조건으로 인정한다.
RIGHT_BARCODE_STABLE_SECONDS = 1.0

# 각 오른팔 회전 후 1초 검사 창에서 /barcode_exist_L이 이 시간 동안
# 중간의 0 없이 연속으로 1을 유지해야 바코드 검출로 인정한다.
# 검사 창 시작 전에 들어온 1은 사용하지 않으며, 검사 중 0이 한 번이라도
# 들어오면 연속 유지 타이머를 즉시 0으로 초기화한다.
LEFT_BARCODE_STABLE_SECONDS = 0.5

# 박스를 놓은 뒤 phase/extra done을 발행하기 전 대기 시간
RIGHT_PHASE_DONE_DELAY_SECONDS = 0.0

# 오른팔이 새 박스 좌표를 고정하기 위해 요구하는 연속 안정 시간이다.
# 기존 1.5초는 컨베이어 정지 후 출발 지연이 크게 느껴지므로 0.5초로 단축한다.
# /barcode_exist_R == 1 경로는 별도의 1.0초 연속 유지 조건도 동시에 만족해야 한다.
TARGET_STABLE_SECONDS = 0.5

# 좌표 안정 판정 허용 흔들림이다. 비전 좌표가 기준점에서 이 값보다 멀어지면
# 안정 타이머가 다시 시작된다. 기존 3 mm에서 5 mm로 완화한다.
TARGET_STABLE_TOLERANCE = 0.005  # 5 mm

RIGHT_ROBOT = "/World/UR10_surface_R"
RIGHT_EE_PATH = "/World/UR10_surface_R/UR10/ee_link"

RIGHT_GRIPPERS = [
    "/World/UR10_surface_R/UR10/ee_link/SurfaceGripper_upper",
    "/World/UR10_surface_R/UR10/ee_link/SurfaceGripper_lower",
]

# 시뮬레이션 한 프레임의 시간이다. 60 Hz 기준으로 프레임/초 변환에 사용한다.
PHYSICS_DT = 1.0 / 60.0

# 왼팔 제어도 같은 시뮬레이션 프레임 시간을 사용한다.
L_PHYSICS_DT = PHYSICS_DT

# ============================================================
# 전체 로봇 공정 배속
#
# 1.0: 기존 속도
# 1.5: 기존보다 약 1.5배 빠름
# 1.75: 권장 균형값
# 2.0: 기존보다 약 2배 빠름
#
# 좌표 안정화 시간과 바코드 판정 시간은 인식 신뢰도를 위해 배속하지 않는다.
# 이동 보간 프레임과 Raycast 접근량만 배속한다.
#
# 중요:
#   수렴 확인, 흡착 안정화, Home 정렬 확인 프레임은 배속하지 않는다.
#   이 구간까지 줄이면 RMPflow/PhysX가 목표에 도달하기 전에 실패 판정될 수 있다.
# ============================================================
PROCESS_SPEED_SCALE = 1.75


def process_motion_frames(base_frames, minimum=1):
    """실제 이동·회전 보간 프레임만 공정 배속에 맞게 줄인다."""
    return max(
        int(minimum),
        int(round(float(base_frames) / float(PROCESS_SPEED_SCALE))),
    )


def process_motion_step(base_step):
    """Raycast 접촉 접근의 프레임당 이동량을 공정 배속만큼 늘린다."""
    return float(base_step) * float(PROCESS_SPEED_SCALE)


def process_control_frames(base_frames, minimum=1):
    """수렴·흡착·안정성 확인 프레임은 실시간 60 Hz 기준으로 유지한다."""
    return max(int(minimum), int(base_frames))


# 실제 Collider를 확인하면서 표면으로 전진하는 Raycast 구간만 별도 가속한다.
# 안전점/프리그라스프 이동까지 이 배율로 프레임을 줄이면 긴 이동에서
# RMPflow가 목표에 도착하기 전에 다음 단계로 넘어갈 수 있다.
GRASP_APPROACH_SPEED_SCALE = 2.20


def grasp_motion_step(base_step):
    """Raycast 접촉 접근의 프레임당 전진량만 별도 배율만큼 높인다."""
    return float(base_step) * float(GRASP_APPROACH_SPEED_SCALE)


# 안전점/프리그라스프의 최대 명령 이동 속도이다.
# 목표 거리를 이 속도로 나눈 값으로 최소 보간 프레임을 자동 산정한다.
GRASP_WAYPOINT_MAX_SPEED_MPS = 0.80


def adaptive_grasp_waypoint_frames(start_position, target_position, configured_frames):
    """이동 거리에 비례해 RMPflow가 실제 도달 가능한 보간 프레임을 계산한다."""
    start_position = np.asarray(start_position, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    distance = float(np.linalg.norm(target_position - start_position))
    distance_frames = int(np.ceil(
        distance / max(GRASP_WAYPOINT_MAX_SPEED_MPS * PHYSICS_DT, 1.0e-9)
    ))
    return max(int(configured_frames), distance_frames, 1)


# 접근 전용 수렴 조건. 긴 이동이 보간 직후에도 조금 남는 경우를 허용한다.
GRASP_MOVE_CONVERGENCE_FRAMES = 60
GRASP_MOVE_STABLE_FRAMES = 4

# RMPflow 정책 좌표계 보정 배율이다.
ROBOT_SCALE_FOR_POLICY = 2.0

# 오른팔 완료 신호 발행 대기 시간을 프레임 수로 변환한 값이다.
RIGHT_PHASE_DONE_DELAY_FRAMES = max(
    1,
    int(round(RIGHT_PHASE_DONE_DELAY_SECONDS / PHYSICS_DT)),
)

# 오른팔 관절 drive stiffness 값이다.
STIFFNESS = 1000000.0

# 오른팔 관절 drive damping 값이다.
DAMPING = 5000.0
# 오른팔 관절 drive 최대 force 값이다.
MAX_FORCE = 1000000.0


# ============================================================
# 오른팔 로봇 각도/속도/이동 튜닝
# 프레임 수는 작을수록 빠르고, 클수록 느리게 움직인다.
# ============================================================

# 기본 파지 자세를 만들 때 사용할 Euler 각도(rx, ry, rz)이다.
RIGHT_GRASP_ORIENTATION_EULER_DEG = (90.0, 90.0, 0.0)

# 안전 웨이포인트까지 이동하는 보간 프레임 수이다.
SAFE_MOVE_FRAMES = process_motion_frames(30, minimum=17)

# 프리그라스프 지점까지 하강하는 보간 프레임 수이다.
DESCEND_FRAMES = process_motion_frames(30, minimum=17)

# EE 위치가 목표에 도달했다고 보는 허용 오차(m)이다.
MOVE_POSITION_TOLERANCE = 0.015

# 보간 후 목표 자세를 유지하며 수렴을 기다리는 최대 프레임 수이다.
MOVE_CONVERGENCE_FRAMES = process_control_frames(45)

# 목표 오차 안에 연속으로 머물러야 도착으로 인정하는 프레임 수이다.
MOVE_STABLE_FRAMES = process_control_frames(8)

# 표면 접촉 접근 시 매 프레임 전진하는 거리(m/frame)이다.
# 60 Hz 기준 0.0008 m/frame는 약 4.8 cm/s이다.
CONTACT_APPROACH_STEP = grasp_motion_step(0.001)

# 비전 표면 좌표를 지나 허용하는 최대 안전 여유(m)이다.
MAX_SURFACE_OVERRUN = 0.012

# 접근 중 Surface Gripper close 명령을 재시도하는 프레임 주기이다.
GRIP_RETRY_EVERY_FRAMES = process_motion_frames(5, minimum=3)

# 흡착 감지 직후 현재 EE 위치를 유지하는 프레임 수이다.
CONTACT_HOLD_FRAMES = process_motion_frames(12, minimum=5)

# Surface Gripper가 접촉으로 간주할 최대 거리(m)이다.
SURFACE_MAX_GRIP_DISTANCE = 0.015

# Surface Gripper 내부 close 재시도 간격(초)이다.
SURFACE_RETRY_INTERVAL = 5.0

# 흡착점에서 월드 -Z 방향으로 검사하는 레이캐스트 거리(m)이다.
RAYCAST_RANGE = 0.30

# 레이 시작점을 표면 방향으로 살짝 띄우는 거리(m)이다.
RAY_ORIGIN_OFFSET = 0.002

# 레이로 감지한 표면 앞에서 멈출 거리(m)이다.
RAY_STOP_DISTANCE = 0.010

# 비전 표면 Z와 레이 히트 Z를 같은 표면으로 보는 허용 오차(m)이다.
RAY_VISION_Z_TOLERANCE = 0.08

# 레이캐스트 실패가 연속으로 몇 프레임이면 접근을 중단할지 정한다.
RAY_MISS_LIMIT_FRAMES = 8

# 레이 정지 거리 도달 후 흡착을 기다리는 프레임 수이다.
RAY_CONTACT_SETTLE_FRAMES = process_control_frames(20)

# 오른팔 상면 자세 추정을 위한 보조 Ray의 TCP 중심 기준 반경(m)이다.
# 두 Attachment Ray만으로는 한 방향 기울기만 알 수 있으므로, 직교 방향
# 보조 Ray를 추가해 3개 이상의 비공선 hit로 실제 상면 평면을 추정한다.
TOP_RAY_PROBE_OFFSET = 0.04

# 상면 평면 SVD 추정에 필요한 최소 유효 hit 수이다.
TOP_PLANE_MIN_HITS = 3

# 추정 평면에 대한 Ray hit RMS 잔차 허용치(m)이다.
TOP_PLANE_FIT_RESIDUAL_TOLERANCE = 0.004

# 이 각도보다 작은 자세 보정은 이미 평행한 것으로 간주한다.
TOP_PLANE_AUTO_ALIGN_MIN_DEG = 0.25

# 잘못된 Collider를 따라 급격히 기울어지는 것을 막는 최대 자동 보정각이다.
TOP_PLANE_AUTO_ALIGN_MAX_DEG = 15.0

# 상면으로 인정할 수 있는 월드 -Z 기준 최대 기울기이다.
TOP_PLANE_MAX_TILT_FROM_WORLD_DOWN_DEG = 30.0

# TCP 중심을 고정한 채 실제 상면 법선으로 자세를 보정하는 프레임 수이다.
TOP_PLANE_AUTO_ALIGN_FRAMES = process_motion_frames(15, minimum=9)

# 자세 보정 후 TCP를 실제 상면 법선상의 프리그라스프 위치로 옮기는 프레임 수이다.
TOP_ALIGNED_PREGRASP_MOVE_FRAMES = process_motion_frames(15, minimum=9)

# 두 흡착컵이 모두 가까워진 뒤 추가로 천천히 누르는 거리(m/frame)이다.
DUAL_GRIP_PRESS_STEP = grasp_motion_step(0.0004)

# upper/lower 모두 붙은 상태가 연속으로 유지돼야 성공으로 확정하는 프레임 수이다.
DUAL_GRIP_STABLE_FRAMES = process_control_frames(5)

# 두 실제 Attachment Ray가 모두 이 거리 안에 들어온 뒤에만 동시에 close한다.
DUAL_GRIP_SYNC_CLOSE_DISTANCE = 0.012

# 동시 close 후 두 흡착컵 상태를 확인하는 최대 프레임 수이다.
DUAL_GRIP_SYNC_CLOSE_SETTLE_FRAMES = process_control_frames(12)

# 한쪽만 붙은 경우 두 컵을 열고 동시 close를 재시도하는 최대 횟수이다.
DUAL_GRIP_SYNC_CLOSE_MAX_RETRIES = 4

# 실제 TCP가 Hard Stop 목표에 수렴했다고 보는 법선 방향 오차(m)이다.
HARD_STOP_TRACKING_TOLERANCE = 0.0005

# 비전 표면 좌표보다 아래로 명령하지 않기 위한 안전 간격(m)이다.
HARD_SURFACE_GAP = 0.004

# 박스 표면 앞 프리그라스프 대기 거리(m)이다.
PREGRASP_DISTANCE = 0.09

# 프리그라스프보다 위쪽에 잡는 안전 웨이포인트 높이(m)이다.
SAFE_ABOVE_Z = 0.12

# 상자 윗면 파지를 위해 EE가 접근하는 월드 방향 벡터이다.
RIGHT_APPROACH_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=float)

# True이면 파지 전에 흡착면 법선축을 월드 -Z로 맞춘다.
FORCE_LEVEL_TOP_GRASP = True

# 흡착면 법선으로 사용하는 EE 로컬 축이다.
EE_SUCTION_LOCAL_AXIS = np.array([1.0, 0.0, 0.0], dtype=float)

# 흡착면이 맞춰야 할 월드 아래 방향 축이다.
EE_WORLD_DOWN_AXIS = np.array([0.0, 0.0, -1.0], dtype=float)

# 흡착면 수평 오차가 이 각도(deg)를 넘으면 접근을 중단한다.
EE_LEVEL_ABORT_TOLERANCE_DEG = 5.0

# 시작 직후 현재 EE 위치에서 흡착면 수평 Home을 다시 잡을지 정한다.
INITIAL_LEVEL_HOME = True

# 초기 수평 Home 자세로 보간하는 프레임 수이다.
INITIAL_LEVEL_FRAMES = process_control_frames(20)

# 초기 수평 Home 자세가 실제로 수렴할 때까지 기다리는 최대 프레임 수이다.
INITIAL_LEVEL_CONVERGENCE_FRAMES = process_control_frames(120)

# 초기 수평 Home 오차 안에 연속으로 머물러야 하는 프레임 수이다.
INITIAL_LEVEL_STABLE_FRAMES = process_control_frames(12)

# 초기 수평 Home 자세의 허용 각도 오차(deg)이다.
INITIAL_LEVEL_TOLERANCE_DEG = 2.0

# 초기 수평 Home 자세의 허용 위치 오차(m)이다.
INITIAL_LEVEL_POSITION_TOLERANCE = 0.015

# True이면 초기 카메라 하향 자세를 유지하는 기존 로직을 사용한다.
KEEP_CAMERA_DOWN_ORIENTATION = True

# CENTER_TOPIC은 상자 윗면 중심의 월드 좌표라고 가정한다.
# 박스 중심 좌표가 아니라 실제 윗면 표면 좌표여야 한다.

# 오른팔 그리퍼 close 후 흡착 상태가 안정될 때까지 기다리는 프레임 수이다.
GRIP_SETTLE_FRAMES = process_control_frames(10)

# 오른팔 파지 후 들어올리는 높이(m)이다.
LIFT_HEIGHT = 0.25

# 오른팔 상승 동작의 보간 프레임 수이다.
LIFT_FRAMES = process_motion_frames(30)

# 오른팔이 한 번에 회전하는 EE 로컬 X축 기준 각도(deg)이다.
EE_ROTATE_STEP_DEG = 90.0

# 왼쪽 바코드 검사를 위해 반복하는 오른팔 회전 횟수이다.
EE_ROTATE_STEPS = 3

# 오른팔 1회 회전 동작의 보간 프레임 수이다.
EE_ROTATE_STEP_FRAMES = process_motion_frames(60)

# 오른팔 각 회전 뒤 왼쪽 바코드를 확인하며 멈춰 있는 시간(초)이다.
EE_ROTATE_PAUSE_SECONDS = 1.0

# 오른팔 회전 뒤 검사 정지 시간을 프레임 수로 변환한 값이다.
EE_ROTATE_PAUSE_FRAMES = max(
    1,
    int(round(EE_ROTATE_PAUSE_SECONDS / PHYSICS_DT)),
)

# 오른팔 검사 회전의 총 누적 각도(deg)이다.
EE_ROTATE_TOTAL_DEG = EE_ROTATE_STEP_DEG * EE_ROTATE_STEPS

# 오른팔 로컬 X축 회전 방향 부호이다. 방향이 반대이면 +1.0으로 바꾼다.
RIGHT_ROTATE_SIGN = -1.0

# /barcode_exist_R == 1일 때 이동할 고정 목적지 EE X/Y 좌표(m)이다.
FIXED_PLACE_EE_XY = np.array([8.516, 0.014], dtype=float)

# 고정 목적지에서 파지 당시 EE Z보다 위에 둘 높이(m)이다.
PLACE_Z_ABOVE_GRASP = 0.20

# 오른팔 배치 위치 상공까지 수평 이동하는 보간 프레임 수이다.
PLACE_TRANSFER_FRAMES = process_motion_frames(30)

# 오른팔 배치 위치까지 수직 하강하는 보간 프레임 수이다.
PLACE_DESCEND_FRAMES = process_motion_frames(10, minimum=5)

# 오른팔 그리퍼 open 후 물체가 떨어지도록 기다리는 프레임 수이다.
RELEASE_SETTLE_FRAMES = process_control_frames(10)

# 오른팔 배치 후 수직으로 이탈하는 높이(m)이다.
RETREAT_Z = 0.25

# 오른팔 이탈 동작의 보간 프레임 수이다.
RETREAT_FRAMES = process_motion_frames(20)

# 오른팔 Home 복귀 전 먼저 수직 상승하는 안전 높이(m)이다.
HOME_SAFE_LIFT_Z = 0.15

# 오른팔 Home 복귀 전 안전 상승 동작의 보간 프레임 수이다.
HOME_SAFE_LIFT_FRAMES = process_motion_frames(20)

# 오른팔 Home 관절 복귀에 사용하는 최소 보간 프레임 수이다.
HOME_JOINT_FRAMES = process_motion_frames(20)

# 오른팔 Home 복귀의 최대 관절 속도(deg/sec)이다.
HOME_MAX_JOINT_SPEED_DEG_PER_SEC = 60.0 * PROCESS_SPEED_SCALE

# 오른팔 Home 관절 목표 유지 후 실제 수렴을 기다리는 프레임 수이다.
HOME_SETTLE_FRAMES = process_control_frames(60)

# 오른팔 Home 관절 오차가 안정됐다고 보기 위한 연속 프레임 수이다.
HOME_STABLE_FRAMES = process_control_frames(10)

# 오른팔 Home 관절 복귀 허용 오차(deg)이다.
HOME_JOINT_TOLERANCE_DEG = 1.0


# ============================================================
# 왼팔 로봇 제어 변수
#
# 단위:
#   - 위치/거리: m
#   - 각도: deg
#   - 시간: s
#   - 프레임: 60 Hz 시뮬레이션 프레임 기준
#
# 튜닝 원칙:
#   - 이동 프레임을 줄이면 빨라지고, 늘리면 느려진다.
#   - 허용 오차를 줄이면 정밀해지지만 수렴 실패 가능성이 커진다.
#   - Raycast/Hard-stop 값은 박스 관통 방지와 직접 관련되므로
#     한 번에 크게 변경하지 않는다.
# ============================================================

# ------------------------------------------------------------
# 1. 왼팔 측면 접근 방향과 EE 축 정의
# ------------------------------------------------------------

# 왼팔 TCP가 박스 측면으로 전진하는 월드 방향이다.
# 현재 배치는 +Y 쪽에서 박스를 향해 -Y로 접근하는 구조다.
# 실제 월드에서 반대쪽에서 접근하면 [0.0, +1.0, 0.0]으로 변경한다.
L_LEFT_APPROACH_DIRECTION = np.array([0.0, -1.0, 0.0], dtype=float)

# SurfaceGripper 흡착면의 법선으로 간주하는 ee_link 로컬축이다.
# 현재 툴 장착 구조에서는 EE 로컬 +X가 흡착면 앞쪽을 가리킨다.
# 이 축을 L_LEFT_APPROACH_DIRECTION과 일치시켜 흡착면을 박스 측면과 평행하게 만든다.
L_EE_SUCTION_LOCAL_AXIS = np.array([1.0, 0.0, 0.0], dtype=float)

# wrist_3가 지면과 평행한지 검사할 때 사용하는 EE 로컬 기준축이다.
# 현재 장착 구조에서는 EE 로컬 +Z가 wrist_3 수평 방향을 대표한다.
L_WRIST3_HORIZONTAL_LOCAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)

# 월드의 위쪽 방향이다. wrist_3 기준축의 Z 성분을 검사할 때 사용한다.
L_WORLD_UP_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)

# ------------------------------------------------------------
# 2. 프로그램 시작 시 왼팔 Home 자세 교정
# ------------------------------------------------------------

# True이면 시작 시 현재 EE 위치는 유지하고,
# 흡착면은 지면과 수직, wrist_3 기준축은 지면과 평행하도록 자세를 교정한다.
L_INITIAL_SIDE_HOME = True

# 초기 자세를 목표 자세로 보간하는 프레임 수이다.
# 작게 하면 빠르게 교정되지만 순간 관절 움직임이 커질 수 있다.
L_INITIAL_SIDE_FRAMES = process_control_frames(30)

# 보간 종료 후 초기 자세 목표를 계속 명령하며 수렴을 기다리는 최대 프레임 수이다.
L_INITIAL_SIDE_CONVERGENCE_FRAMES = process_control_frames(60)

# 위치·흡착면·wrist_3 오차가 허용 범위 안에 연속으로 머물러야 하는 프레임 수이다.
L_INITIAL_SIDE_STABLE_FRAMES = process_control_frames(12)

# 초기 자세 교정 시 EE 위치가 시작 위치에서 벗어나도 되는 최대 거리이다.
L_INITIAL_SIDE_POSITION_TOLERANCE = 0.015

# 초기 자세에서 흡착면 법선과 측면 접근 방향 사이의 최대 허용 각도 오차이다.
L_INITIAL_SIDE_NORMAL_TOLERANCE_DEG = 2.0

# 초기 자세에서 wrist_3 기준축과 지면 사이의 최대 허용 각도 오차이다.
L_INITIAL_WRIST3_HORIZONTAL_TOLERANCE_DEG = 2.0

# ------------------------------------------------------------
# 3. 공통 EE 이동 수렴 조건
# ------------------------------------------------------------

# 왼팔 EE가 목표 위치에 도착했다고 판단하는 위치 오차이다.
L_MOVE_POSITION_TOLERANCE = 0.015

# 지정 보간 프레임 이후에도 목표를 유지하며 추가 수렴을 확인하는 최대 프레임 수이다.
L_MOVE_CONVERGENCE_FRAMES = process_control_frames(45)

# 목표 오차 안에 연속으로 머물러야 이동 완료로 인정하는 프레임 수이다.
L_MOVE_STABLE_FRAMES = process_control_frames(8)

# ------------------------------------------------------------
# 4. 측면 TCP/Raycast 파지 접근
# ------------------------------------------------------------

# 현재 위치에서 박스 바깥쪽 안전 웨이포인트까지 이동하는 프레임 수이다.
L_SAFE_MOVE_FRAMES = process_motion_frames(20, minimum=11)

# 안전 웨이포인트에서 측면 프리그라스프 위치까지 이동하는 프레임 수이다.
L_DESCEND_FRAMES = process_motion_frames(20, minimum=11)

# 박스 측면 표면과 프리그라스프 TCP 사이의 거리이다.
# 값을 키우면 더 멀리서 Raycast 접근을 시작한다.
L_PREGRASP_DISTANCE = 0.09

# 프리그라스프보다 추가로 바깥쪽에 두는 안전 TCP 거리이다.
# 첫 번째 웨이포인트가 박스에서 충분히 떨어지도록 한다.
L_SAFE_OUTSIDE_DISTANCE = 0.12

# Raycast가 표면을 확인한 상태에서 매 프레임 TCP가 전진하는 거리이다.
# 60 Hz에서 0.0008 m/frame은 약 0.048 m/s이다.
L_CONTACT_APPROACH_STEP = grasp_motion_step(0.001)

# TCP 중심과 보조 지점에서 측면 방향으로 발사하는 Ray의 최대 길이이다.
L_RAYCAST_RANGE = 0.30

# Ray 시작점을 흡착점에서 접근 방향으로 조금 이동시키는 거리이다.
# 자기 Collider와 수치적으로 겹치는 문제를 줄인다.
L_RAY_ORIGIN_OFFSET = 0.002

# Ray가 검출한 Collider 앞에서 TCP 이동을 멈추는 거리이다.
L_RAY_STOP_DISTANCE = 0.010

# Ray hit 지점이 비전 측면 좌표와 같은 면이라고 인정하는 법선 방향 허용 오차이다.
L_RAY_SURFACE_PLANE_TOLERANCE = 0.08

# Ray가 연속으로 miss했을 때 접근을 중단하는 횟수이다.
# 너무 작으면 일시적인 scene-query 누락에도 실패하고, 너무 크면 실패 판정이 늦어진다.
L_RAY_MISS_LIMIT_FRAMES = 8

# Hard Stop에 도달한 뒤 현재 자세를 유지하며 두 흡착컵의 결합을
# 마지막으로 재시도하는 프레임 수이다.
L_RAY_CONTACT_SETTLE_FRAMES = process_control_frames(30)

# 한쪽 흡착컵만 붙었거나 Ray 정지 거리 안에 진입한 뒤,
# 두 흡착컵이 모두 붙도록 박스 방향으로 추가 전진하는 거리(m/frame)이다.
# 일반 접근보다 느리게 눌러 두 번째 흡착컵의 접촉 충격을 줄인다.
L_DUAL_GRIP_PRESS_STEP = grasp_motion_step(0.0004)

# upper/lower 두 흡착컵이 이 프레임 수만큼 연속으로 모두 붙어 있어야
# 최종 파지 성공으로 확정한다. 일시적인 상태 조회 오검출을 방지한다.
L_DUAL_GRIP_STABLE_FRAMES = process_control_frames(5)

# 두 실제 Attachment Joint Ray가 모두 이 거리 안에 들어온 뒤에만
# upper/lower SurfaceGripper를 동시에 close한다.
# Ray origin이 Attachment보다 L_RAY_ORIGIN_OFFSET만큼 앞에 있으므로
# 0.012 m는 실제 Attachment 거리 약 0.014 m에 해당한다.
L_DUAL_GRIP_SYNC_CLOSE_DISTANCE = 0.012

# 두 흡착컵 Ray 거리 차이가 이 값보다 크면 흡착면이 박스 측면과
# 평행하지 않은 것으로 판단하고, 두 hit 지점으로 실제 측면 방향을
# 추정해 TCP 중심을 고정한 채 EE 자세를 먼저 교정한다.
L_SIDE_PLANE_DISTANCE_SPREAD_TOLERANCE = 0.003

# 자동 측면 자세 교정의 최대 허용 각도이다. 그 이상이면 잘못된
# Collider를 본 것으로 판단하고 자동 교정을 적용하지 않는다.
# region 교정 각도 추가
L_SIDE_PLANE_AUTO_ALIGN_MAX_DEG = 12 #8.0

# TCP 중심을 고정한 채 실제 박스 측면에 자세를 맞추는 보간 프레임 수이다.
L_SIDE_PLANE_AUTO_ALIGN_FRAMES = process_motion_frames(15, minimum=9)

# 동시 close 후 두 컵의 결합 상태를 확인하는 최대 프레임 수이다.
L_DUAL_GRIP_SYNC_CLOSE_SETTLE_FRAMES = process_control_frames(12)

# 한쪽 컵만 붙은 경우 두 컵을 다시 열고 동시 close를 재시도하는 최대 횟수이다.
L_DUAL_GRIP_SYNC_CLOSE_MAX_RETRIES = 4

# command travel이 최대값에 도달한 뒤에도 실제 TCP가 Hard Stop 목표에
# 수렴할 때까지 같은 최종 목표를 계속 명령한다.
L_HARD_STOP_TRACKING_TOLERANCE = 0.0005

# 비전 측면 좌표를 지나치지 않도록 설정하는 TCP 안전 간격이다.
L_HARD_SURFACE_GAP = 0.004

# 접근 중 SurfaceGripper close 명령을 다시 호출하는 프레임 주기이다.
L_GRIP_RETRY_EVERY_FRAMES = process_motion_frames(5, minimum=3)

# Raycast 실패 또는 타임아웃 뒤 현재 EE 자세를 유지하는 프레임 수이다.
L_CONTACT_HOLD_FRAMES = process_motion_frames(12, minimum=5)

# 박스 측면 중심에 대한 TCP의 X/Z 방향 최대 허용 오차이다.
L_SIDE_LATERAL_TOLERANCE = 0.03

# 파지 접근 전 흡착면 법선 방향의 최대 허용 각도 오차이다.
L_SIDE_NORMAL_ABORT_TOLERANCE_DEG = 5.0

# 파지 접근 전 wrist_3 기준축과 지면 사이의 최대 허용 각도 오차이다.
L_WRIST3_HORIZONTAL_ABORT_TOLERANCE_DEG = 5.0

# SurfaceGripper가 물체를 붙일 수 있는 최대 물리 거리이다.
# L_RAY_STOP_DISTANCE보다 너무 작으면 Ray는 닿았지만 흡착은 실패할 수 있다.
L_SURFACE_MAX_GRIP_DISTANCE = 0.015

# SurfaceGripper 내부 재시도 간격 속성에 적용하는 값이다.
L_SURFACE_RETRY_INTERVAL = 5.0

# ------------------------------------------------------------
# 5. 그리퍼 해제, 상승, 회전
# ------------------------------------------------------------

# 왼팔 그리퍼를 연 뒤 물체가 완전히 분리되도록 기다리는 프레임 수이다.
L_RELEASE_SETTLE_FRAMES = process_control_frames(10)

# 파지 성공 후 EE를 월드 +Z 방향으로 들어 올리는 높이이다.
L_LIFT_HEIGHT = 0.25

# 상승 동작의 보간 프레임 수이다.
L_LIFT_FRAMES = process_motion_frames(30)

# 왼팔이 한 번에 회전하는 EE 로컬 X축 기준 각도이다.
L_EE_ROTATE_DEG = 90.0

# 90도 회전 한 번에 사용하는 보간 프레임 수이다.
# L_EE_ROTATE_FRAMES = process_motion_frames(30)
L_EE_ROTATE_FRAMES = process_control_frames(30)

# 각 90도 회전이 끝난 뒤 최종 자세와 TCP 위치를 유지하는 프레임 수이다.
L_ROTATE_HOLD_FRAMES = process_control_frames(60)

# 회전 중 TCP가 고정점에서 이 값보다 크게 벗어나면 회전을 실패로 처리한다.
L_ROTATE_TCP_ABORT_TOLERANCE = 0.020

# 회전 시작 전과 회전 중 upper/lower 두 흡착컵이 모두 붙어 있어야 한다.
# 파지 접근 단계가 두 컵 모두 붙을 때까지 전진하므로 회전도 같은 조건을 사용한다.
L_ROTATE_REQUIRE_ALL_GRIPPERS = True

# 회전 시작 전 두 흡착컵의 결합을 재확인하는 최대 프레임 수이다.
L_ROTATE_GRIP_CONFIRM_FRAMES = process_control_frames(30)

# 회전 중 그립 조회가 일시적으로 비어도 즉시 실패하지 않고, 이 프레임 수만큼
# 연속으로 모든 흡착이 사라졌을 때만 실제 그립 손실로 판정한다.
L_ROTATE_GRIP_LOSS_CONFIRM_FRAMES = process_control_frames(8)

# /right_phase_done 경로에서 수행하는 기본 90도 회전 횟수이다.
# 현재 2회이므로 총 180도 회전한다.
L_LEFT_SCAN_STEPS = 2

# 왼팔 회전 방향 부호이다. 현재 -1.0은 코드 기준 시계 방향이다.
# 실제 회전 방향이 반대이면 +1.0으로 변경한다.
L_LEFT_ROTATE_SIGN = -1.0

# ------------------------------------------------------------
# 6. 원래 위치 재배치와 이탈
# ------------------------------------------------------------

# 회전 검사 후 원래 파지 위치로 내려놓는 이동 프레임 수이다.
L_PLACE_FRAMES = process_motion_frames(40)

# 저장한 파지 EE 위치보다 Z축으로 추가 보정해 내려놓는 값이다.
# 양수이면 저장 위치보다 조금 위에서 해제한다.
L_PLACE_Z_OFFSET = 0.02

# 물체 해제 후 왼팔이 박스에서 바깥쪽으로 빠지는 Y 이동량이다.
L_RETREAT_Y = 0.20

# 이탈 중 동시에 위로 올리는 Z 이동량이다.
L_RETREAT_Z = 0.05

# 배치 후 이탈 동작의 보간 프레임 수이다.
L_RETREAT_FRAMES = process_motion_frames(20)

# ------------------------------------------------------------
# 7. 왼팔 Home 관절 복귀
# ------------------------------------------------------------

# 저장된 초기 Home 관절각으로 복귀하는 보간 프레임 수이다.
L_RETURN_HOME_FRAMES = process_motion_frames(40)

# Home 관절 목표를 유지하며 실제 관절 수렴을 기다리는 최대 프레임 수이다.
L_HOME_SETTLE_FRAMES = process_control_frames(60)

# 각 관절이 Home에 도달했다고 판단하는 최대 각도 오차이다.
L_HOME_JOINT_TOLERANCE_DEG = 1.0

L_CENTER_TOPIC = "/box_coordinate_center_L"

# 왼팔이 180도 회전한 뒤 오른팔 카메라로 바코드를 확인할 때 사용하는 토픽이다.
# 주의: 오른팔 회전 검사에 사용하는 /barcode_exist_L과는 역할이 다르다.
L_BARCODE_TOPIC = "/barcode_exist_R"

# 오른팔 일반 회전·재배치 공정이 끝났음을 알리는 시작 이벤트이다.
# 이 이벤트를 받으면 왼팔은 90도 두 번, 총 180도 모드로 동작한다.
L_RIGHT_PHASE_DONE_TOPIC = "/right_phase_done"

# 오른팔이 barcode_L 검출 후 추가 90도 회전·재배치를 끝냈음을 알리는 이벤트이다.
# 이 이벤트를 받으면 왼팔은 90도 한 번만 회전한다.
L_RIGHT_EXTRA_DONE_TOPIC = "/right_extra_done"

# 왼팔 180도 회전과 Home 복귀가 모두 끝난 뒤 발행하는 완료 토픽이다.
L_LEFT_ROTATE_180_TOPIC = "/left_rotate_180"

# 왼팔 90도 회전과 Home 복귀가 모두 끝난 뒤 발행하는 완료 토픽이다.
L_LEFT_ROTATE_90_TOPIC = "/left_rotate_90"

# 왼팔 검사 성공과 실패 상태를 외부에서 개별 확인하기 위한 토픽이다.
L_LEFT_SCAN_SUCCESS_TOPIC = "/left_scan_success"
L_LEFT_SCAN_FAILED_TOPIC = "/left_scan_failed"

# 왼팔 최종 검사 결과를 오른팔 공정 상태 머신에 전달하는 토픽이다.
# 1은 바코드 검출, 0은 미검출이다.
L_INSPECTION_RESULT_TOPIC = "/inspection_result"

# 오른팔 완료 이벤트에서 왼팔 시작 신호로 인정하는 Int32 값이다.
L_TRIGGER_VALUE = 1

# /barcode_exist_R에서 바코드 검출로 판단하는 값이다.
L_BARCODE_DETECTED_VALUE = 1

# 왼팔 검사 결과 발행값이다.
L_RESULT_OK = 1
L_RESULT_FAIL = 0

# 왼팔이 180도 회전한 뒤 오른팔 카메라 barcode_R을 확인하는 전체 시간이다.
L_BARCODE_CHECK_SECONDS = 1.0

# 위 검사 시간을 물리 프레임 수로 변환한 값이다.
L_BARCODE_CHECK_FRAMES = max(
    1,
    int(round(L_BARCODE_CHECK_SECONDS / L_PHYSICS_DT)),
)

# 왼팔 좌표가 이 시간 동안 안정돼야 파지 목표로 고정한다.
L_TARGET_STABLE_SECONDS = 1.5

# 후보 좌표가 이 거리보다 많이 변하면 안정화 타이머를 다시 시작한다.
L_TARGET_STABLE_TOLERANCE = 0.003  # 3 mm

# USD 내 왼팔 Articulation 루트 Prim 경로이다.
L_LEFT_ROBOT = "/World/UR10_surface_L"

# RMPflow가 제어하는 왼팔 end-effector Prim 경로이다.
L_LEFT_EE_PATH = "/World/UR10_surface_L/UR10/ee_link"

# 왼팔의 upper/lower SurfaceGripper Prim 경로이다.
L_LEFT_GRIPPERS = [
    "/World/UR10_surface_L/UR10/ee_link/SurfaceGripper_upper",
    "/World/UR10_surface_L/UR10/ee_link/SurfaceGripper_lower",
]

# 월드 좌표 목표를 RMPflow 정책 좌표로 변환할 때 사용하는 배율이다.
# 현재 월드/로봇 스케일 구성에 맞춘 값이므로 임의 변경하지 않는 것이 안전하다.
L_ROBOT_SCALE_FOR_POLICY = 2.0

# 왼팔 관절 Drive의 stiffness이다. 클수록 목표각을 강하게 추종한다.
L_STIFFNESS = 1000000.0

# 왼팔 관절 Drive의 damping이다. 진동과 오버슈트를 억제한다.
L_DAMPING = 5000.0

# 왼팔 관절 Drive가 사용할 수 있는 최대 force이다.
L_MAX_FORCE = 1000000.0
