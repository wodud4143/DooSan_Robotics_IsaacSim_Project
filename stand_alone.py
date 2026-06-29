"""Isaac Sim standalone dual-arm + conveyor sorter + box spawner controller.

통합 기능:
    - 오른팔 상면 TCP/Raycast 파지 + 왼팔 측면 TCP/Raycast 파지
    - 왼팔 초기 흡착면 수직 및 wrist3 기준축 지면 평행 자세 교정
    - /sorter_switch Int32 구독 후 Sorter ActionGraph binary_switch 제어
    - 기본 10초 간격 카드박스 자동 소환

실행 예:
    ./python.sh right_robot_ros2_standalone.py --usd /absolute/path/to/world.usd

ROS 2 입력:
    /box_coordinate_center_R  (geometry_msgs/msg/Point)
    /barcode_exist_R          (std_msgs/msg/Int32)
    /barcode_exist_L          (std_msgs/msg/Int32)
    /left_rotate_180          (std_msgs/msg/Int32)
    /left_rotate_90           (std_msgs/msg/Int32)

ROS 2 출력:
    /right_phase_done         (std_msgs/msg/Int32, data=1)
    /right_extra_done         (std_msgs/msg/Int32, data=1)
    /sorter_switch            (std_msgs/msg/Int32, data=0 또는 1)

동작 조건:
    1. 안정된 박스 좌표가 준비되고 /barcode_exist_R이 0 또는 1로 수신되면
       오른팔 작업을 시작한다. 왼팔 회전 완료 신호도 목적지 직행 트리거가 된다.
    2. /barcode_exist_R == 0:
       - 상자를 파지하고 상승
       - 90도씩 회전하며 각 회전 뒤 /barcode_exist_L을 1초 검사
       - /barcode_exist_L == 1이면 90도 추가 회전 후 원래 위치에 재배치
       - 270도까지 미검출이면 원래 위치에 재배치
    3. 접근 시작부터 실제 파지 함수 반환 직전까지 /barcode_exist_R을
       매 시뮬레이션 프레임마다 계속 확인한다.
    4. 파지 전 /barcode_exist_R == 1이 1초 이상 연속 유지되면:
       - 회전 경로를 취소하고 목적지 직행으로 변경
       - sorter 값을 0으로 확정
       - 모든 ROS 입력 콜백 잠금
       - 진행 중인 파지를 완료한 뒤 고정 목적지로 직행
    5. /left_rotate_180 == 1 또는 /left_rotate_90 == 1이 확인되면
       해당 사이클을 목적지 직행 경로로 변경한다.
    6. /sorter_switch == 1이 되는 유일한 조건:
       /left_rotate_180 == 1 경로이고 파지 완료까지 /barcode_exist_R == 0인 경우.
       그 외 모든 목적지 이동에서는 /sorter_switch == 0을 발행한다.
    7. sorter는 파지 성공 후 고정 목적지 이동 직전에 정확히 한 번 발행한다.
    8. 회전 검사 경로에서는 다른 입력은 잠그되 /barcode_exist_L만
       검사 활성 구간 동안 계속 수신한다.
"""

from codebaro_standalone.bootstrap import bootstrap


def run():
    bootstrap()
    from codebaro_standalone.workflow import main, shutdown

    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt received.")
    finally:
        shutdown()


if __name__ == "__main__":
    run()
