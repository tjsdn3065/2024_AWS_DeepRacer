from math import cos,sin,pi,sqrt,atan2, degrees
import numpy as np

def reward_function(params):

    # Read input parameters
    x = params['x'] # float. 트랙이 포함된 시뮬레이션 환경에서 x축과 y축에 따른 agent의 중앙 위치(m)를 나타냄.
    y = params['y']
    track_width = params['track_width'] # float. 트랙의 너비(m). 트랙의 흰색 경계선 사이 거리를 의미
    distance_from_center = params['distance_from_center'] # float. 0~track_width/2. Agent의 중앙과 트랙의 중앙 사이의 거리(m). Agent의 한 쪽 바퀴가 트랙의 흰색 경계선을 벗어났을 때 최댓값.
    speed = params['speed'] # float. 0.0~4.0. m/s
    steps = params['steps'] # int. 완료한 step의 개수를 나타냄. step은 agent가 state 정보를 입력 받고 현재의 policy에 따라 취할 하나의 행동을 결정하는 일련의 과정 한 번을 의미. 참고로 딥레이서에서 보상함수에 적용되는 첫번째 step은 2부터 시작.
    is_offtrack = params['is_offtrack'] # Boolean, Agent가 종료 시점에 트랙을 벗어났으면 True, 벗어나지 않았으면 False를 갖는 변수. 이 값이 True면 해당 에피소드는 종료.
    progress = params['progress'] # float. 0~100. 트랙을 얼만큼 주행했는지를 퍼센트로 나타냄.
    all_wheels_on_track = params['all_wheels_on_track'] # Boolean. Agent의 트랙 이탈 여부를 확인하는 Boolean값. 바퀴가 하나라도 트랙의 흰색 경계선을 벗어나면 False.
    is_left_of_center = params['is_left_of_center'] # Boolean. Agent가 트랙 중앙에서 왼쪽에 있으면 True, 오른쪽에 있으면 False.
    waypoints = params['waypoints'] # [[float,float],[float,float],....]. 트랙 중앙을 따라 정해진 waypoint의 위치를 순서대로 나열한 리스트. 순환 트랙의 경우 시작과 끝 지점의 waypoint는 같음.
    closest_waypoints = params['closest_waypoints'] # [int,int]. Agent의 현재 위치에서 가장 가깝게 위치한 두 waypoint의 index를 나타냄. Agent와의 유클리드 거리 계산법으로 구함. 첫 번째 값은 agent의 뒤에서 가장 가까운 waypoint를 나타내고, 두 번째 값은 agent의 앞에서 가장 가까운 waypoint를 나타냄. 가까운 waypoint를 찾을 때의 기준은 앞바퀴 위치.
    heading = params['heading'] # float, -180~180. 트랙의 x축에 대한 agent의 진행 방향(각도)를 나타냄.
    is_reversed = params['is_reversed'] # Boolean. Agent의 바퀴가 시계 방향으로 주행하면 True, 반시계 방향으로 주행하면 False.
    current_steering_angle = params['steering_angle'] # float. -30~30.

    SPEED_THRESHOLD_straight = 3.0  # 직진 코스에서 속도 기준
    SPEED_THRESHOLD_curve = 2.0  # 곡선 코스에서 속도 기준
    DIRECTION_THRESHOLD = 3.0

    reward = 0.0 # 보상
    minimum_reward = 1e-3

    vehicle_width = 0.107
    vehicle_length = 0.235
    lfd = 0.6
    is_look_ahead_point = False
    look_ahead_point = None

    radian_heading = heading * pi / 180

    expect_time = 10.0  # 원하는 시간
    expect_steps = 145  # 원하는 스텝

    is_correct_heading = False
    is_correct_distance = False
    is_correct_speed = False
    is_correct_position = False
    is_correct_steering = False

    optimal_path = [
        (0.546942781193672, 2.646942781193672, 0, 0),
        (0.7213277658761738, 2.2864970997890772, 0, 0),
        (0.9727440448917937, 1.9749178598796095, 0, 0),
        (1.2958513964669303, 1.7384404663923037, 0, 0),
        (1.6633361142279943, 1.5794191718422295, 1, 0),
        (2.048928660610384, 1.4714686318681485, 1, 0),
        (2.4391508896210152, 1.3816787289099692, 1, 0),
        (2.8263986456075902, 1.2797087232376851, 1, 0),
        (3.200713228233672, 1.1377882746012977, 1, 0),
        (3.55342354550835, 0.9480658721038812, 1, 0),
        (3.901138175524185, 0.7493824313356614, 1, 0),
        (4.269643234447392, 0.5928826296080705, 0, 0),
        (4.66676852613121, 0.5446530757720666, 0, 0),
        (5.060134279892209, 0.6211393468360495, 0, 0),
        (5.4278848797949895, 0.7792282161527238, 1, 0),
        (5.76646680324789, 0.9930703224574187, 1, 0),
        (6.0745097632435, 1.2488456670494454, 1, 0),
        (6.351341716251018, 1.538163919442136, 1, 0),
        (6.6006572306238604, 1.851493973086848, 1, 0),
        (6.830371882767349, 2.1794774479351995, 1, 0),
        (7.049690584264791, 2.5145036615198344, 1, 0),
        (7.259673181231572, 2.8554454552824966, 1, 0),
        (7.452239231492758, 3.2065272663115283, 1, 0),
        (7.62102814146515, 3.569620945956636, 1, 0),
        (7.7617415897057995, 3.944531518182296, 1, 0),
        (7.866130285167188, 4.331093255025442, 1, 0),
        (7.913444018896524, 4.72876381341105, 0, 0),
        (7.858225104101608, 5.126006400244746, 0, 0),
        (7.64802895532069, 5.464182280939867, 0, 0),
        (7.314968436163228, 5.688575498890277, 0, 0),
        (6.934858654546222, 5.813740571803263, 0, 0),
        (6.536481730025649, 5.8552952408356305, 0, 0),
        (6.138802076613392, 5.810101738332164, 0, 0),
        (5.759220227761692, 5.682572578765676, 1, 0),
        (5.406227925919514, 5.493532831908302, 1, 0),
        (5.077436319179341, 5.264977871542469, 1, 0),
        (4.764392191041476, 5.0152858551284485, 1, 0),
        (4.453946755105928, 4.762373129508327, 1, 0),
        (4.130828512210158, 4.525821489212927, 1, 1),
        (3.7806711517033693, 4.331500143752449, 1, 1),
        (3.401599148935731, 4.203287026407926, 1, 1),
        (3.007728280279531, 4.130447855710442, 1, 1),
        (2.6099881328543177, 4.084343270277187, 1, 1),
        (2.2120136747242714, 4.040108398454938, 1, 0),
        (1.8172624853713786, 3.972670994490456, 1, 0),
        (1.4348968524634207, 3.8542223042732924, 0, 0),
        (1.0850048237375602, 3.6584763863496415, 0, 0),
        (0.800546369782337, 3.3790129889954623, 0, 0),
        (0.5808430906019764, 3.0399891236094505, 0, 0)
    ]

    if not is_reversed: # 차량이 반시계 방향으로 주행하고 있다면
        # 🔹 **가장 가까운 최적 경로점 찾기**
        min_dist = float("inf")
        closest_index = 0

        for i, point in enumerate(optimal_path):
            dist = sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_index = i

        # 다음 점을 경로 순환에 맞게 처리
        next_index = (closest_index + 1) % len(optimal_path)  # 순환 경로 처리

        # 현재 위치와 그 다음 점을 연결하는 직선과의 거리 계산
        point1 = optimal_path[closest_index]
        point2 = optimal_path[next_index]
        x1, y1 = point1
        x2, y2 = point2

        # 최적 경로 벡터 V (Global 좌표계)
        vector_global = np.array([x2 - x1, y2 - y1])  # [dx, dy]
        rotation_matrix = np.array([
            [cos(radian_heading), sin(radian_heading)],
            [-sin(radian_heading), cos(radian_heading)]
        ])
        # 로컬 좌표계로 변환된 벡터
        vector_local = rotation_matrix.dot(vector_global)
        # 벡터 (1,0)과의 각도 오차 계산
        vector_angle = atan2(vector_local[1], vector_local[0])  # 로컬 벡터의 방향
        heading_error = abs(degrees(vector_angle))  # 도(°) 단위로 변환

        # heading 오차가 5도 이하인지 확인
        is_correct_heading = heading_error <= 5

        if is_correct_heading:
            reward += 5  # 트랙 방향과 정렬이 잘 맞을수록 보상 증가

        # 점과 직선의 거리 계산
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distance_to_line = numerator / denominator

        # 최적 경로와 가까운지 확인
        is_correct_distance = distance_to_line <= vehicle_width / 2

        if is_correct_distance:
            reward += 5  # 최적 경로와의 거리 유지

        # 속도가 적절한지 확인
        if optimal_path[closest_index][2] == 1:  # 직선 구간
            is_correct_speed = speed >= SPEED_THRESHOLD_straight
        else:  # 곡선 구간
            is_correct_speed = speed <= SPEED_THRESHOLD_curve

        if is_correct_speed:
            reward += 5  # 구간별 적절한 속도를 유지

        # 차량 위치가 적절한지 확인
        is_correct_position = (optimal_path[closest_index][3] == 0 and is_left_of_center) or \
                              (optimal_path[closest_index][3] == 1 and not is_left_of_center)

        if is_correct_position:
            reward += 3  # 트랙의 올바른 방향(왼쪽/오른쪽) 유지

        # 🔹 **look-ahead point 찾기**
        for i in range(closest_index, len(optimal_path)):
            point = optimal_path[i]
            dist = sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)

            if dist > lfd:
                look_ahead_point = point
                is_look_ahead_point = True
                break

        # 조향 각도 오차가 작은지 확인
        if is_look_ahead_point:
            t = np.array([
                [cos(radian_heading), -sin(radian_heading), x],
                [sin(radian_heading), cos(radian_heading), y],
                [0, 0, 1]
            ])

            det_t = np.array([
                [t[0][0], t[1][0], -(t[0][0] * x + t[1][0] * y)],
                [t[0][1], t[1][1], -(t[0][1] * x + t[1][1] * y)],
                [0, 0, 1]
            ])
            global_look_ahead_point = [look_ahead_point[0], look_ahead_point[1], 1]
            local_look_ahead_point = det_t.dot(global_look_ahead_point)
            theta = atan2(local_look_ahead_point[1], local_look_ahead_point[0])
            target_steering_angle = atan2(2 * vehicle_length * sin(theta), lfd) * 180 / pi * 1 / 6  # -30~30 정규화
            steering_angle_error = abs(target_steering_angle - current_steering_angle)

            is_correct_steering = steering_angle_error <= 5  # 조향 오차가 5° 이하

        if is_correct_steering:
            reward += 2  # 조향 오차가 적을수록 보상 증가

        # 5가지 조건을 모두 만족하면 추가 보상
        if is_correct_heading and is_correct_distance and is_correct_speed and is_correct_position and is_correct_steering:
            reward += 10  # 완벽한 주행을 하면 추가 보상

        # 50steps마다 더 큰 보상 -> 더 빠르게 학습하기 위해
        if (steps % 50) == 0 and progress >= (steps / expect_steps) * 100:
            reward += 30.0

        # 트랙 완주에 가까워질수록 더 큰 보상
        if progress == 100:  # 완주 시
            if steps < expect_time * 15:  # 기대 시간보다 15배 이내로 완주한 경우
                reward += 100 * (expect_time * 15 / steps)
            else:
                reward += 100

        elif is_offtrack:  # 트랙 이탈시
            reward -= 50

    return float(reward)