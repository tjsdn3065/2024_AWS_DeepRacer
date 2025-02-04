from math import cos, sin, pi, sqrt, atan2, degrees
import numpy as np

def reward_function(params):
    # Read input parameters
    x = params['x']
    y = params['y']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    steps = params['steps']
    is_offtrack = params['is_offtrack']
    progress = params['progress']
    all_wheels_on_track = params['all_wheels_on_track']
    is_left_of_center = params['is_left_of_center']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    is_reversed = params['is_reversed']
    current_steering_angle = params['steering_angle']

    # Constants and thresholds
    vehicle_width = 0.107
    vehicle_length = 0.235
    lfd = 0.6
    minimum_reward = 1e-3
    expect_time = 10.0  # desired time
    expect_steps = 145  # desired steps
    MAX_STEERING_ERROR = 10  # 최대 조향 오차 (°)
    MAX_OPTIMAL_DISTANCE = vehicle_width  # 최적 경로와의 최대 허용 거리
    SPEED_THRESHOLD_straight = 3.0  # 직진 코스에서 속도 기준
    SPEED_THRESHOLD_curve = 1.0  # 곡선 구간에서 속도 기준

    # 가중치 (각 항목별 보상 기여도)
    weight_heading = 2.0
    weight_steering = 2.0
    weight_distance = 2.0
    weight_speed = 2.0
    weight_position = 0.0

    # 보상 요소 초기화
    heading_reward = 0.0
    steering_reward = 0.0
    distance_reward = 0.0
    speed_reward = 0.0
    position_reward = 0.0

    # 현재 heading을 라디안으로 변환
    radian_heading = heading * pi / 180

    # 만약 역주행하거나 트랙을 이탈하면 최소 보상 반환
    if is_reversed or is_offtrack:
        return float(minimum_reward)

    # 최적 경로 (예시 데이터; 각 튜플: (x, y, 구간타입, 위치정보))
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

    # ★ 최적 경로 상에서 가장 가까운 점 찾기
    min_dist = float("inf")
    closest_index = 0
    for i, point in enumerate(optimal_path):
        dist = sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_index = i

    # 다음 점 (경로 선)을 선택 (순환 처리)
    next_index = (closest_index + 1) % len(optimal_path)
    point1 = optimal_path[closest_index]
    point2 = optimal_path[next_index]
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    # ★ 전역 좌표를 차량 로컬 좌표로 변환하기 위한 행렬 생성
    t = np.array([
        [cos(radian_heading), -sin(radian_heading), x],
        [sin(radian_heading), cos(radian_heading), y],
        [0, 0, 1]
    ])
    det_t = np.array([
        [t[0][0], t[1][0], -(t[0][0]*x + t[1][0]*y)],
        [t[0][1], t[1][1], -(t[0][1]*x + t[1][1]*y)],
        [0, 0, 1]
    ])

    # 최적 경로 상의 두 점을 로컬 좌표로 변환
    global_point1 = [x1, y1, 1]
    global_point2 = [x2, y2, 1]
    local_point1 = det_t.dot(global_point1)
    local_point2 = det_t.dot(global_point2)

    # ★ 경로 선과 차량 진행 방향(x축) 사이의 각도 오차 계산 (heading error)
    vector_angle = atan2(local_point2[1] - local_point1[1], local_point2[0] - local_point1[0])
    heading_error = abs(degrees(vector_angle))
    max_heading_error = 10.0
    heading_error_capped = min(heading_error, max_heading_error)
    heading_reward = max(minimum_reward, (1 - (heading_error_capped / max_heading_error)) * 10)

    # ★ 차량과 최적 경로 선 사이의 수직 거리 계산
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distance_to_line = numerator / denominator if denominator != 0 else float("inf")
    distance_error = min(distance_to_line, MAX_OPTIMAL_DISTANCE)
    distance_reward = max(minimum_reward, (1 - (distance_error / MAX_OPTIMAL_DISTANCE)) * 10)

    # ★ 속도 보상 (연속적 평가)
    speed_error = None
    if point1[2] == 1:  # 직선 구간
        target_speed = 4.0
        speed_tolerance = 1.0
        speed_error = min(abs(target_speed - speed), speed_tolerance)
        speed_reward = max(minimum_reward, (1 - (speed_error / speed_tolerance)) * 10)

    is_position_condition = False
    # ★ 위치 보상: optimal_path에 정의된 위치 정보와 차량의 상대적 위치 비교
    if (point1[3] == 0 and is_left_of_center) or (point1[3] == 1 and not is_left_of_center):
        position_reward = 10
        is_position_condition = True
    else:
        position_reward = minimum_reward

    # ★ 조향각(steering) 보상
    is_look_ahead_point = False
    look_ahead_point = None
    for i in range(closest_index, len(optimal_path)):
        point = optimal_path[i]
        dist = sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if dist > lfd:
            look_ahead_point = point
            is_look_ahead_point = True
            break

    if is_look_ahead_point:
        global_look_ahead_point = [look_ahead_point[0], look_ahead_point[1], 1]
        local_look_ahead_point = det_t.dot(global_look_ahead_point)
        theta = atan2(local_look_ahead_point[1], local_look_ahead_point[0])
        # pure pursuit 공식에 의한 연속 목표 조향각 (°)
        target_steering_angle = atan2(2 * vehicle_length * sin(theta), lfd) * 180 / pi
        steering_angle_error = min(abs(target_steering_angle - current_steering_angle), MAX_STEERING_ERROR)
        steering_reward = max(10 - steering_angle_error, minimum_reward)
    else:
        steering_reward = minimum_reward

    # ★ 개별 보상 항목들을 가중치 적용하여 합산
    total_reward = (weight_heading * heading_reward +
                    weight_steering * steering_reward +
                    weight_distance * distance_reward +
                    weight_speed * speed_reward +
                    weight_position * position_reward)

    # ★ 보너스: 모든 조건이 매우 우수하면 추가 보상
    if is_position_condition and heading_error < 1 and (speed_error is not None and speed_error < 0.2) and \
       (is_look_ahead_point and abs(target_steering_angle - current_steering_angle) < 1) and \
       distance_to_line < (MAX_OPTIMAL_DISTANCE / 5):
        total_reward += 200

    if progress > 80:
        total_reward += 20  # 트랙 진행률 80% 이상 추가 보상
    elif progress > 90:
        total_reward += 50  # 90% 이상 추가 보상
    elif progress == 100:
        total_reward += 200  # 완주 시 큰 보상

    # ★ 최적 경로에서 벗어난 정도에 따른 패널티 (제곱함수 적용)
    if distance_to_line <= MAX_OPTIMAL_DISTANCE:
        # 거리 비율에 대해 제곱 함수를 적용하여 부드럽게 감소하는 보상 계수 산출
        penalty_factor = (1 - (distance_to_line / MAX_OPTIMAL_DISTANCE)) ** 2
    else:
        penalty_factor = 0.1  # 최대 허용 거리보다 멀면 강한 패널티
    total_reward *= penalty_factor

    total_reward = max(total_reward, minimum_reward)

    return float(total_reward)