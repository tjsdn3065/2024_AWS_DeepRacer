def reward_function(params):
    speed = params['speed']
    is_offtrack = params['is_offtrack']
    is_reversed = params['is_reversed']
    is_crashed = params['is_crashed']

    # 역주행 또는 트랙 이탈 시 최소 보상 반환
    if is_reversed or is_offtrack or is_crashed:
        return float(0)

    return float(speed)