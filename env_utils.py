import numpy as np
from collections import deque
from skimage import transform
from vizdoom import DoomGame


def create_environment():
    """
    deadly_corridor.cfg / deadly_corridor.wad 기반으로 Doom 환경 생성.
    """
    game = DoomGame()

    # 설정 파일 & 시나리오
    game.load_config("deadly_corridor.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")

    game.init()

    # 7개의 버튼 → one-hot action
    possible_actions = np.identity(7, dtype=int).tolist()

    return game, possible_actions


def preprocess_frame(frame):
    """
    frame: (H, W, C) 혹은 (H, W) numpy array (uint8)

    - 위/아래, 좌우 일부 crop
    - 0~1 normalize
    - (100,120)으로 resize
    """
    # frame.shape = (H, W, C) 또는 (H, W)
    # Doom 기본은 (3, H, W) 형태일 수도 있어서 필요시 transpose 해줘야 할 수 있음
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        # (C,H,W) → (H,W)
        frame = np.mean(frame, axis=0)

    # [Up:Down, Left:Right]
    cropped_frame = frame[15:-5, 20:-20]

    # Normalize
    normalized_frame = cropped_frame / 255.0

    # Resize to (100, 120)
    preprocessed_frame = transform.resize(normalized_frame, (100, 120))

    return preprocessed_frame  # (100,120)


def init_stacked_frames(stack_size):
    """
    0으로 된 프레임을 stack_size개 가진 deque 반환.
    """
    stacked_frames = deque(
        [np.zeros((100, 120), dtype=np.float32) for _ in range(stack_size)],
        maxlen=stack_size,
    )
    return stacked_frames


def stack_frames(stacked_frames, state, is_new_episode, stack_size=4):
    """
    state: raw frame (Doom screen buffer)
    stacked_frames: deque of frames
    is_new_episode: 새로운 에피소드 시작 여부
    """
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque(
            [np.zeros((100, 120), dtype=np.float32) for _ in range(stack_size)],
            maxlen=stack_size,
        )
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2)  # (100,120,4)

    return stacked_state, stacked_frames
