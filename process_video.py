from run_inverse_dynamics_model import json_action_to_env_action
from data_loader import composite_images_with_alpha
from agent import resize_image, AGENT_RESOLUTION

import cv2
import numpy as np
import json
import os

CURSOR_FILE = os.path.join(os.path.dirname(__file__), "cursors", "mouse_cursor_white_16x16.png")

MINEREC_ORIGINAL_HEIGHT_PX = 720

N_BATCHES = 32

def _check_op(action):
        # return True if there is action, False if no-op
        key_filter = ['ESC', 'swapHands', 'pickItem']
        flag = False
        for key, val in action.items():
            if key in key_filter:
                continue
            if not np.all(val == 0):
                flag = True
                break
        return flag

def get_frame_and_action(video_path, json_path):

    result = []

    cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
    # Assume 16x16
    cursor_image = cursor_image[:16, :16, :]
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_image = cursor_image[:, :, :3]

    video = cv2.VideoCapture(video_path)
    # NOTE: In some recordings, the game seems to start
    #       with attack always down from the beginning, which
    #       is stuck down until player actually presses attack
    # NOTE: It is uncertain if this was the issue with the original code.
    attack_is_stuck = False
    # Scrollwheel is allowed way to change items, but this is
    # not captured by the recorder.
    # Work around this by keeping track of selected hotbar item
    # and updating "hotbar.#" actions when hotbar selection changes.
    # NOTE: It is uncertain is this was/is an issue with the contractor data
    last_hotbar = 0

    with open(json_path) as json_file:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)
    for i in range(len(json_data)):
        step_data = json_data[i]

        if i == 0:
            # Check if attack will be stuck down
            if step_data["mouse"]["newButtons"] == [0]:
                attack_is_stuck = True
        elif attack_is_stuck:
            # Check if we press attack down, then it might not be stuck
            if 0 in step_data["mouse"]["newButtons"]:
                attack_is_stuck = False
        # If still stuck, remove the action
        if attack_is_stuck:
            step_data["mouse"]["buttons"] = [button for button in step_data["mouse"]["buttons"] if button != 0]

        # this is the action that will be sent to the agent
        action, _ = json_action_to_env_action(step_data)
        # this is the action to check if the action is null
        action_op, _ = json_action_to_env_action(step_data, predicted=False)

        # Update hotbar selection
        current_hotbar = step_data["hotbar"]
        if current_hotbar != last_hotbar:
            action["hotbar.{}".format(current_hotbar + 1)] = 1
        last_hotbar = current_hotbar

        # Read frame even if this is null so we progress forward
        ret, frame = video.read()
        isGuiOpen = False
        if ret:
            # Skip null actions as done in the VPT paper
            # NOTE: in VPT paper, this was checked _after_ transforming into agent's action-space.
            #       We do this here as well to reduce amount of data sent over.
            if not _check_op(action_op):
                continue
            if step_data["isGuiOpen"]:
                isGuiOpen = True
                camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            frame = resize_image(frame, AGENT_RESOLUTION)
            result.append((i, frame, action, isGuiOpen))
        else:
            print(f"Could not read frame from video {video_path}")
            break
    
    video.release()

    return result

def get_frame_and_predict_action(video_path, agent):
    result = []

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[..., ::-1])

    for start in range(0, len(frames), N_BATCHES):
        end = start + N_BATCHES
        batch_frames = np.stack(frames[start:end])
        predicted_actions = agent.predict_actions(batch_frames)

        for i in range(len(batch_frames)):
            frame = batch_frames[i]
            action = {key: val[0][i] for key, val in predicted_actions.items()}
            if _check_op(action):
                result.append((start + i, frame, action, False))
    
    return result

if __name__ == '__main__':
    unique_id = 'hazy-thistle-chipmunk-f153ac423f61-20220121-162611'
    video_path = os.path.join('./data/videos/', unique_id + '.mp4')
    json_path = os.path.join('./data/jsonl/', unique_id + '.jsonl')
    result = get_frame_and_action(video_path, json_path)