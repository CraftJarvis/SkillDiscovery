from process_video import *
from load_model import load_minecraft_model, load_idm_model
from lib.tree_util import tree_map

import torch
import os
import numpy as np
import argparse
import cv2

DEVICE = 'cuda'

window_size = 1
preview_len = 0
video_max_len = 12800
video_min_len = 2

def compute_boundary(video_path, json_path, agent, 
                     events_frame=[], idm_agent=None):
    
    GAP = 18 if json_path is not None else 17

    boundaries = []
    events_frame = sorted(list(events_frame))
    events_frame.append(1e9)    # add a large number to avoid empty list

    policy = agent.policy

    if json_path is not None:
        data = get_frame_and_action(video_path, json_path)
    else:
        assert idm_agent is not None
        data = get_frame_and_predict_action(video_path, idm_agent)

    agent_state = policy.initial_state(1)
    losses = []
    avg_loss = 0
    prev_id = 0
    prev_frame_id = 0
    flag = False
    dummy_first = torch.from_numpy(np.array((False,))).to(DEVICE)


    for id, datum in enumerate(data):
        frame_id, image, action, isGuiOpen = datum
        agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
        agent_obs = agent._env_obs_to_agent({"pov": image})
        pi_distribution, v_prediction, agent_state = policy.get_output_for_observation(
                        agent_obs,
                        agent_state,
                        dummy_first
                    ) 
        agent_state = tree_map(lambda x: x.detach(), agent_state)
      
        if (frame_id < events_frame[0]) and (id - prev_id <= preview_len):
            continue

        log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)
        discount = 0.75 if isGuiOpen else 1.0            # still need this?
        losses.append(-log_prob.item() * discount)
        if len(losses) > window_size:
            avg_loss = sum(losses)/len(losses)
            if losses[-1] - avg_loss > GAP:
                flag = True

        if id - prev_id >= video_min_len and (flag or (frame_id >= events_frame[0]) or id - prev_id >= video_max_len):
            if frame_id >= events_frame[0]:
                events_frame.pop(0)
            boundaries.append((prev_frame_id, frame_id - 1))         # (begin, end)
            agent_state = policy.initial_state(1)
            losses = []
            avg_loss = 0
            prev_id = id
            prev_frame_id = frame_id
            flag = False

    if len(data) > 0:
        boundaries.append((prev_frame_id, data[-1][0]))
    
    return boundaries
    
if __name__ == '__main__':
    # unique_id = 'gimpy-jade-panda-b0f731ae8a1b-20211231-054458'
    # video_path = os.path.join('./data/videos/', unique_id + '.mp4')
    # json_path = os.path.join('./data/jsonl/', unique_id + '.jsonl')

    parser = argparse.ArgumentParser(description="Split a video file into segments.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    args = parser.parse_args()
    video_path = args.video_path
    json_path = None

    model = './models/foundation-model-3x.model'
    weights = './weights/bc-early-game-3x.weights'
    agent = load_minecraft_model(model, weights)

    idm_model = './models/4x_idm.model'
    idm_weights = './weights/4x_idm.weights'
    idm_agent = load_idm_model(idm_model, idm_weights)

    boundaries = compute_boundary(video_path, json_path, agent, idm_agent=idm_agent)
    print(boundaries)

    # save the video clips according to the boundaries
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = f'./result/{video_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    for count, (begin, end) in enumerate(boundaries):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_dir, f'{count}.mp4'), fourcc, fps, (width, height))
        for i in range(begin, end + 1):
            out.write(frames[i])
        out.release()
    print(f"Video clips saved to {output_dir}")