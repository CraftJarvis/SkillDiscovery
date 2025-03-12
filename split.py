import ray
import os
import random
import json
import datetime
import argparse

from segmentation import compute_boundary
from tools.list_events import list_file_events
from load_model import load_minecraft_model

@ray.remote(num_gpus=0.5)
def split_file_raw(file, dir, agent):
    if not file.endswith(".mp4"): return
    unique_id = file.split('.')[0]
    video_path = os.path.join(dir, 'videos/', file)
    json_path = os.path.join(dir, 'jsonl/', unique_id + '.jsonl')
    boundaries = compute_boundary(video_path, json_path, agent)
    result = {"file": file, "boundaries": boundaries}
    return result

@ray.remote(num_gpus=0.5)
def split_file_events(file, dir, agent):
    if not file.endswith(".mp4"): return
    unique_id = file.split('.')[0]
    video_path = os.path.join(dir, 'videos/', file)
    json_path = os.path.join(dir, 'jsonl/', unique_id + '.jsonl')
    events = list_file_events(os.path.join(dir, 'zip/', unique_id + '.jsonl'))
    boundaries = compute_boundary(video_path, json_path, agent, events.keys())

    result = {"file": file, "boundaries": boundaries}
    return result

def split_files(split_fn, dir: str, output_path: str, model: str, weights: str, sample_num: int = 0):
    video_files = sorted(os.listdir(os.path.join(dir, 'videos/')))
    if sample_num > 0 and sample_num < len(video_files):
        video_files = random.sample(video_files, sample_num)

    # skip all existing files
    if os.path.exists(output_path):
        with open(output_path, 'r') as output_file:
            for line in output_file:
                data = json.loads(line)
                video_files.remove(data['file']) 
    print(f"{len(video_files)} files to process")

    agent = load_minecraft_model(model, weights)
    agent_ref = ray.put(agent)

    futures = [split_fn.remote(file, dir, agent_ref) for file in video_files]

    with open(output_path, 'a') as output_file:
        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=60)
            for future in done:
                result = ray.get(future)
                if result:
                    json.dump(result, output_file)
                    output_file.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video files into segments.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--info', action='store_true', help='Use events info if specified, otherwise use raw split.')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use.')
    parser.add_argument('--num_cpus', type=int, default=32, help='Number of CPUs to use.')
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    dir = './data/'
    model = './models/foundation-model-3x.model'
    weights = './weights/bc-early-game-3x.weights'
    
    split_fn = split_file_events if args.info else split_file_raw
    split_files(split_fn, dir, args.output_path, model, weights)
    
    ray.shutdown()
    print("finished at ", datetime.datetime.now())