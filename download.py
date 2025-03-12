import json
import random
import os
import multiprocessing
import tools.zip_data as zip_data
import argparse
import cv2

parser = argparse.ArgumentParser(description='Download and process videos.')
parser.add_argument('--sample_num', type=int, default=32, help='Number of samples to download')
parser.add_argument('--worker_count', type=int, default=16, help='Number of worker processes')
args = parser.parse_args()

index = json.load(open('data/all_7xx_Apr_6.json'))
SAMPLE_NUM = args.sample_num

base_dir, rel_paths = index['basedir'], index['relpaths']

random.seed(1001)
rel_paths = random.sample(rel_paths, SAMPLE_NUM)

os.makedirs('data/videos', exist_ok=True)
os.makedirs('data/jsonl', exist_ok=True)
os.makedirs('data/zip', exist_ok=True)

def download_and_zip(rel_path):
    path = rel_path[:-4]
    unique_id = path.split('/')[-1]
    video_path = os.path.join(base_dir, path + '.mp4')
    jsonl_path = os.path.join(base_dir, path + '.jsonl')
    os.system(f'wget {video_path} -O data/videos/{unique_id}.mp4')
    # check if the video file is not broken
    cap = cv2.VideoCapture(f'data/videos/{unique_id}.mp4')
    if not cap.isOpened():
        print(f'Error: {video_path} is broken')
        os.system(f'rm data/videos/{unique_id}.mp4')
        return
    os.system(f'wget {jsonl_path} -O data/jsonl/{unique_id}.jsonl')
    # zip the jsonl file, only keep useful information
    zip_data.zip_file(f'data/jsonl/{unique_id}.jsonl', f'data/zip/{unique_id}.jsonl')

worker_count = args.worker_count

with multiprocessing.Pool(worker_count) as pool:
    pool.map(download_and_zip, rel_paths)