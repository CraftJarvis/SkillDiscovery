import os
import json
import argparse
from typing import OrderedDict

bias = 16
VIDEO_MAX_LEN = 6000

stat_keep = ["craft_item", "mine_block", "kill_entity", "use_item"]
stat_bias = ["kill_entity"]

def list_file_events(jsonl_path:str):
    result = OrderedDict()
    file = open(jsonl_path, 'r', errors='ignore')
    old_event, event = set(), set()
    for i, line in reversed(list(enumerate(file))):
        if not event == set():
            old_event = event
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
        else:
            continue
        
        data = data.get("stats_change", {})
        bias_flag = False
        event = set()
        for stat in data.keys():
            if True in [(s in stat) for s in stat_keep]:
                event.add(stat)
            if True in [(s in stat) for s in stat_bias]:
                bias_flag = True
        if not event == set() and not old_event == event:
            id = min(i + bias * bias_flag, VIDEO_MAX_LEN)
            result[id] = event  
        
    return result

if __name__ == '__main__':
    dir = "./data/zip/"
    parser = argparse.ArgumentParser(description="Split video files into segments.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output JSONL file.')
    args = parser.parse_args()
    output_file = args.output_path
    for file in os.listdir(dir):
        file_len = len(list(open(os.path.join(dir, file), 'r', errors='ignore')))
        boundaries = list_file_events(os.path.join(dir, file))
        boundaries = [id for id, events in boundaries.items()][::-1]
        begins = [0] + boundaries
        ends = boundaries + [file_len + 1]
        ends = [i-1 for i in ends]
        if file_len > 0:
            boundaries = list(zip(begins, ends))
        else:
            boundaries = []
        with open(output_file, 'a') as output:
            json.dump({"file": file, "boundaries": boundaries}, output)
            output.write('\n')