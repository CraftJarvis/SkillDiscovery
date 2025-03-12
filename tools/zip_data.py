import json
import os

stat_filter = ["minecraft.custom:minecraft.time_since_death", 
        "minecraft.custom:minecraft.play_one_minute", 
        "minecraft.custom:minecraft.time_since_rest",
        "minecraft.custom:minecraft.walk_one_cm",
        "minecraft.custom:minecraft.sprint_one_cm",
        "minecraft.custom:minecraft.fly_one_cm"] 

key_keep = ["isGuiOpen", "isGuiInventory"]

def inventory_to_dict(inventory: list):
    result = {}
    for item in inventory:
        type, quantity = item["type"], item["quantity"]
        result[type] = quantity
    return result

def zip_stat(stat: str): # from minecraft.TYPE:minecraft.ACTION to TYPE:ACTION
    TYPE, ACTION = stat.split(":")
    TYPE = TYPE.split(".")[1]
    ACTION = ACTION.split(".")[1]
    return f"{TYPE}:{ACTION}"

def zip_key(key: str):
    return key[13:]

def process_mouse(button: int):
    return "left" if button == 0 else "right"


def zip_file(path:str, result_path:str):
    if os.path.exists(result_path):
        return
    file = open(path, 'r', errors='ignore')
    new_file = open(result_path, 'w')
    old_data, data = None, None
    for line in file:
        old_data = data
        line = line.strip()
        if line:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
        else:
            continue
        if not old_data:
            continue
        
        zipped_data = {}
        zipped_data["mouse_button"] = [process_mouse(k) for k in data["mouse"]["buttons"]]
        zipped_data["keyboard_keys"] = [zip_key(k) for k in data["keyboard"]["keys"]]
        
        for key, value in data.items():
            if key == "stats":
                old_value = old_data[key]
                if not (old_value == {}):
                    zipped_value = {}
                    for stat, cnt in value.items():
                        if stat in stat_filter: continue
                        old_cnt = old_value.get(stat, 0)
                        if cnt > old_cnt:
                            zipped_value[zip_stat(stat)] = cnt - old_cnt 
                    zipped_data["stats_change"] = zipped_value
                else:
                    zipped_data["stats_change"] = {}
            elif key == "inventory":
                value = inventory_to_dict(value)
                old_value = inventory_to_dict(old_data[key])
                zipped_value = {}
                for type, quantity in value.items():
                    old_quantity = old_value.get(type, 0)
                    if quantity != old_quantity:
                        zipped_value[type] = quantity - old_quantity
                zipped_data["inventory_change"] = zipped_value
            elif key in key_keep:
                zipped_data[key] = value
        
        json_str = json.dumps(zipped_data)
        new_file.write(json_str + '\n')
