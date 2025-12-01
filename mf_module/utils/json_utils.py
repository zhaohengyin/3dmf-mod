import json
import os

def dump_dict_to_json(data_dict, path):
    """
    Dump a Python dictionary into a JSON file.

    Args:
    - data_dict (dict): The dictionary to be dumped.
    - filename (str): The filename/path of the JSON file.

    Returns:
    - None
    """
    with open(path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file: {e}")
    
    return data

