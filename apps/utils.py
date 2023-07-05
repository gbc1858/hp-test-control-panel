import os
import json

APPS_PATH = os.path.abspath(os.path.dirname(__file__))

STORE_JSON_PATH = os.path.join(APPS_PATH, 'store.json')
DATA_PROCESS_JSON_PATH = os.path.join(APPS_PATH, 'data_process_temp.json')


def get_store():
    try:
        with open(STORE_JSON_PATH, encoding='utf8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def update_store(dic):
    data = get_store()

    for k, v in dic.items():
        data[k] = v
    
    with open(STORE_JSON_PATH, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)


def get_data_process_store():
    try:
        with open(DATA_PROCESS_JSON_PATH, encoding='utf8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def update_data_process_store(dic):
    data = get_data_process_store()

    for k, v in dic.items():
        data[k] = v

    with open(DATA_PROCESS_JSON_PATH, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)
