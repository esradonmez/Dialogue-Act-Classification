import json

# load mfcc features
file_path = "/mount/studenten/dialog-system/2021/teams/mauldasch/features/audio_feat.json"  
def read_data(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file) 
    return data

# get max length
def get_max_length(data):
    max = 0
    for k, v in data.items():
        if len(v) > max:
            max = len(v)
    return max