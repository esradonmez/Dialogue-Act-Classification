import json
import numpy as np

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

# pad the input
def pad_features(data, max_len):
  # get the values
  x = []
  for k,v in data.items():
    print(v)
    x.append(v)
  
  # pad to the max_lenght
  zeros = np.zeros(13)
  padded = []
  for i in x:
    temp = i
    temp.append(((max_len - len(i)) * zeros))
    padded.append(temp)
  
  return padded