from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import json
import os

root = '/mount/studenten/dialog-system/2021/teams/mauldasch/data/audios'
output = {}
for audio in os.listdir(root):
    path = os.path.join(root,audio)
    (rate,sig) = wav.read(path)
    mfcc_feat = mfcc(sig,rate)
    output[audio] = mfcc_feat.tolist()
print("number of audios:", len(output))
with open('audio_feat.json', 'w', encoding='utf-8') as f:
    json.dump(output, f)
print("finishing")
with open('audio_feat.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(len(data))