import json
import os
from tqdm import tqdm

coms = []

for file in os.listdir('data'):
    if file.endswith('json'):
        file = 'data/' + file
        with open(file, 'r') as f:
            data = json.load(f)
            for item in tqdm(data):
                id = item['videoID']
                vid = id[:-14]
                start = int(id[-13:-7])
                end = int(id[-6:])
                
                command = f'\'./data/yt-dlp -f mp4 https://www.youtube.com/watch?v={vid} -o data/raw/{vid}.mp4 --download-sections "*{start}-{end}" > /dev/null\''
                coms.append(command)

with open('data/download.sh', 'w') as f:
    for v in coms:
        f.write(v + '\n')