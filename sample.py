import h5py
import os
import json
import shutil

# MAD
movies = set(['10142']) # limit movies others: '9952'

with h5py.File('data/mad/CLIP_L14_frames_features_5fps.h5', 'r') as all_movies, \
    h5py.File('sample_data/mad/CLIP_L14_frames_features_5fps.h5', 'w') as sub_movies:

    for key in movies:
        sub_movies.create_dataset(key, data=all_movies[key][:])

# Check the new file to see that it exists
with h5py.File('sample_data/mad/CLIP_L14_frames_features_5fps.h5', 'r') as sub_movies:
    for key in sub_movies.keys():
        print(key, sub_movies[key][:].shape)

for file in os.listdir('data/mad/'):
    if 'filtered' not in file:
        continue

    out_data = []
    with open('data/mad/' + file, 'r') as jifile:
        for line in jifile:
            sub = json.loads(line)
            if sub['movie'] in movies:
                out_data.append(sub)
    
    with open('sample_data/mad/' + file, 'w') as jofile:
        for sub in out_data:
            jofile.write(json.dumps(sub, ensure_ascii=False) + '\n')


# VATEX

youtube = set(['00cwEcZZcu4_000003_000013', '00EyMeYimqo_000036_000046', '00MGOdCfLGM_000066_000076', \
        '00RS_1vCHm4_000007_000017', '0_19Dr5THso_000149_000159', '01BFInmg3Zs_000001_000011', \
        '01fAWEHzudA_000002_000012', '01UETgu-H60_000035_000045', '01yY6LqwUC0_000000_000010', \
        '01zAPiCvRJU_000038_000048', 'hJA9pbSKLyY_000046_000056', '7pCkjsvH3gQ_000030_000040'])

# copy video files
for group in ['private_test', 'public_test', 'val']:
    group_path = 'data/vatex/' + group + '/'

    for video in os.listdir(group_path):
        vid = video.split('.')[0]
        if vid not in youtube:
            continue
        
        file_path = group_path + video
        # print(file_path)
        shutil.copyfile(file_path, 'sample_' + file_path)

for file in os.listdir('data/vatex/'):
    if not file.endswith('json'):
        continue
    with open('data/vatex/' + file, 'r') as jifile:
        data = json.load(jifile)
    
    out_data = []
    for seg in data:
        if seg['videoID'] in youtube:
            out_data.append(seg)

    with open('sample_data/vatex/' + file, 'w') as jofile:
        json.dump(out_data, jofile, ensure_ascii=False)

# Tape and compress
os.system('tar -zcf sample_data.tar.gz sample_data/')
