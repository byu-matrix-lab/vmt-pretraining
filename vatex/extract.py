import os
import json

data_folder = '../../compute/vatex_baseline/Video-guided-Machine-Translation/results/mask_30_rand/'

outputf = data_folder + 'submission.json'

compf = '../../compute/vatex_baseline/Video-guided-Machine-Translation/data/mask_30_rand_vatex_test.json'

with open(outputf, 'r') as file:
    data = json.load(file)

with open(compf, 'r') as file:
    targs = json.load(file)

sub = {}
for item in targs:
    sub[item['videoID']] = item
targs = sub
del sub

srcs = []
tgts = []
preds = []
for k, v in data.items():
    video = k[:-2]
    ind = int(k[-1]) + 5
    comp = targs[video]
    tgt = comp['chCap'][ind]
    src = comp['enCap'][ind]
    v = v.replace(' ','') # do this or no?
    # print(video, ind, v)
    srcs.append(src)
    tgts.append(tgt)
    preds.append(v)
    # break

with open(data_folder + 'src.txt', 'w', encoding='utf-8') as file:
    for line in srcs:
        file.write(line + '\n')

with open(data_folder + 'tgt.txt', 'w', encoding='utf-8') as file:
    for line in tgts:
        file.write(line + '\n')

with open(data_folder + 'preds.txt', 'w', encoding='utf-8') as file:
    for line in preds:
        file.write(line + '\n')
