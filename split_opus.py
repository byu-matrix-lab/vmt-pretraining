import random
from tqdm import tqdm

working_dir = '../compute/data/opus/OpenSubtitles/'

path1 = working_dir + 'OpenSubtitles.en-zh_cn.en'
path2 = working_dir + 'OpenSubtitles.en-zh_cn.zh_cn'

def save(path, data):
    path = working_dir + path

    with open(path, 'w') as f:
        for l1, l2 in tqdm(data):
            l1 = l1.strip()
            l2 = l2.strip()
            f.write(f'{l1}\t{l2}\n')

with open(path1, 'r') as f1, open(path2, 'r') as f2:
    data = list(tqdm(zip(f1, f2)))
    
    random.shuffle(data)

    split = int(len(data) * 0.95)

    train = data[:split]
    val = data[split:]

    save('train.tsv', train)
    save('val.tsv', val)

