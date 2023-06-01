
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import h5py

# import h5py

# special token values
PAD = 0
UNK = 1
BOS = 2
EOS = 3

RANDOM_SEED = 42

# Dataloader parameters
DATA_DIR = '../../data/sample_data/' # replace this
MAX_VIDEO_LEN = 300
MAX_TEXT_LEN = 300
MAD_FILES = ['filtered_comet.txt']
VATEX_TRAIN_FILES = ['vatex_training_v1.0.json']
VATEX_VAL_FILES = ['new_vatex_validation.json']
# VATEX_TRAIN_FILES = ['mask_vatex_train.json']
# VATEX_VAL_FILES = ['mask_vatex_validation.json']

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def generate_mask(size, lens):
  ans = torch.ones((len(lens), size)) # TODO: dtype?
  for i, l in enumerate(lens):
    ans[i,l:] = 0
  return ans

# Dataloaders
def pad_to_longest(batch):
  videos, src, tgt = zip(*batch)

  # pad videos
  vid_lens = [len(v) for v in videos]
  pad_len = max(vid_lens)
  vid_mask = generate_mask(pad_len, vid_lens)
  pad_videos = []
  emb_size = len(videos[0][0])
  for v in videos:
    v = v.tolist()
    if len(v) < pad_len: # pad with zeros
      v += [[0] * emb_size] * (pad_len - len(v)) # careful of correferences
    pad_videos.append(v)

  src_lens = [len(s) for s in src]
  pad_len = max(src_lens)
  src_mask = generate_mask(pad_len, src_lens)
  pad_src = [s + [PAD] * (pad_len - len(s)) for s in src]

  tgt_lens = [len(s) for s in tgt]
  pad_len = max(tgt_lens)
  tgt_mask = generate_mask(pad_len, tgt_lens)
  pad_tgt = [s + [PAD] * (pad_len - len(s)) for s in tgt]

  pad_videos = torch.tensor(pad_videos)
  pad_src = torch.tensor(pad_src)
  pad_tgt = torch.tensor(pad_tgt)

  return pad_videos, vid_mask, pad_src, src_mask, pad_tgt, tgt_mask

class MADDataset(Dataset):
  def __init__(self, files, tokenizer):
    files = [DATA_DIR + 'mad/' + f for f in files]

    with h5py.File(DATA_DIR + '/mad/CLIP_L14_frames_features_5fps.h5', 'r') as all_movies:
      movie_data = {}
      for key in tqdm(all_movies.keys(), desc = 'Loading movie features'):
        movie_data[key] = all_movies[key][:]

    self.data = []

    for file in files:
      with open(file, 'r') as labels:
        for line in tqdm(labels.readlines(), desc='Loading json file ' + file.split('/')[-1]):
          label_data = json.loads(line)
          
          start_time, end_time = map(lambda x: int(5*x), label_data['ext_timestamps'])
          movie_features = movie_data[label_data['movie']][start_time:end_time]

          en = label_data['en_sentence']
          zh = label_data['zh_sentence']

          en = [tokenizer.bos_id()] + tokenizer.encode(en) + [tokenizer.eos_id()]
          zh = [tokenizer.bos_id()] + tokenizer.encode(zh) + [tokenizer.eos_id()]

          self.data.append((movie_features, en, zh))

  def __getitem__(self, i):
    return self.data[i]
  
  def __len__(self):
    return len(self.data)

class VaTeXDataset(Dataset):
  def __init__(self, files, tokenizer, train=True):
    files = [DATA_DIR + 'vatex/' + f for f in files]

    # video_path = DATA_DIR + 'vatex/' + ('val/' if train else 'private_test/')
    video_path = DATA_DIR + 'vatex/val/'

    self.data = []

    for file in files:
      with open(file, 'r') as labels:
        raw_data = json.load(labels)
        for line in tqdm(raw_data, desc='Loading json file'):
          videoid = line['videoID']
          
          video_feats = np.load(video_path + videoid + '.npy')[0]

          video_feats = video_feats[::5]

          en = line['enCap'][-5:]
          zh = line['chCap'][-5:]

          en = [[tokenizer.bos_id()] + sent + [tokenizer.eos_id()] for sent in tokenizer.encode(en)]
          zh = [[tokenizer.bos_id()] + sent + [tokenizer.eos_id()] for sent in tokenizer.encode(zh)]

          pairs = list(zip(en, zh))

          # if train:
          #   self.data.append((video_feats, pairs))
          # else:
          for pair in pairs: # no random for test
            self.data.append((video_feats, [pair]))
          # self.data.append((video_feats, pairs[:1]))

  def __getitem__(self, i):
    video, pairs = self.data[i]
    en, zh = random.choice(pairs)
    return video, en, zh
  
  def __len__(self):
    return len(self.data)

