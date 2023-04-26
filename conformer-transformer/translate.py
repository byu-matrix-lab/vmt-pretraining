import argparse
import os
import sys
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from conformertransformer import *

# special token values
PAD = 0
UNK = 1
BOS = 2
EOS = 3

MAX_LEN = 60

RANDOM_SEED = 42

MODEL_DIR = '../../compute/models/conformer-transformer-vatex/dmodel_run1/'
MODEL_SAVE = 'best_model-8.pt'

# Dataloader parameters
DATA_DIR = '../../compute/data/'
MAX_VIDEO_LEN = 300
MAX_TEXT_LEN = 300
TOKEN_DIR = '../../compute/data/tokenizing/'
# MAD_FILES = ['filtered_comet.txt']
# VATEX_TRAIN_FILES = ['vatex_training_v1.0.json']
VATEX_VAL_FILES = ['new_vatex_validation.json']
VATEX_TEST_FILES = ['new_vatex_test.json']
# VATEX_TEST_FILES = ['mask_vatex_test.json']

# Model parameters
EPOCHS = 2000
VALIDATION_EPOCHS = 10
SAVE_EVERY = 10
EARLY_STOPPING = 10 # increase this a lot?

SOURCE_VOCAB_SIZE = 16000
TARGET_VOCAB_SIZE = 8000
BATCH_SIZE = 64
NUMBER_OF_LAYERS = 6
VIDEO_EMB_SIZE=1024 # 768 for MAD, 1024 for VATEX
LEARNING_RATE = 1e-4
SMOOTHING = 0.1

TRANS_DIR = MODEL_DIR + 'output/'

if not os.path.exists(TRANS_DIR):
    os.makedirs(TRANS_DIR)

# Set device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')
print('Using device', device)

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load sentencepiece models
sp_en = spm.SentencePieceProcessor(model_file=TOKEN_DIR + 'en.model')
sp_zh = spm.SentencePieceProcessor(model_file=TOKEN_DIR + 'zh.model')

# Dataloaders
def pad_to_longest(batch):
  videos, src, tgt = zip(*batch)

  # pad videos
  pad_len = max(len(v) for v in videos)
  pad_videos = []
  emb_size = len(videos[0][0])
  for v in videos:
    v = v.tolist()
    if len(v) < pad_len: # pad with zeros
      v += [[0] * emb_size] * (pad_len - len(v)) # careful of correferences
    pad_videos.append(v)

  pad_len = max(len(s) for s in src)
  pad_src = [s + [sp_en.pad_id()] * (pad_len - len(s)) for s in src]

  pad_len = max(len(s) for s in tgt)
  pad_tgt = [s + [sp_zh.pad_id()] * (pad_len - len(s)) for s in tgt]

  pad_videos = torch.tensor(pad_videos)
  pad_src = torch.tensor(pad_src)
  pad_tgt = torch.tensor(pad_tgt)

  return pad_videos, pad_src, pad_tgt


class VaTeXDataset(Dataset):
  def __init__(self, files, train=True):
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

          en = [[sp_en.bos_id()] + sent + [sp_en.eos_id()] for sent in sp_en.encode(en)]
          zh = [[sp_zh.bos_id()] + sent + [sp_zh.eos_id()] for sent in sp_zh.encode(zh)]

          pairs = list(zip(en, zh))

          # if train:
          #   self.data.append((video_feats, pairs))
          # else:
          for pair in pairs: # no random for test
            self.data.append((video_feats, [pair]))

          # if len(self.data) >= 2:
          #   break # remove this

  def __getitem__(self, i):
    video, pairs = self.data[i]
    en, zh = random.choice(pairs)
    return video, en, zh
  
  def __len__(self):
    return len(self.data)

dataloader = DataLoader(VaTeXDataset(VATEX_TEST_FILES, train=False), batch_size=BATCH_SIZE, collate_fn=pad_to_longest, shuffle=True) # remove shuffle
# test_dataloader = DataLoader(VaTeXDataset(VATEX_TEST_FILES, train=False), batch_size=BATCH_SIZE, collate_fn=pad_to_longest, shuffle=True) # remove shuffle

print("Amount of data: ", len(dataloader))

model = TransformerModel(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, d_model=256, N=NUMBER_OF_LAYERS, video_emb_size=VIDEO_EMB_SIZE)
model.load_state_dict(torch.load(MODEL_DIR + MODEL_SAVE, map_location=device))
model = model.to(device)

srcs = []
tgts = []
preds = []

with torch.no_grad():
    for vid, src, tgt in tqdm(dataloader, desc='Translating'):
        vid = vid.to(device)

        batch = Batch(src, tgt, PAD)
        batch.src = batch.src.to(device)
        batch.src_mask = batch.src_mask.to(device)
        
        out = greedy_decode(model, batch.src, batch.src_mask, MAX_LEN, BOS, EOS, video=vid, tokenizer=sp_zh)

        src = sp_en.decode(src.tolist())
        tgt = sp_zh.decode(tgt.tolist())

        srcs += src
        tgts += tgt
        preds += [o[0] for o in out]

with open(TRANS_DIR + 'src.txt', 'w') as file:
  for line in srcs:
    file.write(line + '\n')

with open(TRANS_DIR + 'tgt.txt', 'w') as file:
  for line in tgts:
    file.write(line + '\n')

with open(TRANS_DIR + 'preds.txt', 'w') as file:
  for line in preds:
    file.write(line + '\n')
