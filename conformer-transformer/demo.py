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

MODEL_DIR = '../../compute/models/conformer-transformer-vatex/masked_run1/'
MODEL_SAVE = 'best_model-14.pt'

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
print('Loading tokenizers')
sp_en = spm.SentencePieceProcessor(model_file=TOKEN_DIR + 'en.model')
sp_zh = spm.SentencePieceProcessor(model_file=TOKEN_DIR + 'zh.model')
  
#   video_feats = np.load(video_path + videoid + '.npy')[0]
# video_feats = video_feats[::5]

# en = line['enCap'][-5:]
# zh = line['chCap'][-5:]

# en = [[sp_en.bos_id()] + sent + [sp_en.eos_id()] for sent in sp_en.encode(en)]
# zh = [[sp_zh.bos_id()] + sent + [sp_zh.eos_id()] for sent in sp_zh.encode(zh)]

print('Loading model')
model = TransformerModel(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, d_model=256, N=NUMBER_OF_LAYERS, video_emb_size=VIDEO_EMB_SIZE)
model.load_state_dict(torch.load(MODEL_DIR + MODEL_SAVE, map_location=device))
model = model.to(device)
model.eval()

print('Loading videos')
vids = ['us_pXO_vwPc_000031_000041', 'khURP8O3k_k_000241_000251']
vids = ['../../compute/data/vatex/val/' + v for v in vids]
vids = [np.load(v + '.npy')[0][::5] for v in vids]

print('Tokenizing source sentences')
srcs = ['A <mask> boy is playing tennis',
  'A woman makes <mask> out of wire'
]

print()

with torch.no_grad():
    for vid, src in zip(vids, srcs):
        vid = torch.tensor(vid).to(device)

        print('Input sentence:')
        print(src)
        src = torch.tensor([sp_en.bos_id()] + sp_en.encode(src) + [sp_en.eos_id()])

        src = src.unsqueeze(0)
        vid = vid.unsqueeze(0)

        batch = Batch(src, None, PAD)
        batch.src = batch.src.to(device)
        batch.src_mask = batch.src_mask.to(device)
        
        out = greedy_decode(model, batch.src, batch.src_mask, MAX_LEN, BOS, EOS, video=vid, tokenizer=sp_zh)

        print('Produced translation:')
        print(out[0][0])
        print()
