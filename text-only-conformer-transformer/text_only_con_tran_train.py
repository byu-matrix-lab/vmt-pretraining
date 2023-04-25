import argparse
import os
import sys
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ExponentialLR

from text_only_conformertransformer import *

# special token values
PAD = 0
UNK = 1
BOS = 2
EOS = 3

# # Get the arguments from yaml
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
# args = parser.parse_args()
# with open(args.config) as f:
#     args = yaml.load(f, Loader=yaml.FullLoader)

# print("Welcome to the train script that Ammon made. This print statement is to test if the slurm job is working or not.")

RANDOM_SEED = 42

MODEL_DIR = '../../compute/models/text-only/dmodel2/'

# Dataloader parameters
DATA_DIR = '../../compute/data/'
MAX_VIDEO_LEN = 300
MAX_TEXT_LEN = 300
TOKEN_DIR = '../../compute/data/tokenizing/'
MAD_FILES = ['filtered_comet.txt']
VATEX_TRAIN_FILES = ['vatex_training_v1.0.json']
VATEX_VAL_FILES = ['new_vatex_validation.json']
# VATEX_TRAIN_FILES = ['mask_vatex_train.json']
# VATEX_VAL_FILES = ['mask_vatex_validation.json']

# Model parameters
EPOCHS = 200
VALIDATION_EPOCHS = 1
SAVE_EVERY = 10
EARLY_STOPPING = 10

SOURCE_VOCAB_SIZE = 16000
TARGET_VOCAB_SIZE = 8000
BATCH_SIZE = 64
NUMBER_OF_LAYERS = 6
VIDEO_EMB_SIZE=1024 # 768 for MAD, 1024 for VATEX
LEARNING_RATE = 1e-4
SMOOTHING = 0.1
D_MODEL = 512

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Set device
device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

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

class MADDataset(Dataset):
  def __init__(self, files):
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

          en = [sp_en.bos_id()] + sp_en.encode(en) + [sp_en.eos_id()]
          zh = [sp_zh.bos_id()] + sp_zh.encode(zh) + [sp_zh.eos_id()]

          self.data.append((movie_features, en, zh))

  def __getitem__(self, i):
    return self.data[i]
  
  def __len__(self):
    return len(self.data)

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

  def __getitem__(self, i):
    video, pairs = self.data[i]
    en, zh = random.choice(pairs)
    return video, en, zh
  
  def __len__(self):
    return len(self.data)

# train_dataloader = DataLoader(MADDataset(MAD_FILES), batch_size=BATCH_SIZE, collate_fn=pad_to_longest)
# val_dataloader = DataLoader(MADDataset(MAD_FILES), batch_size=BATCH_SIZE, collate_fn=pad_to_longest)
train_dataloader = DataLoader(VaTeXDataset(VATEX_TRAIN_FILES), batch_size=BATCH_SIZE, collate_fn=pad_to_longest)
val_dataloader = DataLoader(VaTeXDataset(VATEX_VAL_FILES), batch_size=BATCH_SIZE, collate_fn=pad_to_longest)

# Load model
model = TransformerModel(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, d_model=D_MODEL, N=NUMBER_OF_LAYERS, video_emb_size=VIDEO_EMB_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
criterion = LabelSmoothing(size=TARGET_VOCAB_SIZE, padding_idx=PAD, smoothing=SMOOTHING)

train_loss_compute = LossFunction(model.generator, criterion, optimizer)
val_loss_compute = ValidationLossFunction(model.generator, criterion)


training_losses = []
validation_losses = []

print("Number of training batches", len(train_dataloader), flush=True)
print("Number of validation batches", len(val_dataloader), flush=True)

# Train loop
stop = False
best_val_loss_avg = float('inf')
best_model = None
best_epoch = None

start_time = time.time()

for epoch in range(EPOCHS):
    if stop:
        print(f"Validation loss did not improve after {EARLY_STOPPING} epochs, stopping training.")
        break

    # Initialize first tqdm bar
    # loop = tqdm(total=len(train_dataloader), leave=True, position=0)
 
    # Train
    batch_train_losses = []
    for vid, src, tgt in train_dataloader:
        vid = vid.to(device)

        batch = Batch(src, tgt, PAD)
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask, vid)
        loss = train_loss_compute(out, batch.tgt_y, MAX_TEXT_LEN).item()
        # training_losses.append((epoch, loss))
        batch_train_losses.append(loss)

        # loop.update(1)
        # loop.set_description(f'Train Epoch {epoch}, loss: {loss:.4f}')

    # scheduler.step()
    # loop.close()

    train_avg = sum(batch_train_losses)/len(batch_train_losses)
    training_losses.append((epoch, train_avg))
    # print(f"Epoch {epoch}\tTrain Loss Average: {train_avg}\tTime: {time.time() - start_time:.2f} seconds\tLearning Rate: {scheduler.get_last_lr()}", flush=True)
    print(f"Epoch {epoch}\tTrain Loss Average: {train_avg}\tTime: {time.time() - start_time:.2f} seconds", flush=True)


    # Validation every VALIDATION_EPOCHS epochs
    if (epoch+1) % VALIDATION_EPOCHS == 0:
        val_start_time = time.time()
        # val_loop = tqdm(total=len(val_dataloader), leave=True, position=0)
        with torch.no_grad():
            batch_val_losses = []
            for vid, src, tgt in val_dataloader:
                vid = vid.to(device)

                batch = Batch(src, tgt, PAD)
                out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask, vid)
                loss = val_loss_compute(out, batch.tgt_y, MAX_TEXT_LEN).item()
                # validation_losses.append((epoch, loss))
                batch_val_losses.append(loss)

                # val_loop.update(1)
                # val_loop.set_description(f'  Val Epoch {epoch}, loss: {loss:.4f}')
            
            val_avg = sum(batch_val_losses)/len(batch_val_losses)
            validation_losses.append((epoch, val_avg))

            print(f"Epoch {epoch}\tValidation Loss Average: {val_avg}\tVal Time: {time.time() - val_start_time:.2f} seconds", flush=True)

            # Early stopping if validation loss doesn't decrease for EARLY STOPPING epochs
            if val_avg < best_val_loss_avg:
                stop_count = 0
                best_val_loss_avg = val_avg
                best_model = model.state_dict()
                best_epoch = epoch
                stop = False
            else:
                stop_count += 1
                if stop_count == EARLY_STOPPING:
                    stop = True
        # val_loop.close()
    
    # Save the model every SAVE_EVERY epochs
    if epoch % SAVE_EVERY == 0 and epoch != 0:
        torch.save(model.state_dict(), MODEL_DIR + f'model-{epoch}.pt')

# Save the best model
if best_model is not None:
    torch.save(best_model, MODEL_DIR + f'best_model-{best_epoch}.pt')

# Plot losses over epochs
plt.plot([x[0] for x in training_losses], [x[1] for x in training_losses], label='Training loss')
plt.plot([x[0] for x in validation_losses], [x[1] for x in validation_losses], label='Validation loss')
plt.legend()
plt.savefig(MODEL_DIR + 'losses.png')

# Save model
torch.save(model.state_dict(), MODEL_DIR + 'model.pt')
