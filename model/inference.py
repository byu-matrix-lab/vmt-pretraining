import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset import pad_to_longest

import math
import copy
import os

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

def prep_model_inputs(include_video = True, **input_kwargs):
    if not include_video:
        if 'video_input' in input_kwargs: del input_kwargs['video_input']
        if 'video_attention_mask' in input_kwargs: del input_kwargs['video_attention_mask']
    return input_kwargs

def to_device(tensors):
    return [i.to(device) for i in tensors]

def save_output(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.write(line + '\n')

def run_inference(
    model,
    tokenizer,
    test_dataset,
    include_video = True,
    batch_size = 20,
    save_path = None):

    model = model.to(device)

    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_to_longest)

    srcs = []
    tgts = []
    preds = []
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(dataloader, desc='Running test set')
        for data in val_bar:
            v, vm, s, sm, t, tm = to_device(data)
            
            model_inputs = prep_model_inputs(
                include_video,
                input_ids=s,
                attention_mask=sm,
                video_input=v,
                video_attention_mask=vm,
                num_beams=5,
                min_length=0,
                max_length=200
            )

            output_ids = model.generate(**model_inputs)

            srcs += tokenizer.decode(s.tolist())
            tgts += tokenizer.decode(t.tolist())
            preds += tokenizer.decode(output_ids.tolist())

    if save_path:
        if not os.path.exists(save_path): os.makedirs(save_path)
        save_output(save_path + 'srcs.txt', srcs)
        save_output(save_path + 'tgts.txt', tgts)
        save_output(save_path + 'preds.txt', preds)

    return srcs, tgts, preds





