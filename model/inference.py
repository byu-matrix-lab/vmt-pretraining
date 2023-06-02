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

def run_inference(
    model,
    tokenizer,
    test_dataset,
    include_video = True,
    batch_size = 20):

    model = model.to(device)

    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_to_longest)

    total_loss = 0
    total_segs = 0
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(dataloader)
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

            output = tokenizer.decode(output_ids.tolist())
            print(output)
            break
            # val_bar.set_description(f'Testing Loss: {total_loss/total_segs:.4f}')






