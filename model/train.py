import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset import pad_to_longest

from nltk.translate.chrf_score import corpus_chrf

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

def run_train(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    include_video = True,
    batch_size = 32,
    val_every = 1,
    early_stop = 5,
    lr = 1e-4):

    model = model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_to_longest, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pad_to_longest)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -math.inf
    best_model = None
    early_stop_c = 0

    for epoch in range(1, 10000):
        # increasing batch size
        # train_dataloader = DataLoader(train_dataset, batch_size=min(20, batch_size+epoch-1), collate_fn=pad_to_longest, shuffle=True)

        # train
        model.train()
        total_loss = 0
        total_segs = 0
        train_bar = tqdm(train_dataloader)
        for data in train_bar:
            v, vm, s, sm, t, tm = to_device(data)
            
            model_inputs = prep_model_inputs(
                include_video,
                input_ids=s,
                attention_mask=sm,
                video_input=v,
                video_attention_mask=vm,
                labels=t,
                decoder_attention_mask=tm
                )
            outputs = model(**model_inputs)

            loss = outputs.loss
            total_loss += loss.item()
            total_segs += v.size(0) / batch_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_bar.set_description(f'Train Epoch: {epoch} Cur Loss: {loss.item():.4f} Avg Loss: {total_loss/total_segs:.4f}')
        
        if epoch % val_every == 0:
            total_loss = 0
            total_segs = 0
            model.eval()
            output_pair = None
            with torch.no_grad():
                val_bar = tqdm(val_dataloader)
                for data in val_bar:
                    v, vm, s, sm, t, tm = to_device(data)

                    # model_inputs = prep_model_inputs(
                    #     include_video,
                    #     input_ids=s,
                    #     attention_mask=sm,
                    #     video_input=v,
                    #     video_attention_mask=vm,
                    #     labels=t,
                    #     decoder_attention_mask=tm
                    #     )
                    # outputs = model(**model_inputs)
                    # loss = outputs.loss
                    # total_loss -= loss.item()
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

                    outputs = model.generate(**model_inputs)
                    outputs = tokenizer.decode(outputs.tolist())
                    refs = tokenizer.decode(t.tolist())
                    if output_pair is None: output_pair = (refs[0], outputs[0])
                    total_loss += corpus_chrf(outputs, refs)
                    total_segs += s.shape[0]/batch_size
                    val_bar.set_description(f'Val Epoch: {epoch} Score: {total_loss/total_segs:.4f}')
            if output_pair is not None: print('Example translation:',*output_pair, sep=' -> ')
            cur_score = total_loss / total_segs

            if cur_score > best_score:
                print(f'Saving model weights {cur_score:.4f} > {best_score:.4f}')
                early_stop_c = 0
                best_score = cur_score
                best_model = copy.deepcopy(model.state_dict())
            else:
                early_stop_c += 1
                if early_stop_c >= early_stop and best_score > -math.inf:
                    print('Stopping after model failed to improve in',early_stop,'epochs')
                    break

    model.load_state_dict(best_model)





