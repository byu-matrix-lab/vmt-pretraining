import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset import pad_to_longest

import math
import copy

def prep_model_inputs(include_video = True, **input_kwargs):
    if not include_video:
        if 'video_input' in input_kwargs: del input_kwargs['video_input']
        if 'video_attention_mask' in input_kwargs: del input_kwargs['video_attention_mask']
    return input_kwargs

def run_train(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    include_video = True,
    batch_size = 10,
    val_every = 10,
    early_stop = 10):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_to_longest, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pad_to_longest)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    best_score = math.inf
    best_model = None
    early_stop_c = 0

    for epoch in range(1, 10000):

        # train
        train_bar = tqdm(train_dataloader)
        for v, vm, s, sm, t, tm in train_bar:
            
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

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_bar.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.4f}')
        
        if epoch % val_every == 0:
            total_loss = 0
            total_segs = 0
            with torch.no_grad():
                val_bar = tqdm(val_dataloader)
                for v, vm, s, sm, t, tm in val_bar:
                    # output = model.generate(input_ids=s, attention_mask=sm, video_input=v, video_attention_mask=vm, num_beams=5, min_length=0, max_length=100)
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
                    total_loss += outputs.loss.item()
                    total_segs += s.shape[0]/batch_size
                    val_bar.set_description(f'Val Epoch: {epoch} Loss: {total_loss/total_segs:.4f}')
            cur_score = total_loss / total_segs

            if cur_score < best_score:
                print(f'Saving model weights {cur_score:.4f} < {best_score:.4f}')
                early_stop_c = 0
                best_score = cur_score
                best_model = copy.deepcopy(model.state_dict())
            else:
                early_stop_c += 1
                if early_stop_c >= early_stop:
                    print('Stopping after model failed to improve in',early_stop,'epochs')
                    break

    model.load_state_dict(best_model)





