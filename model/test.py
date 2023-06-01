import transformers

from transformers import BartConfig # extend this
from model import BartForConditionalGeneration

import torch
import sentencepiece as spm

from torch.utils.data import DataLoader

from dataset import VaTeXDataset, pad_to_longest

tokenizer = spm.SentencePieceProcessor(model_file='../../data/tokenizing/en-and-zh.model')

config = BartConfig(
                            vocab_size = tokenizer.vocab_size(),
                            max_position_embeddings = 512,
                            
                            encoder_layers = 6,
                            encoder_ffn_dim = 2048,
                            encoder_attention_heads = 8,
                            encoder_layerdrop = 0.0,
                            encoder_input_dim = 0, # do nothing

                            video_encoder_layers = 6,
                            video_encoder_conformer = True,
                            video_encoder_input_dim = 1024, # VaTeX

                            decoder_layers = 6,
                            decoder_ffn_dim = 2048,
                            decoder_attention_heads = 8,
                            decoder_layerdrop = 0.0,
                            
                            activation_function = 'swish',
                            d_model = 512,
                            dropout = 0.1,
                            attention_dropout = 0.0,
                            activation_dropout = 0.0,
                            init_std = 0.02,
                            classifier_dropout = 0.0,
                            scale_embedding = True,
                            is_encoder_decoder = True,
                            pad_token_id = 0, # TODO: do these need to be in a different format???
                            bos_token_id = 2,
                            eos_token_id = 3,
                            # decoder_start_token_id = tokenizer.token_to_id("[EN]"),
                            # forced_eos_token_id = tokenizer.token_to_id("[CLS]"),
                            # num_labels = len(langs)
                            ) # for descriminator

model = BartForConditionalGeneration(config)

# src = torch.tensor([[1,2,3]])
# video = torch.tensor([[[1,2,3],[4,5,6],[4,5,6]]]).float()
# tgt = torch.tensor([[1,10,11]])

# outputs = model(input_ids=src, video_input=video, decoder_input_ids=tgt)


train_dataset = VaTeXDataset(['vatex_training_v1.0.json'], tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=5, collate_fn=pad_to_longest, shuffle=False)
dummy = torch.ones((5, 1), dtype=int)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
optimizer.zero_grad()


for epoch in range(1, 10000):
    for v, vm, s, sm, t, tm in train_dataloader:
        
        outputs = model(input_ids=s, video_input=v, labels=t)

        loss = outputs.loss

        print(f'{epoch=} {loss.item()=:0.6f}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # exit(0)
        break
    
    if epoch % 100 == 0:
        with torch.no_grad():
            for v, vm, s, sm, t, tm in train_dataloader:
                output = model.generate(input_ids=s, video_input=v, num_beams=1, min_length=0, max_length=20)
                for targ, pred in zip(t, output):
                    print(tokenizer.decode(targ.tolist()))
                    print(tokenizer.decode(pred.tolist()))
                break
        