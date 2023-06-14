from transformers import BartConfig # extend this
import torch
from torch import nn
import sentencepiece as spm
import sys
sys.path.append('../model')

from model import BartForConditionalGeneration
from dataset import VaTeXDataset, MADDataset
from train import run_train

tokenizer = spm.SentencePieceProcessor(model_file='../../compute/data/tokenizing/en-and-zh.model')

config = BartConfig(
                            vocab_size = tokenizer.vocab_size(),
                            max_position_embeddings = 512,
                            
                            encoder_layers = 6,
                            encoder_ffn_dim = 2048,
                            encoder_attention_heads = 8,
                            encoder_layerdrop = 0.0,
                            encoder_input_dim = 0, # do nothing

                            video_encoder_layers = 6,
                            video_encoder_conformer = False,
                            video_encoder_input_dim = 768, # MAD
                            # video_encoder_input_dim = 1024, # VaTeX

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
                            )

model = BartForConditionalGeneration(config)

model.load_state_dict(torch.load('../../compute/models/none-tran/mask-60-end-mad-pretrain'))

# edit up projection layer for vatex shape
model.model.encoder.video_encoder.project = nn.Linear(1024, 512)

train_dataset = VaTeXDataset(['mask_60_end_vatex_train.json'], tokenizer)
val_dataset = VaTeXDataset(['mask_60_end_vatex_validation.json'], tokenizer)

# train_dataset = VaTeXDataset(['vatex_training_v1.0.json'], tokenizer)
# val_dataset = VaTeXDataset(['new_vatex_validation.json'], tokenizer)

# train_dataset = MADDataset(['mad-train.txt'], tokenizer)
# val_dataset = MADDataset(['mad-val.txt'], tokenizer)

run_train(model, tokenizer, train_dataset, val_dataset, include_video=False, lr=5e-5)

torch.save(model.state_dict(), '../../compute/models/none-tran/mask-60-end-vatex-finetune')


        