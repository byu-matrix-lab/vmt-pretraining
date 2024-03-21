from transformers import BartConfig # extend this
import torch
from torch import nn
import sentencepiece as spm
import sys
sys.path.append('../model')

from model import BartForConditionalGeneration
from dataset import VaTeXDataset, MADDataset, OpusDataset
from train import run_train

# mad con_tran 1e-4 "" ""
# mad/vatex, con/tran/none, lr, prefix, source_model
print(sys.argv)
_, dataset, model_type, lr, prefix, source_model = sys.argv

if source_model != 'false': # true / false
    source_model_path = f'../../compute/models/{model_type}/{prefix}{source_model}_only'
else: 
    source_model_path = ''
    source_model = ''
lr = float(lr)

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
                            video_encoder_conformer = (model_type == 'con_tran'),
                            conv_depthwise_kernel_size = 31,
                            video_encoder_input_dim = (768 if ('mad' in source_model if source_model else dataset == 'mad') else 1024), # MAD / VaTeX

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
                            pad_token_id = 0,
                            bos_token_id = 2,
                            eos_token_id = 3,
                            )

model = BartForConditionalGeneration(config)

if source_model_path:
    model.load_state_dict(torch.load(source_model_path))

    # edit up projection layer for vatex shape
    if ('mad' == dataset) != ('mad' in source_model):
        model.model.encoder.video_encoder.project = nn.Linear((768 if dataset == 'mad' else 1024), 512)

if dataset == 'mad':
    val_dataset = MADDataset([f'{prefix}mad_val.txt'], tokenizer)
    train_dataset = MADDataset([f'{prefix}mad_train.txt'], tokenizer)
elif dataset == 'vatex':
    train_dataset = VaTeXDataset([f'{prefix}vatex_train.json'], tokenizer)
    val_dataset = VaTeXDataset([f'{prefix}vatex_validation.json'], tokenizer)
else:
    train_dataset = OpusDataset(['OpenSubtitles/train.tsv'], tokenizer)
    val_dataset = OpusDataset(['OpenSubtitles/val.tsv'], tokenizer)

include_video = (model_type != 'none_tran')
print('Including video', include_video)

model_suffix = f'finetune_{source_model}' if source_model else 'only'
save_path = f'../../compute/models/{model_type}/{prefix}{dataset}_{model_suffix}'

run_train(model, tokenizer, train_dataset, val_dataset, lr=lr, include_video=include_video, save_path=save_path)

torch.save(model.state_dict(), save_path)


        
