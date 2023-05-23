import transformers

from transformers import BartConfig # extend this
from model import BartForConditionalGeneration

import torch

config = BartConfig(
                            # vocab_size = tokenizer.get_vocab_size(),
                            vocab_size = 100,
                            max_position_embeddings = 512,
                            
                            encoder_layers = 6,
                            encoder_ffn_dim = 2048,
                            encoder_attention_heads = 8,
                            encoder_layerdrop = 0.0,
                            encoder_input_dim = 0, # do nothing

                            video_encoder_layers = 6,
                            video_encoder_conformer = True,
                            video_encoder_input_dim = 3,

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
                            # pad_token_id = tokenizer.token_to_id("[PAD]"),
                            # bos_token_id = 0,
                            # eos_token_id = tokenizer.token_to_id("[CLS]"),
                            # decoder_start_token_id = tokenizer.token_to_id("[EN]"),
                            # forced_eos_token_id = tokenizer.token_to_id("[CLS]"),
                            # num_labels = len(langs)
                            ) # for descriminator

model = BartForConditionalGeneration(config)

outputs = model(input_ids=torch.tensor([[1,2,3]]), video_input=torch.tensor([[[1,2,3],[4,5,6],[4,5,6]]]).float(), decoder_input_ids=torch.tensor([[1,10,11]]))


# look for pd numbers
