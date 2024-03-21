from transformers import BartConfig # extend this
import torch
import sentencepiece as spm
import sys
sys.path.append('../model')

from model import BartForConditionalGeneration
from dataset import VaTeXDataset, MADDataset
from inference import run_inference

# con_tran "" ""
_, model_type, prefix, suffix = sys.argv
print(sys.argv)

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
                            pad_token_id = 0,
                            bos_token_id = 2,
                            eos_token_id = 3,
                            )

model = BartForConditionalGeneration(config)

model.load_state_dict(torch.load(f'../../compute/models/{model_type}/{prefix}vatex_{suffix}',map_location=torch.device('cpu')))

test_dataset = VaTeXDataset([f'{prefix}vatex_test.json'], tokenizer)
# test_dataset = VaTeXDataset([f'{prefix}vatex_validation.json'], tokenizer) # remove this

run_inference(
    model,
    tokenizer,
    test_dataset,
    include_video=(model_type != 'none_tran'),
    save_path = f'../../compute/data/outputs/{model_type}/{prefix}vatex_{suffix}/'
)

