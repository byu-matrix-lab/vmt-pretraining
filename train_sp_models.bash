spm_train --input=all_source_train.txt --model_prefix=en --vocab_size=16000 --character_coverage=1.0 --model_type=bpe
spm_train --input=all_target_train.txt --model_prefix=zh --vocab_size=8000 --character_coverage=0.9995 --model_type=bpe
