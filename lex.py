import os
import json
import shutil
from lexicalrichness import LexicalRichness


# OPUS (OpenSubtitles)
with open('../compute/data/opus/OpenSubtitles/OpenSubtitles.en-zh_cn.en') as f:
    text = [line.strip() for line in f.readlines()]

    lex = LexicalRichness(' '.join(text))
    print('OPUS length', len(text))
    print('MTLD OPUS', lex.mtld())
    print('VOCD OPUS', lex.vocd())
    print('MSTTR OPUS', lex.msttr())


# MSR-VTT
with open('../compute/data/msr-vtt/captions.txt') as f:
    text = [line.strip() for line in f.readlines()]
    # scores = [LexicalRichness(seg).mtld() for seg in text if seg]

    lex = LexicalRichness(' '.join(text))
    print('MSR-VTT length', len(text))
    print('MTLD MSR-VTT', lex.mtld())
    print('VOCD MSR-VTT', lex.vocd())
    print('MSTTR MSR-VTT', lex.msttr())

# MAD

with open('../compute/data/mad/MAD_train.json') as jifile:
    data = json.load(jifile)

    text = []
    for key, value in data.items():
        text.append(' '.join(value['tokens']))

    lex = LexicalRichness(' '.join(text))
    print('MAD length', len(text))
    print('MTLD MAD', lex.mtld())
    print('VOCD MAD', lex.vocd())
    print('MSTTR MAD', lex.msttr())

# VATEX

text = []

# for file in os.listdir('../compute/data/vatex/'):
#     if not file.endswith('json') or 'new' in file:
#         continue
with open('../compute/data/vatex/vatex_train.json', 'r') as jifile:
    data = json.load(jifile)
    for seg in data:
        text += seg['enCap'][5:]

lex = LexicalRichness(' '.join(text))
print('VATEX length', len(text))
print('MTLD VATEX', lex.mtld())
print('VOCD VATEX', lex.vocd())
print('MSTTR VATEX', lex.msttr())


    
