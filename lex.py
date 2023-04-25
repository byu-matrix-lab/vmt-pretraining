import os
import json
import shutil
from lexicalrichness import LexicalRichness


# MAD

with open('../compute/data/mad/MAD_train.json') as jifile:
    data = json.load(jifile)

    text = []
    for key, value in data.items():
        text.append(' '.join(value['tokens']))

    print(text[:5])
    
    scores = [LexicalRichness(seg).mtld() for seg in text if seg]

    # Compute lexical richness
    mtld = sum(scores) / len(scores)

    print('MAD', mtld)

exit(0)

# VATEX

text = []

for file in os.listdir('../compute/data/vatex/'):
    if not file.endswith('json') or 'new' in file:
        continue
    with open('../compute/data/vatex/' + file, 'r') as jifile:
        data = json.load(jifile)
        for seg in data:
            text += seg['enCap'][5:]


scores = [LexicalRichness(seg).mtld() for seg in text if seg]

# Compute lexical richness
mtld = sum(scores) / len(scores)

print('VaTeX', mtld)

# # Tape and compress
# os.system('tar -zcf sample_data.tar.gz sample_data/')
