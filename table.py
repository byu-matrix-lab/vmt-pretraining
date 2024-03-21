import os
from collections import defaultdict

model_types = ['none_tran', 'tran_tran', 'con_tran']

def get_bleu(path):
    if not os.path.exists(path): return -1
    with open(path, 'r') as f:
        for line in f:
            if 'score' in line:
                parts = line.strip().split()
                return parts[-1][:-1]
    return -1

def get_comet(path):
    if not os.path.exists(path): return -1
    with open(path, 'r') as f:
        line = f.readlines()
        if not len(line): return -1
        line = line[-1]
        parts = line.strip().split()
        return parts[-1]

def get_scores(path):
    # print(path)
    val_comet = get_comet(path + 'val_comet_metrics.txt')
    val_bleu = get_bleu(path + 'val_metrics.txt')
    test_comet = get_comet(path + 'comet_metrics.txt')
    test_bleu = get_bleu(path + 'metrics.txt')
    return val_comet, val_bleu, test_comet, test_bleu

groups = defaultdict(list)

parent = '../compute/data/outputs/'
for model_type in model_types:
    path = parent + model_type + '/'
    for model in os.listdir(path):
        if 'vatex_' not in model: continue
        prefix, suffix = model.split('vatex_')
        if suffix == 'finetune': suffix = 'MAD'
        if suffix == 'only': suffix = ''
        if suffix == 'finetune_opus': suffix = 'OpenSubtitles'

        if model_type == 'none_tran' and suffix == 'MAD':
            suffix += ' (text-only)'

        model_path = path + model + '/'
        scores = get_scores(model_path)
        if -1 in scores: print('ERROR', model_path)
        groups[prefix].append((model_type, suffix, *scores))

parent = '../compute/data/outputs/text_only_finetune/'
for model_type in model_types:
    if model_type == 'none_tran': continue
    path = parent + model_type + '/'
    for model in os.listdir(path):
        if 'vatex_' not in model: continue
        prefix, suffix = model.split('vatex_')
        if suffix == 'finetune': suffix = 'MAD'
        if suffix == 'only': suffix = ''
        if suffix == 'finetune_opus': suffix = 'OpenSubtitles'

        if suffix: suffix += ' (text-only)'

        model_path = path + model + '/'
        scores = get_scores(model_path)
        if -1 in scores: print('ERROR', model_path)
        groups[prefix].append((model_type, suffix, *scores))

model_map = {'none_tran': 'Text-only', 'tran_tran': 'Transformer', 'con_tran': 'Conformer'}

for group, data in sorted(groups.items()):
    data = sorted(data)
    print(group)
    print(r"""\begin{table*}[]
\centering
\begin{tabular}{|l|l|ll|ll|}
\hline
 &  & \multicolumn{2}{c|}{Validation} & \multicolumn{2}{c|}{Test}  \\
Architecture & Pretraining Corpus & COMET & BLEU & COMET & BLEU \\
\hline""")
    for item in sorted(data):
        item = list(item)
        item[0] = model_map[item[0]]
        print(' & '.join(item),'\\\\')
    print(r"""\hline
\end{tabular}
\caption{}
\label{tab:}
\end{table*}""")
    print()
    print()
