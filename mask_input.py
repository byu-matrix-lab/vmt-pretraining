import json
import random

working_dir = '../compute/data/'
MASK_PER = 15
MASK_TYPE = 'rand'

def mask(sent):
    # print(sent)
    parts = sent.split()
    n = len(parts)
    if MASK_TYPE == 'rand':
        to_mask = random.sample(range(n), n*MASK_PER//100)
    else:
        to_mask = range(n - n*MASK_PER//100, n)
    for i in to_mask:
        parts[i] = '<mask>'
    return ' '.join(parts)


def mask_vatex(input, output):
    input = working_dir + 'vatex/' + input
    output = working_dir + 'vatex/' + output

    with open(input, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        for i in range(10):
            item['enCap'][i] = mask(item['enCap'][i])

    with open(output, 'w', encoding='utf-8') as file:
        json.dump(data, file)


def mask_mad(input, output):
    input = working_dir + 'mad/' + input
    output = working_dir + 'mad/' + output

    with open(input, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file.readlines()]
    
    for item in data:
        item['en_sentence'] = mask(item['en_sentence'])

    with open(output, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item))
            file.write('\n')


#mask_mad('mad_train.txt', f'mask_{MASK_PER}_{MASK_TYPE}_mad_train.txt')
#mask_mad('mad_val.txt', f'mask_{MASK_PER}_{MASK_TYPE}_mad_val.txt')

#mask_vatex('vatex_train.json', f'mask_{MASK_PER}_{MASK_TYPE}_vatex_train.json')
#mask_vatex('vatex_validation.json', f'mask_{MASK_PER}_{MASK_TYPE}_vatex_validation.json')
#mask_vatex('vatex_test.json', f'mask_{MASK_PER}_{MASK_TYPE}_vatex_test.json')

for MASK_PER in [15, 30, 60]:
    for MASK_TYPE in ['rand', 'end']:
        mask_vatex('clip_vatex_train.json', f'mask_{MASK_PER}_{MASK_TYPE}_clip_vatex_train.json')
        mask_vatex('clip_vatex_validation.json', f'mask_{MASK_PER}_{MASK_TYPE}_clip_vatex_validation.json')
        mask_vatex('clip_vatex_test.json', f'mask_{MASK_PER}_{MASK_TYPE}_clip_vatex_test.json')
