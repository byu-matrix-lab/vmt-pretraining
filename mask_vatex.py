import json
import random

working_dir = '../compute/data/vatex/'
MASK_RATIO = 0.5

def mask(sent):
    # print(sent)
    parts = sent.split()
    n = len(parts)
    to_mask = random.sample(range(n), int(n*MASK_RATIO))
    for i in to_mask:
        parts[i] = '<mask>'
    return ' '.join(parts)


def mask_file(input, output):
    input = working_dir + input
    output = working_dir + output

    with open(input, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        for i in range(10):
            item['enCap'][i] = mask(item['enCap'][i])

    with open(output, 'w', encoding='utf-8') as file:
        json.dump(data, file)


mask_file('vatex_training_v1.0.json', 'mask_vatex_train.json')
mask_file('new_vatex_validation.json', 'mask_vatex_validation.json')
mask_file('new_vatex_test.json', 'mask_vatex_test.json')
