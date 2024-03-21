import os

# os.system('rm -rf jobs')

with open('train-temp.sh', 'r') as f:
    train_temp = f.read()

with open('inf-temp.sh', 'r') as f:
    inf_temp = f.read()

data_groups = [""]

for mp in [15, 30, 60]:
    for my in ['end', 'rand']:
        data_groups.append(f'mask_{mp}_{my}_')

def write_template(path, temp, args):
    path = 'jobs/' + path
    temp = temp.replace("$INSERT_ARGS_HERE", args)
    temp = temp.replace("$INSERT_HARGS_HERE", args.replace('"', '').replace('  ', ' '))
    sub_dir = '/'.join(path.split('/')[:-1])
    if not os.path.exists(sub_dir): os.makedirs(sub_dir)
    with open(path, 'w') as f:
        f.write(temp)

dataset = 'mad'
for model_type in ['con_tran', 'none_tran', 'tran_tran']:
    for prefix in data_groups:
        # if '15' not in prefix and 'none' not in model_type: continue # remove this
        path = f'train-{dataset}/{prefix}{model_type}.sh'
        args = f'{dataset} {model_type} 1e-4 "{prefix}" false'
        write_template(path, train_temp, args)

dataset = 'opus'
for model_type in ['con_tran', 'none_tran', 'tran_tran']:
    for prefix in data_groups:
        # if '15' not in prefix and 'none' not in model_type: continue # remove this
        path = f'train-{dataset}/{prefix}{model_type}.sh'
        args = f'{dataset} {model_type} 1e-4 "{prefix}" false'
        write_template(path, train_temp, args)

dataset = 'vatex'
for model_type in ['con_tran', 'none_tran', 'tran_tran']:
    for prefix in data_groups:
        # if '15' not in prefix and 'none' not in model_type: continue # remove this
        # training
        path = f'train-{dataset}/{prefix}{model_type}.sh'
        args = f'{dataset} {model_type} 1e-4 "{prefix}" false'
        write_template(path, train_temp, args)

        path = f'finetune-{dataset}/{prefix}{model_type}.sh'
        args = f'{dataset} {model_type} 5e-5 "{prefix}" mad'
        write_template(path, train_temp, args)

        path = f'finetune-{dataset}/{prefix}{model_type}.sh'
        args = f'{dataset} {model_type} 5e-5 "{prefix}" opus'
        write_template(path, train_temp, args)

        # inference
        path = f'eval/{prefix}{model_type}_finetune_opus.sh'
        args = f'{model_type} "{prefix}" finetune_opus'
        write_template(path, inf_temp, args)

        path = f'eval/{prefix}{model_type}_finetune_mad.sh'
        args = f'{model_type} "{prefix}" finetune_mad'
        write_template(path, inf_temp, args)

        path = f'eval/{prefix}{model_type}_only.sh'
        args = f'{model_type} "{prefix}" only'
        write_template(path, inf_temp, args)
