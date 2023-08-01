import os

# os.system('rm -rf jobs')

with open('inf-temp.sh', 'r') as f:
    inf_temp = f.read()

def write_template(path, temp, args):
    path = 'jobs/' + path
    temp = temp.replace("$INSERT_ARGS_HERE", args)
    temp = temp.replace("$INSERT_HARGS_HERE", args.replace('"', '').replace('  ', ' '))
    sub_dir = '/'.join(path.split('/')[:-1])
    if not os.path.exists(sub_dir): os.makedirs(sub_dir)
    with open(path, 'w') as f:
        f.write(temp)

parent_dir = '../../compute/models/'

model_types = ['con_tran', 'tran_tran', 'none_tran']

for model_type in os.listdir(parent_dir):
    print(f'{parent_dir}{model_type}/')
    if model_type not in model_types: continue
    # print('yes')
    for model in os.listdir(f'{parent_dir}{model_type}/'):
        if 'vatex_' not in model: continue
        prefix, suffix = model.split('vatex_')
        print(model_type, prefix, suffix)

        path = f'val/{prefix}{model_type}_{suffix}.sh'
        args = f'{model_type} "{prefix}" {suffix}'
        write_template(path, inf_temp, args)
