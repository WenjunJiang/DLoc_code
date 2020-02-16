import sys
import yaml
import os
import importlib
#%%
data_dir = '/home/aarun/Research/DLoc_code/python/params/'
data_dir_yaml = '/home/aarun/Research/DLoc_code/python/params_yaml/'
files = [f for f in os.listdir(data_dir) if '.py' in f]
# sys.path.append(data_dir)
for fname in files: 
    opt_exp = None
    opt_decoder = None
    opt_encoder = None
    opt_offset_decoder = None
    print('Importing %s'%fname)
    m = importlib.import_module(fname.split('.')[0])
    # exec(compile(open(data_dir + fname, "rb").read(), data_dir + fname, 'exec'))
    opt_exp = dict(m.opt_exp)
    if 'gen' in fname:
        opt_gen = dict(m.opt_gen)
        opt_gen['parent_exp'] = dict(m.opt_gen['parent_exp'])
    else:
        opt_decoder = dict(m.opt_decoder)
        opt_decoder['parent_exp'] = dict(m.opt_decoder['parent_exp'])
        opt_encoder = dict(m.opt_encoder)
        opt_encoder['parent_exp'] = dict(m.opt_encoder['parent_exp'])
        opt_offset_decoder = dict(m.opt_offset_decoder)
        opt_offset_decoder['parent_exp'] = dict(m.opt_offset_decoder['parent_exp'])
    
    with open(data_dir_yaml + fname.split('.')[0] + '.yml', 'w+') as file:
        if 'gen' in fname:
            data = {'opt_exp': opt_exp, 'opt_gen': opt_gen}
        else:
            data = {'opt_exp': opt_exp, 'opt_decoder': opt_decoder, \
                'opt_encoder': opt_encoder, \
                'opt_offset_decoder': opt_offset_decoder}
        yaml.dump(data, file, default_flow_style=False)
#%%
        