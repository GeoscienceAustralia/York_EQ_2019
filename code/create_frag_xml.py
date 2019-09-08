# scrip to write xml file for WA project using the GA PGA based vulnerability models

import os
import sys
#import cPickle
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/hyeuk/Projects/oq-tools/input')
from fragilityTxt2NRML import FragilityTxtReader, FragilityWriter

# sys.path.insert(0, '/Users/hyeuk/Projects/scenario_Guildford/code')
# from create_fragility import compute_vulnerability

# either PGA, SA03 or MMI
IMT = 'PGA'
if IMT in ['PGA', 'SA03', 'MMI']:
    print('creating {} based fragility/fatality model'.format(IMT))
else:
    sys.exit('IMT should be either PGA, SA03 or MMI')

imt_dic = {'SA03': 'SA(0.3)', 
           'MMI': 'MMI',
           'PGA': 'PGA'}

def read_pe_by_ds(filename):
    a = pd.read_csv(filename, index_col=0)
    poe = []
    for line in a.values:
        poe.append(['{}'.format(x) for x in line])
    return poe

# read fragility 
#path_frag = '/Volumes/exFAT120/scenario_Perth/input/fragility_csv_May19'
#path_frag = '/Volumes/exFAT120/scenario_Perth/input/fragility_csv_org_May19'
path_frag = '/Volumes/exFAT120/scenario_Perth/input/fragility_csv_AK_May19'
frag_functions = {}
for _file in os.listdir(path_frag):
    if ('pe_by_ds' in _file) and (IMT in _file):
        idx = re.search('pe_by_ds_(.+?).csv',_file).group(1)
        bldg = '_'.join(x for x in idx.split('_')[:2])
        frag_functions[bldg] = read_pe_by_ds(os.path.join(path_frag, _file))
        imt_list = pd.read_csv(os.path.join(path_frag, _file), 
                            index_col=0).columns.tolist()

# building mapping
path_mapping = '/Users/hyeuk/Projects/scenario_Guildford/input'
bldg_mapping = pd.read_csv(os.path.join(path_mapping, 'bldg_class_mapping.csv'))

metadata = {'assetCategory': 'buildings',
            'description': 'GA {} based fragility models'.format(IMT),
            'lossCategory': 'structural',
            'fragilityModelID': 'GA_{}_AK'.format(IMT),
            'limitStates': ['slight', 'moderate', 'extensive', 'complete']}

frag_def = []
for _, row in bldg_mapping.iterrows():

    if row['MAPPING2'][:2] in ['UR', 'W1', 'W2']:
        pivot_year = 1945
    else:
        pivot_year = 1996 

    for year_str in ['Pre', 'Post']:

        mapped_frag_id = '{}_{}'.format(row['MAPPING3'], year_str)
        fn_id = '{}_{}{:d}'.format(row['NEXIS_CONS'], year_str, pivot_year)

        try:
            poe = frag_functions[mapped_frag_id]
        except KeyError:
            print('{} not available'.format(mapped_frag_id))
        else:
            print('{} -> {}'.format(mapped_frag_id, fn_id))
            tmp = {'imt': imt_list,
                'poe': poe,
                'frag_format': 'discrete',
                'nodamage_limit': '0.01',
                'imt_str': '{}'.format(imt_dic[IMT]),
                'fragilityFunctionId': fn_id}

            frag_def.append(tmp)

writer = FragilityWriter()
_file = '/Volumes/exFAT120/scenario_Perth/input/fragility_GA_{}_AK_May2019.xml'.format(IMT)
writer.serialize(_file, metadata, frag_def)

# check plot
output = '/Volumes/exFAT120/scenario_Perth/input/output'
x = [float(x) for x in imt_list]
for bldg, value in frag_functions.items():
    plt.figure()
    for i in range(4):
        y = [float(z) for z in value[i]]
        plt.plot(x, y, label=i)
    plt.legend(loc=2)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim([0, 1])
    plt.grid(1)
    plt.savefig(os.path.join(output, './frag_{}_{}.png'.format(bldg, IMT)))
    plt.close()