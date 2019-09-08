# scrip to write xml file for WA project using the GA fatality models
import os
import sys
import re
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/hyeuk/Projects/oq-tools/input')
from vulnerabilityTxt2NRML import VulnerabilityTxtReader, VulnerabilityWriter

#sys.path.insert(0, '/Users/hyeuk/Projects/scenario_Guildford/code')

# either PGA, SA03 or MMI
IMT = 'PGA'
if IMT in ['PGA', 'SA03', 'MMI']:
    print('creating {} based fragility/fatality model'.format(IMT))
else:
    sys.exit('IMT should be either PGA, SA03 or MMI')

imt_dic = {'SA03': 'SA(0.3)', 
           'MMI': 'MMI',
           'PGA': 'PGA'}

#path_fat = '/Volumes/exFAT120/scenario_Perth/input/fragility_csv_May19'
#path_fat = '/Volumes/exFAT120/scenario_Perth/input/fragility_csv_org_May19'
path_fat = '/Volumes/exFAT120/scenario_Perth/input/fragility_csv_AK_May19'

# read fatality for residential bldgs
fat_functions = {}
for _file in os.listdir(path_fat):
    if ('casualty' in _file) and (IMT in _file):
        idx = re.search('casualty_(.+?).csv',_file).group(1)
        ids = int(idx.split('_')[0])
        bldg = '_'.join(x for x in idx.split('_')[1:3])
        filename = os.path.join(path_fat, _file)
        fat_functions.setdefault(ids, {})[bldg] = pd.read_csv(filename, index_col=0).loc['HAZUS'].values
        imt_list = pd.read_csv(filename, index_col=0).columns.tolist()

# GA Class to Vul ID
path_vul = '/Users/hyeuk/Projects/scenario_Guildford/input'
bldg_mapping = pd.read_csv(os.path.join(path_vul, 'bldg_class_mapping.csv'))

metadata = {'assetCategory': 'buildings',
            'description': 'GA {} based vulnerability models'.format(IMT),
            'lossCategory': 'occupants',
            'vulnerabilityModelID': 'GA_{}'.format(IMT)}

#imt_list = ['{}'.format(x) for x in mmi_range]
#cov_list = ['{}'.format(x) for x in np.zeros_like(imt_list)]
zeros_list = ['{}'.format(x) for x in np.zeros(len(imt_list))]

for ids in range(1, 5):

    vuln_def = []
    for _, row in bldg_mapping.iterrows():

        if row['MAPPING2'][:2] in ['UR', 'W1', 'W2']:
            pivot_year = 1945
        else:
            pivot_year = 1996 

        for year_str in ['Pre', 'Post']:

            mapped_vul_id = '{}_{}'.format(row['MAPPING3'], year_str)
            fn_id = '{}_{}{:d}'.format(row['NEXIS_CONS'], year_str, pivot_year)
            try:
                loss_ratio = ['{}'.format(x) for x in fat_functions[ids][mapped_vul_id]]
            except KeyError:
                loss_ratio = zeros_list
            else:
                print('{}: {} -> {}'.format(ids, mapped_vul_id, fn_id))

            tmp = {'imt': imt_list,
                  'lossRatio': loss_ratio,
                  'probabilityDistribution': 'LN',
                  'coefficientVariation': zeros_list,
                  'imt_str': '{}'.format(imt_dic[IMT]),
                  'vulnerabilityFunctionId': fn_id}

            vuln_def.append(tmp)

    writer = VulnerabilityWriter()
    # outfile = '/Volumes/exFAT120/scenario_Perth/input/fatality_GA_severity_{}_{}_May2019.xml'.format(ids, IMT)
    outfile = '/Volumes/exFAT120/scenario_Perth/input/fatality_GA_severity_{}_{}_AK_May2019.xml'.format(ids, IMT)
    writer.serialize(outfile, metadata, vuln_def)

