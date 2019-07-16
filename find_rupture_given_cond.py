import pandas as pd
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.platform == 'Windows_NT':
    PROJ_PATH = os.path.join('c:/', 'Apps', 'York')
elif sys.platform == 'linux2':
    PROJ_PATH = '/home/547/hxr547/Projects/York'
OUTPUT_PATH = os.path.join(PROJ_PATH, 'PSHA_PGA')

ID = 8
gmf_file = os.path.join(OUTPUT_PATH, 'gmf-data_{}.csv'.format(ID))
rupture_file = os.path.join(OUTPUT_PATH, 'ruptures_{}.csv'.format(ID))

gmf = pd.read_csv(gmf_file)
# multiple gmf from one rupture
gmf['rupid'] = gmf['eid'] // 2 ** 32

#                eid  sid   gmv_PGA rupid
#0  3171416736268288    0  0.000619

rupture = pd.read_csv(rupture_file, skiprows=1, delimiter='\t', index_col=0)

assert set(gmf['rupid']).issubset(set(rupture.index))  

# >>> rupture.shape
# (1182583, 10)
# >>> gmf.shape
# (5477750, 4)
# multiple gmf from rupture 

rupture.drop(['multiplicity','trt','strike','dip','rake','boundary'], axis=1, inplace=True)

gmf = gmf.merge(gmf['rupid'].apply(lambda x: rupture.loc[x]), left_index=True, right_index=True)

gmf.to_csv('./gmf_combined_{}.csv'.format(ID))


