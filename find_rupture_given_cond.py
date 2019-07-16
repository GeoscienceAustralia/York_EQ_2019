import pandas as pd
import sys
import os
<<<<<<< HEAD
import argparse
import re
=======
>>>>>>> 2d9fd18c083841dd10dd1a666de760618446d5ff
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

<<<<<<< HEAD
def find_rupture(target_pga):

    target_pga_bin = 0.95*target_pga, 1.05*target_pga

    sel = gmf.loc[(gmf['gmv_PGA'] < target_pga_bin[1]) & (gmf['gmv_PGA'] > target_pga_bin[0])].copy()

    assert sel['mag'].min() > mag_bins[0]
    assert sel['mag'].max() < mag_bins[-1]

    assert sel['centroid_depth'].min() > depth_bins[0]
    assert sel['centroid_depth'].max() < depth_bins[-1]

    assert sel['centroid_lon'].min() > lon_bins[0]
    assert sel['centroid_lon'].max() < lon_bins[-1]

    assert sel['centroid_lat'].min() > lat_bins[0]
    assert sel['centroid_lat'].max() < lat_bins[-1]

    sel_pivot = sel.pivot_table(index=['bin_mag','bin_depth'], columns=['bin_lon', 'bin_lat'], values='eid', aggfunc=('count')).fillna(0)

    _dic = {}

    for key, value in events.items():

        mx = label_mag[(mag_bins < value[0]).sum()-1]
        x = sel_pivot.loc[mx].copy()
        total = x.values.sum()

        _sum = 0.0
        i = 0
        while (_sum < total) and (i <= 5):
            id_depth = np.unravel_index(np.argmax(x.values, axis=None), x.values.shape)
            depth_bin = x.iloc[id_depth[0]].name
            location_bin = x.iloc[id_depth[0]].argmax()

            _sum += x.values[id_depth]
            i += 1
            pro = x.values[id_depth] * 100.0 / total

            _dic.setdefault(key, {})[i] = mx, depth_bin, location_bin[0], location_bin[1], x.values[id_depth], pro
            x.values[id_depth] = 0.0

    return _dic

def write_up_gmf_combined():
    """previous history to create gmf_combined_8.csv

    """

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

    return None

"""main
"""

# PROJ_PATH = '/home/547/hxr547/Projects/York'
# OUTPUT_PATH = '/home/547/hxr547/Projects/York/PSHA_PGA'
if 'win' in sys.platform:
    PROJ_PATH = os.path.join('c:/', 'Apps', 'York')
elif 'linux' in sys.platform:
    PROJ_PATH = '/home/547/hxr547/Projects/York'
elif 'osx' in sys.platform:
    PROJ_PATH = '/Users/hyeuk/Projects/York'
else:
    print('No PROJ_PATH defined')

OUTPUT_PATH = os.path.join(PROJ_PATH, 'PSHA_PGA')

gmf = pd.read_csv(os.path.join(PROJ_PATH, 'gmf_combined_8.csv'))

mag_bins = np.arange(4.0, 8.0, 0.2)
depth_bins = np.arange(0.0, 15.0, 0.5)
lon_bins = np.arange(112.0, 121.2, 0.2)
lat_bins = np.arange(-35.7, -28.0, 0.2)

label_mag = ['m{}'.format(x) for x in range(1, len(mag_bins))]
label_lon = ['lo{}'.format(x) for x in range(1, len(lon_bins))]
label_lat = ['la{}'.format(x) for x in range(1, len(lat_bins))]
label_depth = ['d{}'.format(x) for x in range(1, len(depth_bins))]

gmf['bin_mag'] = pd.cut(gmf['mag'].values, mag_bins, labels=label_mag)
gmf['bin_lon'] = pd.cut(gmf['centroid_lon'].values, lon_bins, labels=label_lon)
gmf['bin_lat'] = pd.cut(gmf['centroid_lat'].values, lat_bins, labels=label_lat)
gmf['bin_depth'] = pd.cut(gmf['centroid_depth'].values, depth_bins, labels=label_depth) 

# Calingiri
# Mw = 5.06

events = {'Meckering': (6.58, 10), 
        'Cadoux': (6.1, 3),
        'Calingiri': (5.03, 15),
        'Lake Muir': (5.3, 2)}

target_PGAs = [5.935e-2, 1.0203e-1, 1.9875e-1]
# - PGA at 500 RP: 5.935e-2
# - PGA at 1000 RP: 1.0203e-1
# - PGA at 2500 RP: 1.9875e-1
big_dic = {}

preferred_event = ['Calingiri', 'Lake Muir', 'Meckering']

for i, (pga, pre_event) in enumerate(zip(target_PGAs, preferred_event)):

    _dic = find_rupture(target_pga=pga)
    big_dic[i] = _dic

    sel_bins = _dic[pre_event][1]

    sel_gmf = gmf.loc[
        (gmf['bin_mag']==sel_bins[0]) & 
        (gmf['bin_depth']==sel_bins[1]) & 
        (gmf['bin_lon']==sel_bins[2]) & 
        (gmf['bin_lat']==sel_bins[3]) & 
        (gmf['gmv_PGA'] < 1.05*pga) & 
        (gmf['gmv_PGA'] > 0.95*pga)]

    print('rupid: {}'.format(sel_gmf['rupid'].value_counts().index[0]))

# select rupture
=======
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

>>>>>>> 2d9fd18c083841dd10dd1a666de760618446d5ff

