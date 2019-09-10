import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate

#PROJ_DIR = '/home/hyeuk/Projects/scenario_Perth'
PROJ_DIR = '/home/547/hxr547/Projects/scenario_Perth'
path_list = ['MUN_M42_D15', 'MUN_M44_D10', 'MUN_M48_D8', 'MUN_M51_D8',
             'WACA_M42_D25', 'WACA_M44_D20', 'WACA_M48_D18', 'WACA_M5_D16',
            ]
ids = range(240, 248)
return_periods = [500, 1000, 2500, 5000]*2

# load agg loss curve
loss_curve = pd.read_csv('../NSHA18/output_event/agg_loss_curve-mean_349.csv',skiprows=1)
loss_curve = loss_curve.loc[1:].copy()
y_ = np.array([float(x) for x in loss_curve['loss_value']])
x_ = np.array([float(x) for x in loss_curve['return_period']])
plt.figure()
plt.semilogx(x_, y_/1.0e+6,'-')
f = interpolate.interp1d(y_, np.log(x_))
plt.xlabel('Return period')
plt.ylabel('Loss (M AUD)')
plt.savefig('loss_curve.png',dpi=300)

#
# check classic vs. event-based
cl_loss = pd.read_csv('../NSHA18/output_classic/avg_losses-mean_341.csv')
ev_loss = pd.read_csv('../NSHA18/output_event/avg_losses-mean_349.csv')

ratio = ev_loss['structural'].sum()/cl_loss['structural'].sum()
print(f'classic to event: {ratio:.3f}')

# scenarios 
for path,id_,rt in zip(path_list, ids, return_periods):
    _file = os.path.join(PROJ_DIR, 'NSHA18', path, 'output/AK', f'agglosses_{id_}.csv')
    try:
        assert os.path.exists(_file)
    except AssertionError:
        print(f'missing {_file}')
    else:
        value = pd.read_csv(_file, index_col=0).loc['structural', 'mean']
        try:
            est_rp = np.exp(f(value))
        except ValueError:
            print(f'{path}: value:{value}')
        else:
            print(f'{path}: value:{value} -> Rp:{est_rp}')
            plt.plot(est_rp, value/1.0e+6, 'o', label=f'{path}:{rt}')

plt.legend(loc=2)
plt.xlabel('Return period')
plt.ylabel('Loss (M AUD)')
#plt.yticks(np.arange(0.0, 30000, 1000))
plt.grid(1)
plt.savefig('./loss_curve_vs_scenarios.png', dpi=300)
plt.close()


