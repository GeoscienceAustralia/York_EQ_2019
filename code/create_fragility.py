import os
import sys
import copy
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
#import cPickle
#from collections import OrderedDict
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d

# %matplotlib inline
# get_ipython().magic(u'matplotlib inline')

import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
pylab.rcParams['legend.numpoints'] = 1

# either PGA, SA03 or MMI
IMT = 'PGA'
if IMT in ['PGA', 'SA03', 'MMI']:
    print('creating {} based fragility/fatality model'.format(IMT))
else:
    sys.exit('IMT should be either PGA, SA03 or MMI')

PROJ_PATH = '/home/hyeuk/Projects'
rnd_state = np.random.RandomState(1)
damage_labels = ['no', 'slight', 'moderate', 'extensive', 'complete']
damage_thresholds = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]
# damage_variability = [0.6, 0.6, 0.6, 0.6]
#sa03_range = np.arange(0.0, 2.8, 0.1)

bldg_list = ['Timber_Pre1945', 'Timber_Post1945', 'URM_Pre1945', 'URM_Post1945']
hazus_data_path = '/Users/hyeuk/Projects/data/hazus'
casualty_rate = read_hazus_casualty_data(hazus_data_path, bldg_list)
collapse_rate = read_hazus_collapse_rate(hazus_data_path, bldg_list)

bldg_map_to_HAZUS = {'W1_Pre': 'Timber_Pre1945',
                     'W1_Post': 'Timber_Post1945',
                     'URML_Pre': 'URM_Pre1945',
                     'URML_Post': 'URM_Post1945'}

referred_cv_by_bldg = {'Timber': 0.8, 'URML': 0.8, 'C1L': 1.2, 'C2L': 1.2,
        'C2H': 1.2, 'PC1L': 0.8, 'S2L': 1.2, 'S5L': 1.2, 'base': 0.8, 'full': 0.8}


path_vul = os.path.join(PROJ_PATH, 'York/input')
path_csv = os.path.join('/Users/hyeuk/Projects/scenario_Perth/input/fragility_csv_AK_May19'

def read_hazus_collapse_rate(hazus_data_path, selected_bldg_class=None):
    """
    read hazus collapse rate parameter values
    """

    # read collapse rate (table 13.8)
    fname = os.path.join(hazus_data_path, 'hazus_collapse_rate.csv')
    collapse_rate = pd.read_csv(fname, skiprows=1, names=['Bldg type', 'rate'],
                                index_col=0, usecols=[1, 2])
    collapse_rate = collapse_rate.to_dict()['rate']

    if selected_bldg_class is not None:
        removed_bldg_class = (set(collapse_rate.keys())).difference(set(
            selected_bldg_class))
        [collapse_rate.pop(item) for item in removed_bldg_class]

    return collapse_rate

def read_hazus_casualty_data(hazus_data_path, selected_bldg_class=None):
    """
    read hazus casualty parameter values
    """

    # read indoor casualty (table13.3 through 13.7)
    severity_list = ['Severity{}'.format(i) for i in range(1, 5)]
    list_ds = ['none', 'slight', 'moderate', 'extensive', 'complete', 'collapse']
    colname = ['Bldg type'] + severity_list

    multi_index = pd.MultiIndex.from_product([selected_bldg_class, severity_list], 
                                       names=['Bldg type', 'Severity'])
    casualty_rate = pd.DataFrame(0, index=multi_index, columns=list_ds)

    for ds in list_ds[1:]:

        file_ = 'hazus_indoor_casualty_{}.csv'.format(ds)
        fname = os.path.join(hazus_data_path, file_)

        # tmp = pd.read_csv(fname, skiprows=1, header=None)
        tmp = pd.read_csv(fname, skiprows=1,
                          names=colname, usecols=[1, 2, 3, 4, 5], index_col=0)
        if selected_bldg_class is not None:
            okay = tmp.index.isin(selected_bldg_class)
            dic_ = tmp[okay].to_dict('index')
        else:
            dic_ = tmp.to_dict('index')

        reform = {(outerKey, innerKey): values for outerKey, innerDict in dic_.iteritems() 
                  for innerKey, values in innerDict.iteritems()}

        casualty_rate[ds] = pd.DataFrame({ds: reform}, index=multi_index)

    return casualty_rate # re-order the item

def get_fatality_given_loss_ratio_ATC(bldg_str, ps_value, flag_fig=False):

    # Table 4-3 ATC-13 (1985)
    # row: damage state
    # column: CDF(%), Injured Minor, Injured Serious, Dead
    # estimates are for all types of construction except light steel construction
    # and wood-frame construction
    # For light steel and wood-frame construction, multiply all numerators by 0.1
    CDF_lookup = np.array([[  0,      0,          0,          0],
                    [0.5,    3.0/100000,   1.0/250000,   1.0/1000000],
                    [5,      3.0/10000,    1.0/25000,    1.0/100000],
                    [20,     3.0/1000,     1.0/2500,     1.0/10000],
                    [45,     3.0/100,      1.0/250,      1.0/1000],
                    [80,     3.0/10,       1.0/25,       1.0/100],
                    [100,    2.0/5,        2.0/5,        1.0/5]])
    CDF_lookup[:, 0] *= 0.01  # percent to float

    if 'Timber' in bldg_str:
        CDF_lookup[:, 1:] *= 0.1

    # fatality only
    imt_range = ps_value.index.values
    f_i = interp1d(CDF_lookup[:, 0], CDF_lookup[:, -1])
    fatality_ATC = np.zeros(shape=(len(imt_range), 2))
    fatality_ATC[:, 0] = imt_range
    fatality_ATC[:, 1] = f_i(ps_value.values)

    if flag_fig:
        plt.figure()
        plt.plot(fatality_ATC[:,0], fatality_ATC[:, 1], '-')
        plt.xlabel('{}'.format(IMT))
        plt.ylabel('Fatality rate')
        plt.title('ATC-13: {}'.format(bldg_str))
        #plt.yticks(np.arange(0, 1.1, 0.1))
        #plt.ylim([0, 1])
        plt.grid(1)
        plt.savefig('fatality_ATC_{}.png'.format(bldg_str), dpi=200)
        plt.close()

    return fatality_ATC

def get_fatality_given_pe(pe_by_ds, bldg_str, collapse_rate, casualty_rate, flag_fig=False):

    # split complete into collapse or not
    pe_by_ds[5] = np.zeros_like(pe_by_ds[4])
    pe_by_ds[5][:, 1] = pe_by_ds[4][:, 1] * collapse_rate[bldg_str] * 0.01
    pe_by_ds[4][:, 1] *= (100-collapse_rate[bldg_str])*0.01

    # pb_by_ds
    pb_by_ds = copy.deepcopy(pe_by_ds)
    for i in sorted(pe_by_ds.keys(), reverse=True): # 5, 4, 3, 2, 1
        if i > 1:
            pb_by_ds[i-1][:, 1] = pe_by_ds[i-1][:, 1] - pe_by_ds[i][:, 1] 

    casualty = {}
    for ids in range(1, 5):
        _casualty_by_ds = casualty_rate.loc[bldg_str, 'Severity{}'.format(ids)].values * 0.01

        fatality = np.zeros_like(pb_by_ds[1])
        fatality[:, 0]  = pb_by_ds[1][:, 0]
        for i in pb_by_ds.keys():
            fatality[:, 1] += _casualty_by_ds[i] * pb_by_ds[i][:, 1]

        casualty[ids] = fatality

    if flag_fig:

        plt.figure()
        for ids in range(1, 5):
            plt.plot(casualty[ids][:,0], casualty[ids][:, 1], '-', label='severity {}'.format(ids))
        plt.xlabel(IMT)
        plt.ylabel('Fatality rate')
        plt.title(bldg_str)
        #plt.yticks(np.arange(0, 1.1, 0.1))
        #plt.ylim([0, 1])
        plt.grid(1)
        plt.savefig('fatality_{}.png'.format(bldg_str), dpi=200)
        plt.close()

    return casualty

def sample_vulnerability_gamma(rnd_state, mean_lratio, nsample=1000, cov=1.0):

    """
    The probability density function for `gamma` is::

        gamma.pdf(x, a) = lambda**a * x**(a-1) * exp(-lambda*x) / gamma(a)

    for ``x >= 0``, ``a > 0``. Here ``gamma(a)`` refers to the gamma function.

    The scale parameter is equal to ``scale = 1.0 / lambda``.

    `gamma` has a shape parameter `a` which needs to be set explicitly. For
    instance:

        >>> from scipy.stats import gamma
        >>> rv = gamma(3., loc = 0., scale = 2.)

    shape: a
    scale: b
    mean = a*b
    var = a*b*b
    cov = 1/sqrt(a) = 1/sqrt(shape)
    shape = (1/cov)^2
    """

    shape_ = np.power(1.0/cov, 2.0)
    scale_ = mean_lratio/shape_
    tf = mean_lratio > 0
    ntf = tf.sum()
    sample = np.zeros(shape=(nsample, len(mean_lratio)))
    sample[:, tf] = rnd_state.gamma(shape_, scale=scale_[tf], size=(nsample, ntf))
    sample[sample > 1] = 1.0

    return sample

def sample_vulnerability_lognormal(rnd_state, mean_lratio, nsample=1000, cov=1.0):

    """
    The probability density function for `gamma` is::

        gamma.pdf(x, a) = lambda**a * x**(a-1) * exp(-lambda*x) / gamma(a)

    for ``x >= 0``, ``a > 0``. Here ``gamma(a)`` refers to the gamma function.

    The scale parameter is equal to ``scale = 1.0 / lambda``.

    `gamma` has a shape parameter `a` which needs to be set explicitly. For
    instance:

        >>> from scipy.stats import gamma
        >>> rv = gamma(3., loc = 0., scale = 2.)

    shape: a
    scale: b
    mean = a*b
    var = a*b*b
    cov = 1/sqrt(a) = 1/sqrt(shape)
    shape = (1/cov)^2
    """

    # m: mean of x, stddev: std. of x
    # mu: mean of lnx, std: std. of lnx 

    stddev = mean_lratio * cov

    # convert m, stddev -> mu, std of lnx 
    sample = np.zeros(shape=(nsample, len(mean_lratio)))
    tf = mean_lratio > 0
    ntf = tf.sum()

    mu = 2.0 * np.log(mean_lratio[tf]) - 0.5 * np.log(stddev[tf]**2.0 + mean_lratio[tf]**2.0)
    std = np.sqrt(np.log(stddev[tf]**2.0 / mean_lratio[tf]**2.0 + 1))

    sample[:, tf] = rnd_state.lognormal(mean=mu, sigma=std, size=(nsample, ntf))
    sample[sample > 1] = 1.0

    return sample


def get_fragility_by_model(bldg_str, est_mean, rnd_state, damage_thresholds, damage_labels,
                           mmi_range, nsample=1000, cov=1.0, flag_fig=False):

    nds = len(damage_labels)
    # est_mean = compute_vulnerability({'MMI': mmi_range, 'BLDG_CLASS': bldg_str})
    sampled = sample_vulnerability_lognormal(rnd_state, mean_lratio=est_mean, nsample=nsample, cov=cov)

    array_ds = np.digitize(sampled, damage_thresholds)
    prob = {}
    for i, mmi in enumerate(mmi_range):
        unique, counts = np.unique(array_ds[:, i], return_counts=True)
        prob[mmi] = dict(zip(unique, counts))

    pe_by_ds = {}
    for i, ds in enumerate(damage_labels[1:], 1): # ds
        _array = np.zeros(shape=(len(mmi_range), 2)) # mmi vs freq
        for j, (key, value) in enumerate(prob.iteritems()):
            _array[j, 0] = key
            for k in range(i + 1, nds + 1):
                if k in value:
                    _array[j, 1] += value[k]
        _array = _array[_array[:,0].argsort()]
        _array[:, 1] /= nsample
        pe_by_ds[i] = _array

    if flag_fig:
        plt.figure()
        for i, ds in enumerate(damage_labels[1:], 1):
            plt.plot(pe_by_ds[i][:, 0] , pe_by_ds[i][:, 1], '-', label=ds)
        plt.legend(loc=2)
        plt.xlabel(IMT)
        plt.ylabel('Prob. of exceedance')
        plt.title(bldg_str)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim([0, 1])
        plt.grid(1)
        plt.savefig(os.path.join(path_csv, 'frag_{}_{}.png'.format(bldg_str, IMT)), dpi=200)
        plt.close()
    # est_vulnerability_to_back
    # expected_loss_by_ds = {1: 0.0}
    # for i, val in enumerate(damage_thresholds[1:], 1):
    #     try:
    #         expected_loss_by_ds[i+1] = 0.5*(val + damage_thresholds[i+1])
    #     except IndexError:
    #         pass

    # est_vul = []
    # for mmi in mmi_range:
    #     temp = 0.0
    #     for key, val in prob[mmi].iteritems():
    #         temp += val * expected_loss_by_ds[key] / nsample
    #     est_vul.append(temp)

    # if flag_fig:
    #     plt.figure()
    #     plt.plot(mmi_range , est_mean, '-', label='model')
    #     plt.plot(mmi_range, est_vul, '--', label='estimated')
    #     plt.legend(loc=2)
    #     plt.xlabel('MMI')
    #     plt.ylabel('Loss ratio')
    #     plt.title(bldg_str)
    #     # plt.yticks(np.arange(0, 1.1, 0.1))
    #     plt.ylim([0, 0.3])
    #     plt.grid(1)
    #     plt.savefig('comp_vul_{}.png'.format(bldg_str), dpi=200)
    #     plt.close()

    return prob, pe_by_ds


if __name__ == '__main__':

    # UA1, 8, 7, 3, 5 only has population
    # C1L, S5L, S2L, Timber, URML
   if IMT == 'PGA':
        vul_functions = pd.read_csv(os.path.join(path_vul, 'vul_combined.csv'), index_col=0)
    elif IMT == 'SA03':
        print('NOT IMPLEMENTED')
    elif IMT == 'MMI':
        print('NOT IMPLEMENTED')

    intensity = vul_functions.index.tolist()
    intensity_str = ['{:.2f}'.format(x) for x in intensity]

    for bldg_str, value in vul_functions.iteritems():

        cov_factor = [value for x, value in preferred_cv_by_bldg.items() if x in key]
        if len(cov_factor) == 1:
            cov_list = [f'{x*cov_factor[0]:.1f}' for x in np.ones_like(vul.index)]
        else:
            print('SOMETHING WRONG')

        _, pe_by_ds = get_fragility_by_model(bldg_str, value, rnd_state, 
            damage_thresholds, damage_labels, intensity, nsample=50000, cov=cov, flag_fig=1)

        # save pe_by_ds to csv file
        _df = pd.DataFrame(np.zeros(shape=(len(damage_labels)-1, len(intensity))), 
                           index=damage_labels[1:], columns=intensity)
        for i, ds in enumerate(damage_labels[1:], 1):
            _df.loc[ds] = pe_by_ds[i][:, 1]
        # _df.to_csv(os.path.join(path_csv, './pe_by_ds_{}_{}_May2019.csv'.format(bldg_str, IMT)), index=True)
        _df.to_csv(os.path.join(path_csv, './pe_by_ds_{}_{}_AK_May2019.csv'.format(bldg_str, IMT)), index=True)

        # fatality for residential buildings only
        try:
            casualty_HAZUS = get_fatality_given_pe(pe_by_ds, bldg_map_to_HAZUS[bldg_str], collapse_rate, casualty_rate)
            fatality_ATC = get_fatality_given_loss_ratio_ATC(bldg_map_to_HAZUS[bldg_str], value)
        except KeyError:
            print('skip {} for fatality'.format(bldg_str))
        else:
            # save fatality to csv file
            for ids in range(1, 5):
                if ids == 4:
                    _df = pd.DataFrame(np.zeros(shape=(2, len(intensity))), index=['HAZUS','ATC'], columns=intensity_str)
                    _df.loc['HAZUS'] = casualty_HAZUS[ids][:, 1]
                    _df.loc['ATC'] =  fatality_ATC[:, 1]
                    # _df.to_csv(os.path.join(path_csv, './casualty_{}_{}_{}_May2019.csv'.format(ids, bldg_str, IMT)), index=True)
                    _df.to_csv(os.path.join(path_csv, './casualty_{}_{}_{}_AK_May2019.csv'.format(ids, bldg_str, IMT)), index=True)

                    plt.figure()
                    plt.plot(casualty_HAZUS[ids][:, 0], casualty_HAZUS[ids][:, 1], '-', label='HAZUS')
                    plt.plot(fatality_ATC[:,0], fatality_ATC[:, 1], '--', label='ATC')
                    
                    plt.xlabel(IMT)
                    plt.ylabel('Fatality rate')
                    plt.title(bldg_str)
                    plt.legend(loc=2)
                    plt.grid(1)
                    plt.savefig(os.path.join(path_csv, 'comp_fatality_{}_{}_AK_May2019.png'.format(bldg_str, IMT)), dpi=200)
                    plt.close()

                else:
                    _df = pd.DataFrame(np.zeros(shape=(1, len(intensity))), index=['HAZUS'], columns=intensity_str)
                    _df.loc['HAZUS'] = casualty_HAZUS[ids][:, 1]
                    _df.to_csv(os.path.join(path_csv, './casualty_{}_{}_{}_AK_May2019.csv'.format(ids, bldg_str, IMT)), index=True)
