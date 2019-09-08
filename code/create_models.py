# scrip to write xml file for WA project using the GA vulnerability models
# shakemap environment required

import os
import sys
import copy
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, '/home/hyeuk/Projects/oq-tools/input')
from vulnerabilityTxt2NRML import VulnerabilityTxtReader, VulnerabilityWriter
from fragilityTxt2NRML import FragilityTxtReader, FragilityWriter

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
matplotlib.rcParams['legend.numpoints'] = 1

PROJ_PATH = '/home/hyeuk/Projects/York'
HAZUS_DATA_PATH = '/home/hyeuk/Projects/hazus'
PATH_VUL = os.path.join(PROJ_PATH, 'input')
PATH_CSV = os.path.join(PROJ_PATH, 'input/fragility_csv')
if not os.path.exists(PATH_CSV):
    os.mkdir(PATH_CSV)

DAMAGE_LABELS = ['no', 'slight', 'moderate', 'extensive', 'complete']
DAMAGE_THRESHOLDS = [-1.0, 0.02, 0.1, 0.5, 0.8, 1.1]
PREFERRED_CV_BY_BLDG = {'Timber': 0.8, 'URML': 0.8, 'C1L': 1.2, 'C2L': 1.2,
        'C2H': 1.2, 'PC1L': 0.8, 'S2L': 1.2, 'S5L': 1.2, 'base': 0.8, 'full': 0.8}
MAPPING_HAZUS = pd.read_csv(os.path.join(PROJ_PATH, 'input/mapping_hazus.csv'), index_col=0).to_dict()['hazus']

VERSION = 'Sep2019'
IMT = 'PGA'

METADATA_FRAG = {'assetCategory': 'buildings',
                 'description': f'GA {IMT} based fragility models for York',
                 'lossCategory': 'structural',
                 'fragilityModelID': f'GA_{IMT}_{VERSION}',
                 'limitStates': DAMAGE_LABELS[1:]}

METADATA_VUL = {'assetCategory': 'buildings',
                'description': f'GA {IMT} based vulnerability models for York',
                'lossCategory': 'structural',
                'vulnerabilityModelID': f'GA_{IMT}{VERSION}'}

METADATA_FAT = {'assetCategory': 'buildings',
                'description': f'GA {IMT} based casualty models for York',
                'lossCategory': 'occupants',
                'vulnerabilityModelID': f'GA_{IMT}'}


def read_pe_by_ds(filename):
    a = pd.read_csv(filename, index_col=0)
    poe = []
    for line in a.values:
        poe.append(['{}'.format(x) for x in line])
    return poe


def read_hazus_collapse_rate(hazus_data_path, selected_bldg_class=None):
    """
    read hazus collapse rate parameter values
    """
    try:
        assert isinstance(selected_bldg_class, list)
    except AssertionError:
        selected_bldg_class = [selected_bldg_class]

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
    try:
        assert isinstance(selected_bldg_class, list)
    except AssertionError:
        selected_bldg_class = [selected_bldg_class]

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

        reform = {(outerKey, innerKey): values for outerKey, innerDict in dic_.items()
                  for innerKey, values in innerDict.items()}

        casualty_rate[ds] = pd.DataFrame({ds: reform}, index=multi_index)

    return casualty_rate # re-order the item

"""
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
"""

def get_casualty_given_pe(pe_by_ds, bldg_str, collapse_rate, casualty_rate, flag_fig=False):

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

        value = np.zeros_like(pb_by_ds[1])
        value[:, 0]  = pb_by_ds[1][:, 0]
        for i in pb_by_ds.keys():
            value[:, 1] += _casualty_by_ds[i] * pb_by_ds[i][:, 1]

        casualty[ids] = value

    if flag_fig:

        plt.figure()
        for ids in range(1, 5):
            plt.plot(casualty[ids][:,0], casualty[ids][:, 1], '-', label='severity {}'.format(ids))
        plt.xlabel(IMT)
        plt.ylabel('Casualty rate')
        plt.title(bldg_str)
        #plt.yticks(np.arange(0, 1.1, 0.1))
        #plt.ylim([0, 1])
        plt.grid(1)
        plt.savefig('casualty_{}.png'.format(bldg_str), dpi=200)
        plt.close()

    return casualty

#def sample_vulnerability_gamma(rnd_state, mean_lratio, nsample=1000, cov=1.0):
#
#    """
#    The probability density function for `gamma` is::
#
#        gamma.pdf(x, a) = lambda**a * x**(a-1) * exp(-lambda*x) / gamma(a)
#
#    for ``x >= 0``, ``a > 0``. Here ``gamma(a)`` refers to the gamma function.
#
#    The scale parameter is equal to ``scale = 1.0 / lambda``.
#
#    `gamma` has a shape parameter `a` which needs to be set explicitly. For
#    instance:
#
#        >>> from scipy.stats import gamma
#        >>> rv = gamma(3., loc = 0., scale = 2.)
#
#    shape: a
#    scale: b
#    mean = a*b
#    var = a*b*b
#    cov = 1/sqrt(a) = 1/sqrt(shape)
#    shape = (1/cov)^2
#    """
#
#    shape_ = np.power(1.0/cov, 2.0)
#    scale_ = mean_lratio/shape_
#    tf = mean_lratio > 0
#    ntf = tf.sum()
#    sample = np.zeros(shape=(nsample, len(mean_lratio)))
#    sample[:, tf] = rnd_state.gamma(shape_, scale=scale_[tf], size=(nsample, ntf))
#    sample[sample > 1] = 1.0
#
#    return sample

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
    stddev = cov * mean_lratio 

    # convert m, stddev -> mu, std of lnx 
    sample = np.zeros(shape=(nsample, len(mean_lratio)))
    tf = mean_lratio > 0
    ntf = tf.sum()

    mu = 2.0 * np.log(mean_lratio[tf]) - 0.5 * np.log(stddev[tf]**2.0 + mean_lratio[tf]**2.0)
    std = np.sqrt(np.log(stddev[tf]**2.0 / mean_lratio[tf]**2.0 + 1))

    sample[:, tf] = rnd_state.lognormal(mean=mu, sigma=std, size=(nsample, ntf))
    sample[sample > 1] = 1.0

    return sample


def get_fragility_by_model(bldg_str, est_mean, rnd_state, damage_thresholds, DAMAGE_LABELS,
                           mmi_range, nsample=1000, cov=1.0, flag_fig=False):

    nds = len(DAMAGE_LABELS)
    # est_mean = compute_vulnerability({'MMI': mmi_range, 'BLDG_CLASS': bldg_str})
    sampled = sample_vulnerability_lognormal(rnd_state, mean_lratio=est_mean, nsample=nsample, cov=cov)

    array_ds = np.digitize(sampled, damage_thresholds)
    prob = {}
    for i, mmi in enumerate(mmi_range):
        unique, counts = np.unique(array_ds[:, i], return_counts=True)
        prob[mmi] = dict(zip(unique, counts))

    pe_by_ds = {}
    for i, ds in enumerate(DAMAGE_LABELS[1:], 1): # ds
        _array = np.zeros(shape=(len(mmi_range), 2)) # mmi vs freq
        for j, (key, value) in enumerate(prob.items()):
            _array[j, 0] = key
            for k in range(i + 1, nds + 1):
                if k in value:
                    _array[j, 1] += value[k]
        _array = _array[_array[:,0].argsort()]
        _array[:, 1] /= nsample
        pe_by_ds[i] = _array

    if flag_fig:
        plt.figure()
        for i, ds in enumerate(DAMAGE_LABELS[1:], 1):
            plt.plot(pe_by_ds[i][:, 0] , pe_by_ds[i][:, 1], '-', label=ds)
        plt.legend(loc=2)
        plt.xlabel(IMT)
        plt.ylabel('Prob. of exceedance')
        plt.title(bldg_str)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim([0, 1])
        plt.grid(1)
        plt.savefig(os.path.join(PATH_CSV, 'frag_{}_{}.png'.format(bldg_str, IMT)), dpi=200)
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


def create_vulnerability():
    '''
    vul_GA = pd.read_csv('../input/vul_GA_refined.csv', index_col=0)
    # filter rev
    sel = [x for x in vul_GA.columns if 'rev' in x]
    vul_GA = vul_GA[sel].copy()
    vul_GA.rename(columns={x:x.strip('_rev') for x in vul_GA.columns}, inplace=True)

    vul_UA = pd.read_csv('../input/vul_UA.csv', index_col=0)
    vul_UA = vul_UA.loc[vul_GA.index].copy()

    vul = vul_UA.join(vul_GA)
    '''

    vul = pd.read_csv(os.path.join(PROJ_PATH, 'input/vul_combined.csv'), index_col=0)

    # read vulnerability for non residential bldgs
    #path_vul = '/Users/hyeuk/Projects/scenario_Guildford/input'

    #mmi_range = np.arange(4.0, 8.1, 0.25)
    #gar_mmi_orig = np.arange(4.0, 11.0, 0.05)
    #assert np.allclose(mmi_range, gar_mmi_orig[0:-57:5])

    #vul_functions = cPickle.load(open(os.path.join(path_vul, 'vul_function_non_res.p'),'rb'))
    # add two URML_RES_Pre URML_RES_Post
    #vul_functions['URML_RES_Pre'] = compute_vulnerability(
    #    {'BLDG_CLASS': 'URML_Pre', 'MMI': gar_mmi_orig})
    #vul_functions['URML_RES_Post'] = compute_vulnerability(
    #    {'BLDG_CLASS': 'URML_Post', 'MMI': gar_mmi_orig})

    """
    plt.figure()
    for key, value in sorted(vul_functions.iteritems()):
        if 'Pre' in key:
            plt.plot(gar_mmi_orig, value, label=key)

    plt.xlim([4, 8])
    plt.ylim([0, 0.6])
    plt.grid()
    plt.xlabel('MMI')
    plt.ylabel('Loss ratio')
    plt.legend(loc=2)
    plt.savefig('/Volumes/exFAT120/scenario_Perth/input/vulnerability_Pre.png', dpi=300)

    plt.figure()
    for key, value in sorted(vul_functions.iteritems()):
        if 'Post' in key:
            plt.plot(gar_mmi_orig, value, label=key)

    plt.xlim([4, 8])
    plt.ylim([0, 0.6])
    plt.grid()
    plt.xlabel('MMI')
    plt.ylabel('Loss ratio')
    plt.legend(loc=2)
    plt.savefig('/Volumes/exFAT120/scenario_Perth/input/vulnerability_Post.png', dpi=300)
    """

    imt_list = [f'{x:.2f}' for x in vul.index]
    cov_array = np.ones_like(vul.index)
    vuln_def = []
    for key, item in vul.items():

        cov_factor = [value for x, value in PREFERRED_CV_BY_BLDG.items() if x in key]
        if len(cov_factor) == 1:
            cov_list = [f'{x*cov_factor[0]:.1f}' for x in np.ones_like(vul.index)]
        else:
            print('SOMETHING WRONG')

        loss_ratio = [f'{x:.5f}' for x in item]
        idx_zeros = [i for i, e in enumerate(loss_ratio) if e == '0.00000']
        for i in idx_zeros:
            cov_list[i] = '0.0'

        tmp = {'imt': imt_list,
               'lossRatio': loss_ratio,
               'probabilityDistribution': 'LN',
               'coefficientVariation': cov_list,
               'imt_str': f'{IMT}',
               'vulnerabilityFunctionId': key}

        vuln_def.append(tmp)

    writer = VulnerabilityWriter()
    outfile = os.path.join(PROJ_PATH, f'input/vulnerability_GA_{IMT}_{VERSION}.xml')
    try:
        writer.serialize(outfile, METADATA_VUL, vuln_def)
    except Exception as e:
        print(e)
    else:
        print(f'{outfile} is created')

    return vul

def get_UA_fragility_from_excel():

    sheets = [f'GBT{i}' for i in range(1, 7)]
    names=['PGA','slight','moderate','extensive','complete']

    for sheet in sheets:
        a = pd.read_excel(os.path.join(PROJ_PATH, 'input/Numerical fragility curves.xlsx'),
                          sheet_name=sheet, skiprows=6, nrows=29,
                          usecols=[1, 8, 9, 10, 11], index_col=0, names=names)

        b = pd.read_excel(os.path.join(PROJ_PATH, 'input/Numerical fragility curves.xlsx'),
                          sheet_name=sheet, skiprows=53, nrows=29,
                          usecols=[1, 8, 9, 10, 11], index_col=0, names=names)

        file1 = f'pe_by_ds_{sheet[-2:]}base_PGA_Sep2019_UA.csv'
        file2 = f'pe_by_ds_{sheet[-2:]}full_PGA_Sep2019_UA.csv'

        a.T.to_csv(os.path.join(PROJ_PATH, 'input', 'fragility_csv', file1))
        b.T.to_csv(os.path.join(PROJ_PATH, 'input', 'fragility_csv', file2))

def create_fragility_and_casualty(vul_functions):

    intensity = vul_functions.index
    intensity_str = [f'{x:.2f}' for x in intensity]
    rnd_state = np.random.RandomState(1)

    for bldg_str, value in vul_functions.items():
        if ('base' in bldg_str) or ('full' in bldg_str):
            filename = f'pe_by_ds_{bldg_str}_{IMT}_{VERSION}_UA.csv'
            tmp = pd.read_csv(os.path.join(PATH_CSV, filename), index_col=0)
            pe_by_ds = {}
            for i, (key, value) in enumerate(tmp.iterrows(), 1):
                pe_by_ds[i] = np.array([intensity, value.values]).T
            print(f'Reading {filename} instead')
        else:
            cov_factor = [value for x, value in PREFERRED_CV_BY_BLDG.items() if x in bldg_str]
            if len(cov_factor) == 1:
                cov_list = [f'{x*cov_factor[0]:.1f}' for x in np.ones_like(vul.index)]
                cov = float(cov_list[0])
                print(f'{bldg_str}: {cov}')
            else:
                print('SOMETHING WRONG')
            _, pe_by_ds = get_fragility_by_model(bldg_str, value.values, rnd_state,
                DAMAGE_THRESHOLDS, DAMAGE_LABELS, intensity, nsample=50000, cov=cov, flag_fig=1)

            # save pe_by_ds to csv file
            _df = pd.DataFrame(np.zeros(shape=(len(DAMAGE_LABELS)-1, len(intensity))),
                               index=DAMAGE_LABELS[1:], columns=intensity)
            for i, ds in enumerate(DAMAGE_LABELS[1:], 1):
                _df.loc[ds] = pe_by_ds[i][:, 1]
            _df.to_csv(os.path.join(PATH_CSV, f'pe_by_ds_{bldg_str}_{IMT}_{VERSION}.csv'), index=True)

        # casualty 
        try:
            hazus_str = MAPPING_HAZUS[bldg_str]
            casualty_rate = read_hazus_casualty_data(HAZUS_DATA_PATH, hazus_str)
            collapse_rate = read_hazus_collapse_rate(HAZUS_DATA_PATH, hazus_str)

            casualty_HAZUS = get_casualty_given_pe(pe_by_ds, hazus_str, collapse_rate, casualty_rate)
            #fatality_ATC = get_fatality_given_loss_ratio_ATC(bldg_map_to_HAZUS[bldg_str], value)
        except KeyError:
            print(f'something wrong with {bldg_str} for casualty')
        else:
            # save casualty to csv file
            for ids in range(1, 5):
                if ids == 4:
                    _df = pd.DataFrame(np.zeros(shape=(1, len(intensity))), index=['HAZUS'], columns=intensity_str)
                    _df.loc['HAZUS'] = casualty_HAZUS[ids][:, 1]
                    _df.to_csv(os.path.join(PATH_CSV, f'casualty_{ids}_{bldg_str}_{IMT}_{VERSION}.csv'), index=True)

                    plt.figure()
                    plt.plot(casualty_HAZUS[ids][:, 0], casualty_HAZUS[ids][:, 1], '-', label='HAZUS')
                    plt.xlabel(IMT)
                    plt.ylabel('Fatality rate')
                    plt.title(bldg_str)
                    plt.legend(loc=2)
                    plt.grid(1)
                    plt.savefig(os.path.join(PATH_CSV, f'comp_casualty_{bldg_str}_{IMT}_{VERSION}.png'), dpi=200)
                    plt.close()

                else:
                    _df = pd.DataFrame(np.zeros(shape=(1, len(intensity))), index=['HAZUS'], columns=intensity_str)
                    _df.loc['HAZUS'] = casualty_HAZUS[ids][:, 1]
                    _df.to_csv(os.path.join(PATH_CSV, f'casualty_{ids}_{bldg_str}_{IMT}_{VERSION}.csv'), index=True)



def write_fragility_xml(flag_fig=False):

    frag_def = []
    for _file in os.listdir(PATH_CSV):
        if ('pe_by_ds' in _file) and (IMT in _file):

            idx = re.search('pe_by_ds_(.+?).csv',_file).group(1)
            bldg = '_'.join(x for x in idx.split('_')[:2])
            value = read_pe_by_ds(os.path.join(PATH_CSV, _file))
            imt_list = pd.read_csv(os.path.join(PATH_CSV, _file),
                                index_col=0).columns.tolist()

            tmp = {'imt': imt_list,
                   'poe': value,
                   'frag_format': 'discrete',
                   'nodamage_limit': '0.01',
                   'imt_str': f'{IMT}',
                   'fragilityFunctionId': bldg}

            frag_def.append(tmp)

        # check plot
        if flag_fig:

            x = [float(x) for x in imt_list]
            plt.figure()
            for i in range(4):
                y = [float(z) for z in value[i]]
                plt.plot(x, y, label=i)
            plt.legend(loc=2)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.ylim([0, 1])
            plt.grid(1)
            plt.savefig(os.path.join(PATH_CSV, f'frag_{bldg}_{IMT}.png'))
            plt.close()

    writer = FragilityWriter()
    outfile = os.path.join(PROJ_PATH, f'input/fragility_GA_{IMT}_{VERSION}.xml')
    try:
        writer.serialize(outfile, METADATA_FRAG, frag_def)
    except Exception as e:
        print(e)
    else:
        print(f'{outfile} is created')

def write_casualty_xml():

    # read casualty for residential bldgs
    fat_functions = {}
    for _file in os.listdir(PATH_CSV):
        if ('casualty' in _file) and (IMT in _file) and ('csv' in _file):
            idx = re.search('casualty_(.+?).csv',_file).group(1)
            ids = int(idx.split('_')[0])
            bldg = '_'.join(x for x in idx.split('_')[1:3])
            filename = os.path.join(PATH_CSV, _file)
            fat_functions.setdefault(ids, {})[bldg] = pd.read_csv(filename, index_col=0).loc['HAZUS'].values
            imt_list = pd.read_csv(filename, index_col=0).columns.tolist()

    zeros_list = ['{}'.format(x) for x in np.zeros(len(imt_list))]

    # write
    for ids, item in fat_functions.items():

        vuln_def = []
        for key, value in item.items():
            loss_ratio = ['{}'.format(x) for x in value]

            tmp = {'imt': imt_list,
                  'lossRatio': loss_ratio,
                  'probabilityDistribution': 'LN',
                  'coefficientVariation': zeros_list,
                  'imt_str': f'{IMT}',
                  'vulnerabilityFunctionId': key}

            vuln_def.append(tmp)

        writer = VulnerabilityWriter()
        outfile = os.path.join(PROJ_PATH, f'input/casualty_GA_severity_{ids}_{IMT}_{VERSION}.xml')
        try:
            writer.serialize(outfile, METADATA_FAT, vuln_def)
        except Exception as e:
            print(e)
        else:
            print(f'{outfile} is created')

if __name__ == '__main__':

    # create vulnerability and write in xml
    vul = create_vulnerability()

    # use UA fragility
    get_UA_fragility_from_excel()

    # create fragility and casualty 
    create_fragility_and_casualty(vul)

    # write fragility xml
    write_fragility_xml(flag_fig=1)

    # write casualty xml
    write_casualty_xml()
    # UA1, 8, 7, 3, 5 only has population
    # C1L, S5L, S2L, Timber, URML

