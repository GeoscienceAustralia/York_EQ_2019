import pandas as pd
import sys
import os
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME = os.environ['HOME']
if 'u65242' in HOME:
    PROJ_PATH = os.path.join('c:/', 'W10Dev', 'York')
else:
    PROJ_PATH = os.path.join(HOME, 'Projects/York')

DEFAULT_REFERENCE = os.path.join(PROJ_PATH, 'hazard_curve-mean-PGA_116.7683E_31.8912S_York.csv')

def plot_hazard_curve(haz_file, ref_hc):

    ref_hc = pd.read_csv(ref_hc)

    """
    ref_hc2 = pd.read_csv(
        os.path.join(PROJ_PATH, 'nsha18_haz_curves_PGA.csv'))
    ref_hc2.drop(['Unnamed: 1','Unnamed: 2'], axis=1, inplace=True)
    ref_hc2 = ref_hc2.loc[ref_hc2['RETURN PERIODS']=='York'].transpose()
    ref_hc2 = ref_hc2.iloc[1:].copy()
    ref_hc2['AEP'] = ref_hc2.apply(lambda x: 1/float(x.name), axis=1)
    """

    oq = pd.read_csv(haz_file, skiprows=1)
    sel = [x for x in oq.columns.tolist() if 'poe' in x]
    oq = oq[sel].transpose()
    oq['PGA'] = oq.apply(lambda x: float(x.name.strip('poe-')), axis=1)
    oq['lambda'] = oq[0].apply(lambda x: - np.log(1.0-x)/50.0)
    oq['Rp'] = 1/oq['lambda']

    plt.figure()
    plt.loglog(ref_hc['HAZARD_LEVEL(g)'], ref_hc['ANNUAL_PROBABILITY'],'*-',
               # ref_hc2[13], ref_hc2['AEP'],'.-',
               oq['PGA'], oq['lambda'],'o--')
    plt.xlabel('PGA(g)')
    plt.ylabel('Annual rate of exceedance')
    plt.legend(['reference','hc'])
    idx = re.search('hazard_curve-mean-PGA_(.+?).csv', haz_file).group(1)
    output_file = os.path.join(PROJ_PATH, 'hc_s{}.png'.format(idx))
    plt.savefig(output_file, dpi=300)
    plt.close()
    print('{} is created'.format(output_file))

def cmd_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--hazard_curve',
        nargs=1,
        metavar='hazard curve',
        dest='hazard_curve',
        help='Specify the hazard curve file (i.e. hazard_curve-mean-PGA_643.csv)')
    parser.add_argument('-r', '--reference',
        nargs=1,
        metavar='reference',
        dest='reference',
        help='Specify the reference file (i.e. hazard_curve-mean-PGA_116.7683E_31.8912S_York.csv)')

    return parser


def main():

    parser = cmd_parser()
    if len(sys.argv) < 3:
        parser.print_help()
    else:
        args = parser.parse_args()
        hazard_curve = args.hazard_curve[0]
        try:
            ref_curve = args.reference[0]
        except TypeError:
            ref_curve = DEFAULT_REFERENCE
        plot_hazard_curve(hazard_curve, ref_curve)

if __name__ == '__main__':
    main()


