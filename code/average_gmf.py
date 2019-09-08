import os
import sys
import argparse
import pandas as pd
import re

def cmd_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-file',
        nargs=1,
        metavar='input file',
        dest='input_file',
        help='Specify the input file (i.e. gmf-data_572.csv)')

    return parser

def main():

    parser = cmd_parser()
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()

        _path = os.path.abspath(os.path.dirname(args.input_file[0]))
        _file = os.path.split(args.input_file[0])[-1]
        gmf_file = os.path.join(_path, _file)
        idx = re.search('gmf-data_(.+?).csv', _file).group(1)
        rlz_file = os.path.join(_path, 'realizations_{}.csv'.format(idx))
        eid2rlz_file = os.path.join(_path, 'eid2rlz_{}.csv'.format(idx))
        site_file = os.path.join(_path, 'sitemesh_{}.csv'.format(idx))

        # read files
        gmf = pd.read_csv(gmf_file)
        eid2rlz = pd.read_csv(eid2rlz_file, index_col=0)
        if 'rlz' in eid2rlz:  # v3.6 
            eid2rlz = eid2rlz.to_dict()['rlz']
        elif 'rlz_id' in eid2rlz:  # v3.7
            eid2rlz = eid2rlz.to_dict()['rlz_id']
        else:
            sys.exit('Invalid eid2rlz file')

        weights = pd.read_csv(rlz_file, index_col=0).to_dict()['weight']
        site = pd.read_csv(site_file)

        # weighted
        imt_list = gmf.columns.tolist()
        ignored = ['eid', 'sid']
        [imt_list.remove(x) for x in ignored]
        wmt_list = ['w{}'.format(x) for x in imt_list]
        dic_wmt = {x: 'sum' for x in wmt_list}
        gmf['weight'] = gmf['eid'].apply(lambda x: weights[eid2rlz[x]])
        for imt in imt_list:
            gmf['w{}'.format(imt)] = gmf.apply(lambda x: x['weight'] *
                    x[imt], axis=1)

        # rlzi,sid,eid,gmv_PGA,gmv_SA(0.3),gmv_SA(1.0)
        df = gmf.groupby('sid').agg(dic_wmt)
        df['eid'] = gmf.loc[0, 'eid']
        df.reset_index(inplace=True)
        df.rename(index=str, columns={'w{}'.format(x): x for x in imt_list}, inplace=True)

        output_gmf = os.path.join(_path, 'gmf-data_{}_averaged.csv'.format(idx))
        df[ignored + imt_list].to_csv(output_gmf, index=False, float_format='%.6E')
        print('{} is created'.format(output_gmf))

        # all combined with site
        df.index = range(df.shape[0])
        output_all = os.path.join(_path,
                'gmf-data_{}_with_site.csv'.format(idx))
        pd.concat([site, df], axis=1).to_csv(output_all, index=False, float_format='%.6E')
        print('{} is created'.format(output_all))

if __name__ == '__main__':
    main()

