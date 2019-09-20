import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

PROJ_PATH = '/home/hyeuk/Projects/York'

RETURN_PERIODS = [500, 1000, 2500]
LEVELS = range(1, 5)
SCHEME = [1, 2]
RETROFITS = [10, 20, 30]

OUTPUT_PATH = 'output_37n'
UNIT = 1.0e+6  # convert to M AUD
DS_DIC = {f'structural~{ds}': 'sum' for ds in ['slight', 'moderate', 'extensive', 'complete']}

def summary_gmf():

    gmf_output = os.path.join(PROJ_PATH, 'doc', 'gmf_combined_new.csv')

    # ROCK, SOIL
    gmf = {}
    for i, rp in enumerate(RETURN_PERIODS):
        _path = os.path.join(PROJ_PATH, f'Rp{rp}', OUTPUT_PATH)
        gmf_rock_file = os.path.join(_path, f'gmf-data_{37*i+1}_with_site.csv')
        gmf_soil_file = os.path.join(_path, f'gmf-data_{37*i+2}_with_site.csv')

        assert os.path.exists(gmf_rock_file), print(f'Invalid gmf: {gmf_rock_file}')
        assert os.path.exists(gmf_soil_file), print(f'Invalid gmf: {gmf_soil_file}')

        rock = pd.read_csv(gmf_rock_file)
        soil = pd.read_csv(gmf_soil_file)

        if i == 0:
            gmf['site_id'] = rock['site_id'].values
            gmf['lon'] = rock['lon'].values
            gmf['lat'] = rock['lat'].values

        gmf[f'rock_{rp}'] = rock['gmv_PGA'].values
        gmf[f'soil_{rp}'] = soil['gmv_PGA'].values

    # save
    pd.DataFrame(gmf).to_csv(gmf_output, index=False)
    print(f'{gmf_output} written')

def summary_current():

    loss_output = os.path.join(PROJ_PATH, 'doc', f'loss_current.csv')
    loss_group_output = os.path.join(PROJ_PATH, 'doc', f'loss_group_current.csv')
    fat_output = os.path.join(PROJ_PATH, 'doc', f'fat_current.csv')
    dmg_output = os.path.join(PROJ_PATH, 'doc', f'dmg_current.csv')
    dmg_group_output = os.path.join(PROJ_PATH, 'doc', f'dmg_group_current.csv')

    # as-it-is
    loss = []
    loss_group = []
    fat = []
    df_dmg = pd.DataFrame(None)
    df_dmg_group = pd.DataFrame(None)

    for i, rp in enumerate(RETURN_PERIODS):

        _path = os.path.join(PROJ_PATH, f'Rp{rp}', OUTPUT_PATH)
        fat_by_level, loss_by_level, loss_group_by_level = [], [], []

        # risk file
        for k, level in enumerate(LEVELS):
            # 3, 4, 5, 6
            #40, 41, 42, 43
            # 77, 78, 79, 80
            job_id = 37*i + 3 + k
            result_file = os.path.join(_path, f'agglosses_{job_id}.csv')
            assert os.path.exists(result_file), print(f'Invalid aggloss file {job_id}')

            tmp = pd.read_csv(result_file, skiprows=1)
            fat_by_level.append(tmp.loc[0, 'mean'])
            loss_by_level.append(tmp.loc[1, 'mean'])

            result_file = os.path.join(_path, f'losses_by_asset-rlz-000_{job_id}.csv')
            assert os.path.exists(result_file), print(f'Invalid loss_by_asset file {job_id}')
            tmp = pd.read_csv(result_file, skiprows=1)
            tmp = tmp.groupby('heritage').agg({'structural~mean': np.sum})/UNIT
            loss_group_by_level.append(tmp.loc['H', 'structral~mean'])
            loss_group_by_level.append(tmp.loc['NH', 'structral~mean'])

        loss.append(loss_by_level)
        loss_group.append(loss_group_by_level)
        fat.append(fat_by_level)

        # dmg_file
        job_id = 37*i + 7
        result_file = os.path.join(_path, f'dmg_by_event_{job_id}.csv')
        assert os.path.exists(result_file), print(f'Invalid dmg file: {job_id}')

        tmp = pd.read_csv(result_file)
        tmp.index = [f'Rp{rp}']
        df_dmg = df_dmg.append(tmp[DS_DIC].round())

        # group_by
        result_file = os.path.join(_path, f'dmg_by_asset-rlz-000_{job_id}.csv')
        assert os.path.exists(result_file), print(f'Invalid dmg_by_asset file: {job_id}')

        tmp = pd.read_csv(result_file)
        tmp = tmp.groupby('heritage').agg(DS_DIC)
        #tmp.index = [f'Rp{rp}']
        df_dmg_group = df_dmg_group.append(tmp.round())

    # save loss
    df_loss = pd.DataFrame(loss)
    df_loss.columns = ['L1','L2','L3','L4']
    df_loss.to_csv(loss_output)
    print(f'{loss_output} written')

    df_loss_group = pd.DataFrame(loss_group)
    df_loss_group.columns = ['L1','L2','L3','L4']
    df_loss_group.to_csv(loss_group_output)
    print(f'{loss_group_output} written')

    df_fat = pd.DataFrame(fat)
    df_fat.columns = ['L1','L2','L3','L4']
    df_fat.to_csv(fat_output)
    print(f'{fat_output} written')

    # save dmg
    df_dmg.to_csv(dmg_output)
    print(f'{dmg_output} written')

    df_dmg_group.to_csv(dmg_group_output)
    print(f'{dmg_group_output} written')

def summary_retrofit(scheme):

    dic_scheme = {1: 0,
                  2: 15,
                 }

    # retrofitted
    for i, rp in enumerate(RETURN_PERIODS):

        _path = os.path.join(PROJ_PATH, f'Rp{rp}', OUTPUT_PATH)

        loss = []
        fat = []
        df_dmg = pd.DataFrame(None)

        loss_output = os.path.join(PROJ_PATH, 'doc', f'loss_Rp{rp}_s{scheme}.csv')
        fat_output = os.path.join(PROJ_PATH, 'doc', f'fat_Rp{rp}_s{scheme}.csv')
        dmg_output = os.path.join(PROJ_PATH, 'doc', f'dmg_Rp{rp}_s{scheme}.csv')

        for j, retro in enumerate(RETROFITS):
            fat_by_level, loss_by_level = [], []

            for k, level in enumerate(LEVELS):
                # risk file
                job_id = 37*i + 5*j + 8 + k + dic_scheme[scheme]
                result_file = os.path.join(_path, f'agglosses_{job_id}.csv')
                assert os.path.exists(result_file), print(f'Invalid loss file {job_id}')

                tmp = pd.read_csv(result_file, skiprows=1)
                fat_by_level.append(tmp.loc[0, 'mean'])
                loss_by_level.append(tmp.loc[1, 'mean'])

            loss.append(loss_by_level)
            fat.append(fat_by_level)

            # dmg_file
            job_id = 37*i + 5*j + 12 + dic_scheme[scheme]
            result_file = os.path.join(_path, f'dmg_by_event_{job_id}.csv')
            assert os.path.exists(result_file), print(f'Invalid dmg file: {job_id}')

            tmp = pd.read_csv(result_file)
            tmp.index = [f'Rp{rp}']
            df_dmg = df_dmg.append(tmp)

        # save loss
        df_loss = pd.DataFrame(loss)
        df_loss.columns = ['L1','L2','L3','L4']
        df_loss.index = ['Y10','Y20','Y30']
        df_loss.to_csv(loss_output)
        print(f'{loss_output} written')

        df_fat = pd.DataFrame(fat)
        df_fat.columns = ['L1','L2','L3','L4']
        df_fat.index = ['Y10','Y20','Y30']
        df_fat.to_csv(fat_output)
        print(f'{fat_output} written')

        # save dmg
        df_dmg.to_csv(dmg_output)
        print(f'{dmg_output} written')

    # Rp500 - 1, 2
    # current
    # 3, 4, 5, 6
    # dmg 7
    # R1
    # 3 retrofit (10, 20, 30)
    # risk 8, 9, 10, 11
    # dmg 12
    # risk 13, 14, 15, 16
    # dmg 17
    # risk 18, 19, 20, 21
    # dmg 22
    # 3 retrofit (10, 20, 30)
    # risk 23, 24, 25, 26
    # dmg 27
    # risk 28, 29 , 30 , 31
    # dmg 32
    # risk 33, 34, 35, 36
    # dmg 37

    # Rp1000
    # 38, 39
    # Rp2500 - 
    # 75, 76
    # 77, 78, 79, 80 
    # 81
    # schem 1
    # 82, 83, 84, 85
    # 86
    # 87, 88, 89, 90
    # 91
    # 92, 93, 94, 95
    # 96
    # sheme 2
    # 97, 98, 99, 100
    # 101
    # 102, 103, 104, 105
    # 106
    # 107, 108, 109, 110
    # 111

def estimate_rarity():

    # estimate_rarity of three events
    loss_curve = pd.read_csv('../PSRA/output/agg_curves-stats_68.csv', skiprows=1)
    loss_curve = loss_curve.loc[1:].copy()

    y_ = np.array([float(x) for x in loss_curve['loss_value']])
    x_ = np.array([float(x) for x in loss_curve['return_period']])
    plt.figure()
    plt.semilogx(x_, y_/1.0e+6,'-')
    f = interpolate.interp1d(y_, np.log(x_))
    plt.xlabel('Return period')
    plt.ylabel('Loss (M AUD)')
    plt.savefig(os.path.join(PROJ_PATH, 'doc/loss_curve.png'),dpi=300)

    # check classic vs. event-based
    cl_loss = pd.read_csv('../PSRA/output/avg_losses-mean_70.csv', skiprows=1)
    ev_loss = pd.read_csv('../PSRA/output/avg_losses-mean_68.csv', skiprows=1)

    ratio = ev_loss['structural'].sum()/cl_loss['structural'].sum()
    print(f'classic to event: {ratio:.3f}')

    # AAL
    aal_loss = pd.read_csv('../PSRA/output/avg_losses-mean_70.csv', skiprows=1)

    # exposure
    exposure = pd.read_csv('../input/exposure_York_0_0.csv')
    aal_percent = 100*aal_loss['structural'].sum()/exposure['structural'].sum()
    print(f'AAL(%): {aal_percent}')
    print(f'AAL(400K): {4000*aal_percent}')

    # scenarios
    for i, rt in enumerate(RETURN_PERIODS):
        id_ = 37*i + 3
        path_ = f'Rp{rt}'
        _file = os.path.join(PROJ_PATH, path_, OUTPUT_PATH, f'agglosses_{id_}.csv')

        try:
            assert os.path.exists(_file)
        except AssertionError:
            print(f'missing {_file}')
        else:
            value = pd.read_csv(_file, index_col=1, skiprows=1).loc['structural', 'mean']
            try:
                est_rp = np.exp(f(value))
            except ValueError:
                print(f'{path_}: value:{value}')
            else:
                print(f'{path_}: value:{value/1.0e+6}(M$):{value/exposure["structural"].sum()} -> Rp:{est_rp}')
                plt.plot(est_rp, value/1.0e+6, 'o', label=f'{path_}:{est_rp:.0f}')

    plt.legend(loc=2)
    plt.xlabel('Return period')
    plt.ylabel('Loss (M AUD)')
    #plt.yticks(np.arange(0.0, 30000, 1000))
    plt.grid(1)
    plt.savefig(os.path.join(PROJ_PATH, 'doc/loss_curve_vs_scenarios.png'), dpi=300)
    plt.close()

if __name__=="__main__":
    #summary_gmf()
    summary_current()
    #for scheme in SCHEME:
    #    summary_retrofit(scheme)
    #estimate_rarity()
