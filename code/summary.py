import os
import pandas as pd

PROJ_PATH = '/home/hyeuk/Projects/York'

RETURN_PERIODS = [500, 1000, 2500]
LEVELS = range(1, 5)
RETROFITS = [0, 10, 20, 30]

gmf_output = os.path.join(PROJ_PATH, 'doc', 'gmf_combined.csv')

# ROCK, SOIL
gmf = {}
for i, rp in enumerate(RETURN_PERIODS):
    _path = os.path.join(PROJ_PATH, f'Rp{rp}', 'output_37')
    gmf_rock_file = os.path.join(_path, f'gmf-data_{22*i+1}_with_site.csv')
    gmf_soil_file = os.path.join(_path, f'gmf-data_{22*i+2}_with_site.csv')

    assert os.path.exists(gmf_rock_file), print(f'Invalid gmf: {gmf_rock_file}')
    assert os.path.exists(gmf_soil_file), print(f'Invalid gmf: {gmf_soil_file}')

    rock = pd.read_csv(gmf_rock_file)
    soil = pd.read_csv(gmf_soil_file)

    gmf[f'rock_{rp}'] = rock['gmv_PGA'].values
    gmf[f'soil_{rp}'] = soil['gmv_PGA'].values

    if i == 2:
        gmf['site_id'] = rock['site_id'].values
        gmf['lon'] = rock['lon'].values
        gmf['lat'] = rock['lat'].values

    loss = []
    fat = []
    df_dmg = pd.DataFrame(None)

    loss_output = os.path.join(PROJ_PATH, 'doc', f'loss_Rp{rp}.csv')
    fat_output = os.path.join(PROJ_PATH, 'doc', f'fat_Rp{rp}.csv')
    dmg_output = os.path.join(PROJ_PATH, 'doc', f'dmg_Rp{rp}.csv')

    for j, retro in enumerate(RETROFITS):
        fat_by_level, loss_by_level = [], []

        for k, level in enumerate(LEVELS):
            # risk file
            job_id = 22*i + 5*j + 3 + k
            result_file = os.path.join(_path, f'agglosses_{job_id}.csv')
            assert os.path.exists(result_file), print(f'Invalid loss file {job_id}')

            tmp = pd.read_csv(result_file, skiprows=1)
            fat_by_level.append(tmp.loc[0, 'mean'])
            loss_by_level.append(tmp.loc[1, 'mean'])

        loss.append(loss_by_level)
        fat.append(fat_by_level)

        # dmg_file
        job_id = 22*i + 5*j + 7
        result_file = os.path.join(_path, f'dmg_by_event_{job_id}.csv')
        assert os.path.exists(result_file), print(f'Invalid dmg file: {job_id}')

        tmp = pd.read_csv(result_file)
        tmp.index = [f'Rp{rp}']
        df_dmg = df_dmg.append(tmp)

    # save loss
    df_loss = pd.DataFrame(loss)
    df_loss.columns = ['L1','L2','L3','L4']
    df_loss.index = ['Y0','Y10','Y20','Y30']
    df_loss.to_csv(loss_output)

    df_fat = pd.DataFrame(fat)
    df_fat.columns = ['L1','L2','L3','L4']
    df_fat.index = ['Y0','Y10','Y20','Y30']
    df_fat.to_csv(fat_output)

    # save dmg
    df_dmg.to_csv(dmg_output)

# save
pd.DataFrame(gmf).to_csv(gmf_output, index=False)
# Rp500 - 1, 2
# 4 retrofit (0, 10, 20, 30)
# risk 3, 4, 5, 6 
# dmg 7
# risk 8, 9, 10, 11
# dmg 12
# risk 13, 14, 15, 16
# dmg 17
# risk 18, 19, 20, 21
# dmg 22

# Rp2500 - 
# 45, 46
# 47, 48, 49, 50 
# 51 
# 52, 53, 54, 55
# 56
# 57, 58, 59, 60
# 61
# 62, 63, 64, 65
# 66
