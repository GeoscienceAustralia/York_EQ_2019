import os
import pandas as pd
import numpy as np

PROJ_PATH = '/home/hyeuk/Projects/York'
RETROFITS = [10, 20, 30]
RETROFIT_SCHEME = [1, 2]
NEXIS_MAPPING = {'id': 'Corrected UFI',
                 'lon': 'Longitude',
                 'lat': 'Latitude',
                 'taxonomy': 'class',
                 'structural': 'Final_rep_cost',
                 'night': 'Final_pop',
                 'number': 'number',
                 'heritage': 'Heritage',
                 'retrofit': 'retrofit',
                 'SA1': 'Final_SA1'}
dic_wall = {
    'Brick (stretcher bond)': 'URM',
    'Fibro': 'Timber',
    'Unassessable': 'Timber',
    'Weatherboard': 'Timber',
    'Metal': 'Timber',
    'Rendered (unknown)': 'URM',
    'Rendered (brick)': 'URM',
    'Block': 'Timber',
    'Painted brick': 'URM',
    'Brick (header bond)': 'URM',
    'Stone (course ashlar)': 'URM',
    'Stone (broken ashlar)': 'URM',
    'Stone (random rubble)': 'URM',
    'Stone (course rubble)': 'URM',
    'Painted stone': 'URM',
    'Rendered (stone)': 'URM',
    'Rammed earth': 'URM',
    'Rendered masonry': 'URM',
    'Unknown': 'URM'}

WALLS = {}
WALLS['URM']= [key for key, value in dic_wall.items() if value == 'URM']
WALLS['Timber']= [key for key, value in dic_wall.items() if value == 'Timber']


def assign_GA_class(row):

    bldg_type = row['Generic building type']
    lower_wall_material = row['Lower Wall Material']
    bldg_code = row['Building Code']
    vintage = row['Vintage']
    if bldg_code in ['SD', 'SH']:
        if float(bldg_type) > 0:
            return bldg_type
        else:
            if lower_wall_material in WALLS['URM']:
                return 'URM_{}'.format(vintage)
            elif lower_wall_material in WALLS['Timber']:
                return 'Timber_{}'.format(vintage)
            else:
                return 'None'
    else:
        if float(bldg_type) > 0:
            return bldg_type
        else:
            return '{}_{}'.format(bldg_code, vintage)

def write_csv_retrofit(exposure, scheme):

    for year in RETROFITS:
        df = {}
        for key, value in NEXIS_MAPPING.items():
            df[key] = exposure[value]
        df = pd.DataFrame(df)
        df['retrofit'] = scheme
        df['taxonomy'].replace({key: 'UA{:d}_base'.format(key) for key in np.arange(1, 9)}, inplace=True)
        idx = exposure[f'Retrofit cohort {scheme}'] <= year
        df.loc[idx, 'taxonomy'] = df.loc[idx, 'taxonomy'].apply(lambda x: x.replace('base', 'full'))
        output_file = os.path.join(PROJ_PATH, 'input', f'exposure_York_R{scheme}_{year:d}.csv')
        df.to_csv(output_file, index=False)
        print(f'{output_file} is written')

def write_csv(exposure):

    df = {}
    for key, value in NEXIS_MAPPING.items():
        df[key] = exposure[value]
    df = pd.DataFrame(df)
    df['taxonomy'].replace({key: 'UA{:d}_base'.format(key) for key in np.arange(1, 9)}, inplace=True)
    #idx = exposure[f'Retrofit cohort {scheme}'] <= year
    #df.loc[idx, 'taxonomy'] = df.loc[idx, 'taxonomy'].apply(lambda x: x.replace('base', 'full'))
    output_file = os.path.join(PROJ_PATH, 'input', f'exposure_York_0_0.csv')
    df.to_csv(output_file, index=False)
    print(f'{output_file} is written')


def main(exposure):

    assert (exposure['Retrofit cohort 1'].value_counts().values == [15, 15, 15]).all()
    assert (exposure['Retrofit cohort 2'].value_counts().values == [30, 30, 30]).all()

    # add number
    exposure['number'] = 1.0

    # add retrofit
    exposure['retrofit'] = 0

    # H vs NH : Heritage category
    exposure['Heritage Category'].fillna('-99', inplace=True)
    exposure['Heritage'] = exposure['Heritage Category'].apply(lambda x: 'NH' if x == '-99' else 'H')

    # change 
    exposure['Generic building type'].fillna('-99', inplace=True)

    # In[24]:
    exposure['class'] = exposure.apply(assign_GA_class, axis=1)
    #exposure['class'].value_counts()

    #exposure['class'].isin([x for x in np.arange(1, 9)]).sum()

    assert (exposure['Retrofit cohort 1'] >= 10.0).sum() == 45
    assert (exposure['Retrofit cohort 2'] >= 10.0).sum() == 90

    # write to csv
    write_csv(exposure)

    for scheme in RETROFIT_SCHEME:
        write_csv_retrofit(exposure, scheme)



if __name__=='__main__':

    input_file = os.path.join(PROJ_PATH, 'input/York_Building_exposure_spot_checked_with_generic_types.xlsx')
    assert os.path.exists(input_file)
    exposure = pd.read_excel(input_file, nrows=1417)
    assert exposure.shape[0] == 1417
    main(exposure)

