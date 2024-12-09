import pandas as pd
cities = [
    'fribourg',
    'winterthur',
    'neuchatel',
    'schauffhausen',
    'sion',
    'montreux',
    'baar',
    'yberdon_les_bains',
    'caurouge',
    'wadenswil',
    'grenchen',
    'zug',
    'lucerne',
    'st_gallen',
    'lugano',
    'biel',
    'thun',
    'koniz',
    'la_chaux_de_fonds',
    'chur',
    'vernier',
    'uster',
    'lancy',
    'dubendorf',
    'rapperswil_jona',
    'zurich',
    'geneva',
    'basel',
    'bern',
    'lausanne'
]
if __name__ == '__main__':
    all_dfs = []
    output_dir = './Data/outputs/'
    for city in cities:
        df = pd.read_csv(f'{output_dir}{city}/{city}_features.csv')
        df = df.set_index('city')
        all_dfs.append(df)

    all_features = pd.concat(all_dfs)
    all_features.to_csv(f'{output_dir}all_topo_features.csv')