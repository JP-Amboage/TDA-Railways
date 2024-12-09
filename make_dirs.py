import os
from filer_by_coords import filter_fun

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

    parent_dir = './Data/outputs/'

    print(f'Creating directories for {len(cities)} cities inside {parent_dir}')
    for city in cities:
        os.makedirs(parent_dir+city, exist_ok=True)
    
    filtered_dir = './Data/filtered/'
    in_path = 'Data/gtfs_fp2024_2024-10-28'
    os.makedirs(filtered_dir, exist_ok=True)
    
    print(f'Creating filtered GTFS for {len(cities)} cities inside {filtered_dir}')
    for city in cities:
        out_path = f'{filtered_dir}{city}_coords.zip'
        filter_fun(city, in_path, out_path)
        print(f'{city} done')