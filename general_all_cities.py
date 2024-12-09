from general_script import main


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
    filtered_dir = './Data/filtered/'
    general_output_dir = './Data/outputs/'
    for city in cities:
        print(f'[START] Processing {city}')

        gtfs_path = f'{filtered_dir}{city}_coords.zip'
        out_path = f'{general_output_dir}{city}'
        
        print()
        main(gtfs_path, out_path, city)
        print()

        print(f'[Finished] Processing {city}')
        print()
        print()