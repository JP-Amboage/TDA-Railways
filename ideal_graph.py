

import geopy
import pandas as pd
import os
import zipfile
import sys
import geopy.distance


# not working rn
# def initialize_data(path):
#     global stops
#     global indexed_stops
#     stops = pd.read_csv(path + '/stops.txt')
#     indexed_stops = stops.set_index('stop_id')

def geo_dist(lat1,lon1,lat2,lon2):
    coord1= (lat1,lon1)
    coord2= (lat2,lon2)
    return geopy.distance.distance(coord1,coord2).km


def geo_dist_row(row):
    return geo_dist(row['lat1'],row['lon1'],row['lat2'],row['lon2'])


def filter_stops(df, index_list):
    return df.loc[index_list]

def stops_product_df(stops_df):
    aux1=stops_df[['stop_lat','stop_lon']].copy()
    aux2=aux1.copy()
    index1= aux1.index
    index1.names=['loc1']
    aux1=aux1.rename(columns={"stop_lat": "lat1", "stop_lon": "lon1"})
    index2= aux2.index
    index2.names=['loc2']
    aux2=aux2.rename(columns={"stop_lat": "lat2", "stop_lon": "lon2"})
    prod_df=aux1.merge(aux2,how='cross')
    prod_index= pd.MultiIndex.from_product([aux1.index , aux2.index])
    prod_df.index=prod_index
    return prod_df.copy()

def stops_dist_df(stops_df):
    distance_df= stops_product_df(stops_df)
    distance_df['distance']=distance_df.apply(geo_dist_row, axis=1)
    return distance_df.copy()

def dist_df_to_cols(distance_df):
    aux_df=distance_df.reset_index(level=1)
    df_cols=aux_df.pivot(values='distance',columns='loc2')
    return df_cols.copy()
        
def dist_df_to_matrix(distance_df):
    return dist_df_to_cols(distance_df).to_numpy()















