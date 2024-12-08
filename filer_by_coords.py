#%%
import pandas as pd
import os
import zipfile
import sys

zurich_coords = {'min_lat': 47.320348, 'max_lat': 47.434638, 'min_lon': 8.448019, 'max_lon': 8.624798}
geneva_coords = {'min_lat': 46.177942, 'max_lat': 46.231921, 'min_lon': 6.110178, 'max_lon': 6.175809}
basel_coords = {'min_lat': 47.519295, 'max_lat': 47.589862, 'min_lon': 7.554643, 'max_lon': 7.634138}
bern_coords = {'min_lat': 46.919078, 'max_lat': 46.990122, 'min_lon': 7.294518, 'max_lon': 7.495514}
lausanne_coords = {'min_lat': 46.50439, 'max_lat': 46.551218, 'min_lon': 6.58409, 'max_lon': 6.67426}

lucerne_coords = {'min_lat': 47.010, 'max_lat': 47.090, 'min_lon': 8.220, 'max_lon': 8.360}
stgallen_coords = {'min_lat': 47.390, 'max_lat': 47.455, 'min_lon': 9.270, 'max_lon': 9.450}
lugano_coords = {'min_lat': 45.950, 'max_lat': 46.050, 'min_lon': 8.900, 'max_lon': 9.020}
biel_coords = {'min_lat': 47.110, 'max_lat': 47.180, 'min_lon': 7.180, 'max_lon': 7.320}
thun_coords = {'min_lat': 46.700, 'max_lat': 46.790, 'min_lon': 7.560, 'max_lon': 7.690}

fribourg_coords = {'min_lat': 46.784883, 'max_lat': 46.821603, 'min_lon': 7.136037, 'max_lon': 7.184445}
winterthur_coords = {'min_lat': 47.44979, 'max_lat': 47.548369, 'min_lon': 8.656353, 'max_lon': 8.810333}
neuchatel_coords = {'min_lat': 46.977977, 'max_lat': 47.030656, 'min_lon': 6.845157, 'max_lon': 6.969611}
schauffhausen_coords = {'min_lat': 47.685409, 'max_lat': 47.752821, 'min_lon': 8.540467, 'max_lon': 8.703519}
sion_coords = {'min_lat': 46.17877, 'max_lat': 46.256007, 'min_lon': 7.305354, 'max_lon': 7.42499}
montreux_coords = {'min_lat': 46.42124, 'max_lat': 46.486167, 'min_lon': 6.877999, 'max_lon': 6.991982}
baar_coords = {'min_lat': 47.150982, 'max_lat': 47.222963, 'min_lon': 8.496212, 'max_lon': 8.57758}
yberdon_les_bains_coords = {'min_lat': 46.738787, 'max_lat': 46.795344, 'min_lon': 6.604544, 'max_lon': 6.671664}
caurouge_coords = {'min_lat': 46.168491, 'max_lat': 46.190733, 'min_lon': 6.127551, 'max_lon': 6.154525}
wandenswil_coords = {'min_lat': 47.15913, 'max_lat': 47.251714, 'min_lon': 8.615209, 'max_lon': 8.692185}
grenchen_coords = {'min_lat': 47.162365, 'max_lat': 47.240274, 'min_lon': 7.340772, 'max_lon': 7.431409}
zug_coords = {'min_lat': 47.114322, 'max_lat': 47.18979, 'min_lon': 8.475288, 'max_lon': 8.558208}

la_chaux_de_fonds_coords = {'min_lat': 47.066, 'max_lat': 47.119, 'min_lon': 6.769, 'max_lon': 6.866}


#%%
if not os.path.exists('gtfs_temp'):
    os.makedirs('gtfs_temp')
    print(f"Temp directory created at: gtfs_temp")
else:
    # If it exists, print a message and exit
    print(f"The directory gtfs_temp already exists. Please delete it and try again.")
    sys.exit()
#%%
in_path = 'Data/gtfs_fp2024_2024-10-24'
out_path = 'Data/lausanne_coords.zip'
coords = lausanne_coords
#%%
stops = pd.read_csv(in_path + "/stops.txt")
filtered_stops = stops [
    (stops['stop_lat'] >= coords['min_lat']) &
    (stops['stop_lat'] <= coords['max_lat']) &
    (stops['stop_lon'] >= coords['min_lon']) &
    (stops['stop_lon'] <= coords['max_lon'])
]
filtered_stops_ids = set(filtered_stops['stop_id'])
#%%
stop_times = pd.read_csv(in_path + '/stop_times.txt')
filtered_stop_times = stop_times[stop_times['stop_id'].isin(filtered_stops_ids)]
filtered_trip_ids = set(filtered_stop_times['trip_id'])
#%%
trips = pd.read_csv(in_path + '/trips.txt')
filtered_trips = trips[trips['trip_id'].isin(filtered_trip_ids)]
filtered_route_ids = set(filtered_trips['route_id'])
filtered_service_ids = set(filtered_trips['service_id'])
#%%
routes = pd.read_csv(in_path + '/routes.txt')
filtered_routes = routes[routes['route_id'].isin(filtered_route_ids)]
filtered_agency_ids = set(filtered_routes['agency_id'])
#%%
agency = pd.read_csv(in_path + '/agency.txt')
filtered_agency = agency[agency['agency_id'].isin(filtered_agency_ids)]
#%%
calendar = pd.read_csv(in_path + '/calendar.txt')
filtered_calendar = calendar[calendar['service_id'].isin(filtered_service_ids)]
#%%
calendar_dates = pd.read_csv(in_path + '/calendar_dates.txt')
filtered_calendar_dates = calendar_dates[calendar_dates['service_id'].isin(filtered_service_ids)]
#%%
with zipfile.ZipFile(out_path, 'w') as zip_out:
    filtered_agency.to_csv("gtfs_temp/agency.txt", index=False)
    zip_out.write("gtfs_temp/agency.txt", "agency.txt")

    filtered_routes.to_csv("gtfs_temp/routes.txt", index=False)
    zip_out.write("gtfs_temp/routes.txt", "routes.txt")

    filtered_trips.to_csv("gtfs_temp/trips.txt", index=False)
    zip_out.write("gtfs_temp/trips.txt", "trips.txt")

    filtered_stop_times.to_csv("gtfs_temp/stop_times.txt", index=False)
    zip_out.write("gtfs_temp/stop_times.txt", "stop_times.txt")

    filtered_stops.to_csv("gtfs_temp/stops.txt", index=False)
    zip_out.write("gtfs_temp/stops.txt", "stops.txt")

    filtered_calendar.to_csv("gtfs_temp/calendar.txt", index=False)
    zip_out.write("gtfs_temp/calendar.txt", "calendar.txt")

    filtered_calendar_dates.to_csv("gtfs_temp/calendar_dates.txt", index=False)
    zip_out.write("gtfs_temp/calendar_dates.txt", "calendar_dates.txt")
#%%
for file in os.listdir("gtfs_temp"):
        os.remove(f"gtfs_temp/{file}")
os.rmdir("gtfs_temp")
#%%
