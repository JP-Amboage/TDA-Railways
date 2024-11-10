#%%
import pandas as pd
import os
import zipfile
import sys

if not os.path.exists('gtfs_temp'):
    os.makedirs('gtfs_temp')
    print(f"Temp directory created at: gtfs_temp")
else:
    # If it exists, print a message and exit
    print(f"The directory gtfs_temp already exists. Please delete it and try again.")
    sys.exit()
# %%
path = 'Data/gtfs_fp2024_2024-10-24'
# out_path = 'Data/gtfs_fp2024_2024-10-24_filtered.zip'
# agency_ids = ['11']
out_path = 'Data/zurich_filtered.zip'
agency_ids = ['849']
# %%
agency = pd.read_csv(path + '/agency.txt')
filtered_agency = agency[agency['agency_id'].isin(agency_ids)]
filtered_agency_ids = set(filtered_agency['agency_id'])

# %%
routes = pd.read_csv(path + '/routes.txt')
filtered_routes = routes[routes['agency_id'].isin(filtered_agency_ids)]
filtered_route_ids = set(filtered_routes['route_id'])
# %%
trips = pd.read_csv(path + '/trips.txt')
filtered_trips = trips[trips['route_id'].isin(filtered_route_ids)]
filtered_trip_ids = set(filtered_trips['trip_id'])

# %%
stop_times = pd.read_csv(path + '/stop_times.txt')
filtered_stop_times = stop_times[stop_times['trip_id'].isin(filtered_trip_ids)]
filtered_stop_ids = set(filtered_stop_times['stop_id'])
# %%
stops = pd.read_csv(path + '/stops.txt')
filtered_stops = stops[stops['stop_id'].isin(filtered_stop_ids)]
# %%
calendar = pd.read_csv(path + '/calendar.txt')
filtered_service_ids = set(filtered_trips['service_id'])
filtered_calendar = calendar[calendar['service_id'].isin(filtered_service_ids)]
# %%
calendar_dates = pd.read_csv(path + '/calendar_dates.txt')
filtered_calendar_dates = calendar_dates[calendar_dates['service_id'].isin(filtered_service_ids)]
# %%
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

# %%
for file in os.listdir("gtfs_temp"):
        os.remove(f"gtfs_temp/{file}")
os.rmdir("gtfs_temp")
# %%
