import pandas as pd
import os
import zipfile
import sys

def write_to_zip(zip_dir, filtered_data, keep_txt=True):

    #remove the .zip extension from zip_dir and create a folder with the same name if keep_txt is true
    unpacked_folder = zip_dir[:-4]

    if not os.path.exists(unpacked_folder):
        os.makedirs(unpacked_folder)
        print(f"Directory created at: {unpacked_folder}")
    # else:
    #     #     # If it exists, print a message and exit
    #     #     print(f"The directory {unpacked_folder} already exists. Please delete it and try again.")
    #     #     sys.exit()

    with zipfile.ZipFile(zip_dir, 'w') as zip_out:
        for name, filtered_file in filtered_data.items():
            filtered_file.to_csv(f"{unpacked_folder}/{name}.txt", index=False)
            zip_out.write(f"{unpacked_folder}/{name}.txt", f"{name}.txt")


    if keep_txt == False:
        for file in os.listdir(unpacked_folder):
            os.remove(f"{unpacked_folder}/{file}")
        os.rmdir(unpacked_folder)

def filter_by_coords(GTFS_in, coords, city='cityname', save_dir='Data/cities', keep_txt=True):

    zip_dir = save_dir + f"/{city}_filtered_coords.zip"
    #create empty dict to save the filtered data
    filtered_data = {}

    stops = pd.read_csv(GTFS_in + "/stops.txt")
    filtered_stops = stops [
        (stops['stop_lat'] >= coords['min_lat']) &
        (stops['stop_lat'] <= coords['max_lat']) &
        (stops['stop_lon'] >= coords['min_lon']) &
        (stops['stop_lon'] <= coords['max_lon'])
    ]
    filtered_stops_ids = set(filtered_stops['stop_id'])
    filtered_data['stops'] = filtered_stops

    stop_times = pd.read_csv(GTFS_in + '/stop_times.txt')
    filtered_stop_times = stop_times[stop_times['stop_id'].isin(filtered_stops_ids)]
    filtered_trip_ids = set(filtered_stop_times['trip_id'])
    filtered_data['stop_times'] = filtered_stop_times

    trips = pd.read_csv(GTFS_in + '/trips.txt')
    filtered_trips = trips[trips['trip_id'].isin(filtered_trip_ids)]
    filtered_route_ids = set(filtered_trips['route_id'])
    filtered_service_ids = set(filtered_trips['service_id'])
    filtered_data['trips'] = filtered_trips

    routes = pd.read_csv(GTFS_in + '/routes.txt')
    filtered_routes = routes[routes['route_id'].isin(filtered_route_ids)]
    filtered_agency_ids = set(filtered_routes['agency_id'])
    filtered_data['routes'] = filtered_routes

    agency = pd.read_csv(GTFS_in + '/agency.txt')
    filtered_agency = agency[agency['agency_id'].isin(filtered_agency_ids)]
    filtered_data['agency'] = filtered_agency

    calendar = pd.read_csv(GTFS_in + '/calendar.txt')
    filtered_calendar = calendar[calendar['service_id'].isin(filtered_service_ids)]
    filtered_data['calendar'] = filtered_calendar

    calendar_dates = pd.read_csv(GTFS_in + '/calendar_dates.txt')
    filtered_calendar_dates = calendar_dates[calendar_dates['service_id'].isin(filtered_service_ids)]
    filtered_data['calendar_dates'] = filtered_calendar_dates

    write_to_zip(zip_dir, filtered_data, keep_txt)

def filter_by_time(GTFS_in, start_time = '07:00:00', end_time = '10:00:00', save_dir = "Data/by_day/20241111", keep_txt=True):

    str_time_interval = start_time.replace(':', '')[0:4] \
                           + 'to' \
                           + end_time.replace(':', '')[0:4]
    zip_dir = save_dir + f"/{str_time_interval}.zip"

    #create empty dict to save the filtered data
    filtered_data = {}

    stop_times = pd.read_csv(GTFS_in + '/stop_times.txt')
    filtered_stop_times = stop_times[stop_times['departure_time'].between(start_time, end_time)]
    filtered_stop_ids = set(filtered_stop_times['stop_id'])
    filtered_trip_ids = set(filtered_stop_times['trip_id'])
    filtered_data['stop_times'] = filtered_stop_times

    stops = pd.read_csv(GTFS_in + "/stops.txt")
    filtered_stops = stops[stops['stop_id'].isin(filtered_stop_ids)]
    filtered_data['stops'] = filtered_stops

    trips = pd.read_csv(GTFS_in + '/trips.txt')
    filtered_trips = trips[trips['trip_id'].isin(filtered_trip_ids)]
    filtered_route_ids = set(filtered_trips['route_id'])
    filtered_service_ids = set(filtered_trips['service_id'])
    filtered_data['trips'] = filtered_trips

    routes = pd.read_csv(GTFS_in + '/routes.txt')
    filtered_routes = routes[routes['route_id'].isin(filtered_route_ids)]
    filtered_agency_ids = set(filtered_routes['agency_id'])
    filtered_data['routes'] = filtered_routes

    agency = pd.read_csv(GTFS_in + '/agency.txt')
    filtered_agency = agency[agency['agency_id'].isin(filtered_agency_ids)]
    filtered_data['agency'] = filtered_agency

    calendar = pd.read_csv(GTFS_in + '/calendar.txt')
    filtered_calendar = calendar[calendar['service_id'].isin(filtered_service_ids)]
    filtered_data['calendar'] = filtered_calendar

    calendar_dates = pd.read_csv(GTFS_in + '/calendar_dates.txt')
    filtered_calendar_dates = calendar_dates[calendar_dates['service_id'].isin(filtered_service_ids)]
    filtered_data['calendar_dates'] = filtered_calendar_dates

    write_to_zip(zip_dir, filtered_data, keep_txt)

def filter_by_day(GTFS_in, day = 20241111, save_dir = "Data/by_day", keep_txt=True):

    days = [day]
    zip_dir = save_dir + f"/{day}.zip"
    #create empty dict to save the filtered data
    filtered_data = {}

    calendar_dates = pd.read_csv(GTFS_in + '/calendar_dates.txt')
    filtered_calendar_dates = calendar_dates[calendar_dates['date'].isin(days)]
    filtered_service_ids = set(filtered_calendar_dates['service_id'])
    filtered_data['calendar_dates'] = filtered_calendar_dates

    trips = pd.read_csv(GTFS_in + '/trips.txt')
    filtered_trips = trips[trips['service_id'].isin(filtered_service_ids)]
    filtered_route_ids = set(filtered_trips['route_id'])
    filtered_trip_ids = set(filtered_trips['trip_id'])
    filtered_data['trips'] = filtered_trips

    routes = pd.read_csv(GTFS_in + '/routes.txt')
    filtered_routes = routes[routes['route_id'].isin(filtered_route_ids)]
    filtered_agency_ids = set(filtered_routes['agency_id'])
    filtered_data['routes'] = filtered_routes

    agency = pd.read_csv(GTFS_in + '/agency.txt')
    filtered_agency = agency[agency['agency_id'].isin(filtered_agency_ids)]
    filtered_data['agency'] = filtered_agency

    calendar = pd.read_csv(GTFS_in + '/calendar.txt')
    filtered_calendar = calendar[calendar['service_id'].isin(filtered_service_ids)]
    filtered_data['calendar'] = filtered_calendar

    stop_times = pd.read_csv(GTFS_in + '/stop_times.txt')
    filtered_stop_times = stop_times[stop_times['trip_id'].isin(filtered_trip_ids)]
    filtered_stop_ids = set(filtered_stop_times['stop_id'])
    filtered_data['stop_times'] = filtered_stop_times

    stops = pd.read_csv(GTFS_in + '/stops.txt')
    filtered_stops = stops[stops['stop_id'].isin(filtered_stop_ids)]
    filtered_data['stops'] = filtered_stops

    write_to_zip(zip_dir, filtered_data, keep_txt)
