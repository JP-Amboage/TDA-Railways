zurich_points = [
    [47.380063, 8.448019],
    [47.434638, 8.501122],
    [47.355280, 8.624798],
    [47.320348, 8.502899]
]

geneva_points = [
    [46.205755, 6.110178],
    [46.196954, 6.112022],
    [46.231921, 6.133057],
    [46.205909, 6.175809],
    [46.231921, 6.133057],
    [46.177942, 6.156548]
]

basel_points = [
    [47.589862, 7.589043],
    [47.564384, 7.554643],
    [47.544364, 7.555874],
    [47.519295, 7.594780],
    [47.565578, 7.634138]
]

bern_points = [
    [46.923358, 7.294518],
    [46.919078, 7.327599],
    [46.939316, 7.495514],
    [46.990122, 7.451039],
]

lausanne_points = [
    [46.520746, 6.584090],
    [46.504390, 6.626893],
    [46.551218, 6.619602],
    [46.538022, 6.674260]
]

def points2bbox(points):
    min_lat = min([point[0] for point in points])
    max_lat = max([point[0] for point in points])
    min_lon = min([point[1] for point in points])
    max_lon = max([point[1] for point in points])
    return {
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon
    }

if __name__ == '__main__':
    cities = {
        'zurich': zurich_points,
        'geneva': geneva_points,
        'basel': basel_points,
        'bern': bern_points,
        'lausanne': lausanne_points
    }
    for city in cities:
        print(f'{city}_coords = {points2bbox(cities[city])}')