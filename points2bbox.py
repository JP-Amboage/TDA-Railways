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

fribourg_points = [
    [46.784883, 7.159640],
    [46.799163, 7.136037],
    [46.812205, 7.138783],
    [46.821603, 7.159984],
    [46.800044, 7.184445]
]

winterthur_points = [
    [47.449790, 8.709568],
    [47.508728, 8.656353],
    [47.548369, 8.754543],
    [47.475671, 8.810333],
]

neuchatel_points = [
    [46.977977, 6.895110],
    [46.989103, 6.845157],
    [47.030656, 6.910216],
    [47.006547, 6.969611],
]

schauffhausen_points = [
    [47.692179, 8.625764],
    [47.737486, 8.540467],
    [47.752821, 8.581050],
    [47.721905, 8.703519],
    [47.685409, 8.662935]
]

sion_points = [
    [46.178770, 7.367885],
    [46.218296, 7.305354],
    [46.256007, 7.411037],
    [46.238674, 7.424990],
]

montreux_points = [
    [46.421240, 6.923489],
    [46.445375, 6.877999],
    [46.486167, 6.962971],
    [46.449042, 6.991982],
]

baar_points = [
    [47.150982, 8.559040],
    [47.200224, 8.496212],
    [47.222963, 8.554062],
    [47.214918, 8.577580],
    [47.151099, 8.559899]
]

yberdon_les_bains_points = [
    [46.764898, 6.604544],
    [46.795344, 6.635958],
    [46.773481, 6.671664],
    [46.738787, 6.634070],
]

caurouge_points = [
    [46.190733, 6.134908],
    [46.187423, 6.127551],
    [46.168491, 6.139812],
    [46.184622, 6.154525],
]

wandenswil_points = [
    [47.251714, 8.639952],
    [47.199282, 8.615209],
    [47.159130, 8.674935],
    [47.163418, 8.692185],
    [47.220425, 8.690190]
]

grenchen_points = [
    [47.217660, 7.340772],
    [47.240274, 7.411497],
    [47.176836, 7.431409],
    [47.162365, 7.411153],
    [47.162365, 7.381628],
]

zug_points = [ 
    [47.181098, 8.475288],
    [47.189790, 8.500040],
    [47.150152, 8.558208],
    [47.114322, 8.500315],
]

rapperswil_jona_points = [
    [47.214674, 8.843328],
    [47.239038, 8.795778],
    [47.250691, 8.851739],
    [47.228431, 8.918000]
]

dubendorf_points = [
    [47.366770, 8.617796],
    [47.388275, 8.583292],
    [47.409190, 8.629812],
    [47.395131, 8.652471]
]

koniz_points = [
    [46.910, 7.360],
    [46.960, 7.360],
    [46.960, 7.460],
    [46.910, 7.460],
]

lancy_points = [
    [46.167312, 6.140129],
    [46.197085, 6.103050],
    [46.202788, 6.110603],
    [46.181933, 6.131031]
]

uster_points = [
    [47.320264, 8.710372],
    [47.352139, 8.689773],
    [47.383763, 8.701789],
    [47.341039, 8.760313]
]

vernier_points = [
    [46.192085, 6.093860],
    [46.216796, 6.068797],
    [46.224932, 6.111798],
    [46.216262, 6.118665]
]

chur_points = [
    [46.818227, 9.501045],
    [46.885680, 9.454128],
    [46.904382, 9.516329],
    [46.852633, 9.594879], 
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
        'lausanne': lausanne_points,
        'fribourg': fribourg_points,
        'winterthur': winterthur_points,
        'neuchatel': neuchatel_points,
        'schauffhausen': schauffhausen_points,
        'sion': sion_points,
        'montreux': montreux_points,
        'baar': baar_points,
        'yberdon_les_bains': yberdon_les_bains_points,
        'caurouge': caurouge_points,
        'wandenswil': wandenswil_points,
        'grenchen': grenchen_points,
        'zug': zug_points,
        'duendorf': dubendorf_points,
        'rapperswil_jona': rapperswil_jona_points,
        'koniz': koniz_points,
        'lancy': lancy_points,
        'uster': uster_points,
        'vernier': vernier_points,
        'chur': chur_points
    }
    for city in cities:
        print(f'{city}_coords = {points2bbox(cities[city])}')