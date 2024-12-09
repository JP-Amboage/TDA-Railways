import pandas as pd
import os
import peartree as pt
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import persim


from utils import compute_persistence_statistics
from utils import compute_ideal_matrix
from utils import add_earth_distnace
from utils import compute_speed
from utils import compute_ideal_graph
from utils import compute_undirected_graph_no_multi
from utils import stats2features
from utils import features2str

BERN_MAX_SPEED =  74.35452432368538

def main(gtfs_path: str, out_path: str, city: str):
    features = [['city', city]]
    ################################################################################
    ############ Load the GTFS data and do initial processing/filtering ############
    ################################################################################
    print(f"Reading GTFS feed from {gtfs_path}")
    feed = pt.get_representative_feed(gtfs_path)

    print(f"Loading GTFS as graph")
    start = 00*60*60  # 7:00 AM
    end = 24*60*60
    G = pt.load_feed_as_graph(feed, start, end)

    print(f"Saving graph plot to {out_path}")
    graph_plot = pt.generate_plot(G)
    #graph_plot.show()
    graph_plot[0].savefig(out_path+'/regular_graph.png')
    
    print(f'Graph has {len(G.nodes)} nodes and {len(G.edges)} edges')
    print(f'Removing islated nodes')
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f'Pruned graph has {len(G.nodes)} nodes and {len(G.edges)} edges')
    
    #making the graph undirected
    print(f"Making the graph undirected")
    G = compute_undirected_graph_no_multi(G)
    print(f'New graph has {len(G.nodes)} nodes and {len(G.edges)} edges')

    print(f"Adding earth distance as edge attribute")
    add_earth_distnace(G)
    print(f'Edges attributes: {list(G.edges(data=True))[0]}')
    print(f'Node attributes: {list(G.nodes(data=True))[0]}')
    ################################################################################
    ################################################################################

    ################################################################################
    ############## Compute Homology in the Unnormalized Space Domain ###############
    ################################################################################
    print(f"Computing Floyd-Warshall distance matrix")
    dist_mat = nx.floyd_warshall_numpy(G, weight='earth_distance')
    print(f"Distance matrix shape: {dist_mat.shape}")
    has_infinity = np.isinf(dist_mat).any()
    print(f"Distance matrix has infinity? {has_infinity}")
    print(f"Distance matrix is symmetric? {np.array_equal(dist_mat, dist_mat.T)}")
    print(f"Comuting persistence on Floyd-Warshall distance matrix")
    result = ripser(dist_mat, distance_matrix=True, maxdim=1)
    
    print(f"Saving persistence diagrams to {out_path}")
    persistence_diagram = result['dgms']
    f, ax = plt.subplots()
    plot_diagrams(persistence_diagram, show=False)
    f.savefig(out_path+'/persistence_diagram.png')

    print(f'Computing persistence statistics')
    stats = compute_persistence_statistics(persistence_diagram)
    print(stats)
    features += stats2features(stats, 'space')
    ################################################################################
    ################################################################################

    ################################################################################
    ######## Compare with the Ideal Graph in the Unnormalized Space Domain #########
    ################################################################################
    print(f"Computing Ideal distance matrix")
    ideal_dist_mat = compute_ideal_matrix(G)

    print(f"Comuting persistence on Ideal distance matrix")
    ideal_result = ripser(ideal_dist_mat , distance_matrix=True)
    
    print(f"Saving persistence diagrams to {out_path}")
    ideal_persistence_diagram = ideal_result['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_persistence_diagram, show=False)
    f.savefig(out_path+'/persistence_diagram_ideal.png')

    print(f'Computing persistence statistics on the ideal distance matrix')
    ideal_stats = compute_persistence_statistics(ideal_persistence_diagram)
    print(ideal_stats)

    print(f"Computing bottleneck distance between the persistence diagrams")
    H1_bottleneck_distance = persim.bottleneck(persistence_diagram[1], ideal_persistence_diagram[1])
    print(f"Bottleneck distance H1: {H1_bottleneck_distance}")
    H0_bottleneck_distance = persim.bottleneck(persistence_diagram[0], ideal_persistence_diagram[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance}")

    print(f"Checking if ideal matrix is smaller than the original matrix: {np.all(ideal_dist_mat <= dist_mat)}")
    # Not needed as we are using the max speed of Bern for every graph
    # print(f"Speed: {compute_speed(G)}")

    features += [['H1_bn_space_ideal',H1_bottleneck_distance]]
    features += [['H0_bn_space_ideal',H0_bottleneck_distance]]
    # features += stats2features(ideal_stats, 'space_ideal')
    ################################################################################
    ################################################################################

    ################################################################################
    ###### Compare with the Min. Span. Tree in the Unnormalized Space Domain #######
    ################################################################################
    print(f"Computing ideal graph")
    G_ideal = compute_ideal_graph(G, BERN_MAX_SPEED)

    print(f"Computing minimum spanning tree on ideal graph")
    G_ideal_mst = nx.minimum_spanning_tree(G_ideal)

    print(f"Computing distance matrix for the spanning tree on ideal graph")
    dist_mat_mst = nx.floyd_warshall_numpy(G_ideal_mst, weight='earth_distance')

    print(f"Comuting persistence on Min Spanning Tree distance matrix")
    ideal_mst_result= ripser(dist_mat_mst, distance_matrix=True, maxdim=0)
    
    print(f"Saving persistence diagrams to {out_path}")
    ideal_mst_persistence_diagram = ideal_mst_result['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_mst_persistence_diagram, show=False)
    f.savefig(out_path+'/persistence_diagram_mst.png')

    print(f'Computing persistence statistics on the min spanning tree')
    ideal_mst_stats = compute_persistence_statistics(ideal_mst_persistence_diagram)
    print(ideal_mst_stats)

    print(f"Computing bottleneck distance between real and mst diagrams")
    H0_bottleneck_distance_mst = persim.bottleneck(persistence_diagram[0], ideal_mst_persistence_diagram[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance_mst}")
    # H1_bottleneck_distance_mst = persim.bottleneck(persistence_diagram[1], ideal_mst_persistence_diagram[1])
    # print(f"Bottleneck distance H1: {H1_bottleneck_distance_mst}")

    features += [['H0_bn_space_mst',H0_bottleneck_distance_mst]]
    # features += [['H1_bn_space_mst',H1_bottleneck_distance_mst]]
    # features += stats2features(ideal_mst_stats, 'space_mst')
    ################################################################################
    ################################################################################

    ################################################################################
    ############## Compute Homology in the Unnormalized Time Domain ################
    ################################################################################
    print()
    print(f"Moving into the time domain")
    print(f"Computing floyd warshall time matrix on the original graph")
    time_dist_mat = nx.floyd_warshall_numpy(G, weight='length')
    print(f"Comuting persistence on Floyd-Warshall time matrix")
    result_time = ripser(time_dist_mat, distance_matrix=True, maxdim=1)

    print(f"Saving persistence diagrams to {out_path}")
    persistence_diagram_time = result_time['dgms']
    f, ax = plt.subplots()
    plot_diagrams(persistence_diagram_time, show=False)
    f.savefig(out_path+'/persistence_diagram_time.png')

    print(f'Computing persistence statistics')
    stats_time = compute_persistence_statistics(persistence_diagram_time)
    print(stats_time)

    features += stats2features(stats_time, 'time')
    ################################################################################
    ################################################################################
    
    ################################################################################
    ######## Compare with the Ideal Graph in the Unnormalized Time Domain ##########
    ################################################################################
    print(f"Comuting Ideal time matrix")
    ideal_time_mat = ideal_dist_mat / BERN_MAX_SPEED

    print(f"Comuting persistence on Ideal distance matrix")
    ideal_result_time = ripser(ideal_time_mat , distance_matrix=True)
    
    print(f"Saving persistence diagrams to {out_path}")
    ideal_persistence_diagram_time = ideal_result_time['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_persistence_diagram_time, show=False)
    f.savefig(out_path+'/persistence_diagram_ideal_time.png')

    print(f'Computing persistence statistics on the ideal time matrix')
    ideal_stats_time = compute_persistence_statistics(ideal_persistence_diagram_time)
    print(ideal_stats_time)

    print(f"Computing bottleneck distance between the time persistence diagrams")
    H1_bottleneck_distance_time = persim.bottleneck(persistence_diagram_time[1], ideal_persistence_diagram_time[1])
    print(f"Bottleneck distance H1: {H1_bottleneck_distance_time}")
    H0_bottleneck_distance_time = persim.bottleneck(persistence_diagram_time[0], ideal_persistence_diagram_time[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance_time}")

    features += [['H1_bn_time_ideal',H1_bottleneck_distance_time]]
    features += [['H0_bn_time_ideal',H0_bottleneck_distance_time]]
    # features += stats2features(ideal_stats_time, 'time_ideal')
    ################################################################################
    ################################################################################

    ################################################################################
    ###### Compare with the Min. Span. Tree in the Unnormalized Time Domain ########
    ################################################################################
    print(f"Computing time matrix for the spanning tree on ideal graph")
    dist_mat_mst_time = dist_mat_mst / BERN_MAX_SPEED

    print(f"Comuting persistence on Min Spanning Tree distance matrix")
    ideal_mst_result_time= ripser(dist_mat_mst_time, distance_matrix=True, maxdim=0)
    
    print(f"Saving persistence diagrams to {out_path}")
    ideal_mst_persistence_diagram_time = ideal_mst_result_time['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_mst_persistence_diagram_time, show=False)
    f.savefig(out_path+'/persistence_diagram_mst_time.png')

    print(f'Computing persistence statistics on the min spanning tree')
    ideal_mst_stats_time = compute_persistence_statistics(ideal_mst_persistence_diagram_time)
    print(ideal_mst_stats_time)

    print(f"Computing bottleneck distance between real and mst diagrams")
    H0_bottleneck_distance_mst_time = persim.bottleneck(persistence_diagram_time[0], ideal_mst_persistence_diagram_time[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance_mst_time}")
    # H1_bottleneck_distance_mst_time = persim.bottleneck(persistence_diagram_time[1], ideal_mst_persistence_diagram_time[1])
    # print(f"Bottleneck distance H1: {H1_bottleneck_distance_mst_time}")

    features += [['H0_bn_time_mst',H0_bottleneck_distance_mst_time]]
    # features += [['H1_bn_time_mst',H1_bottleneck_distance_mst_time]]
    # features += stats2features(ideal_mst_stats_time, 'time_mst')
    ################################################################################
    ################################################################################

    ################################################################################
    ############### Compute Homology in the Normalized Space Domain ################
    ################################################################################
    print()
    print(f"Moving into the normalized distance domain")
    normalizer = np.mean(ideal_dist_mat)
    print(f"Normalizer: {normalizer}")
    print(f"Median: {np.median(ideal_dist_mat)}")
    dist_mat_normalized = dist_mat / normalizer

    print(f"Comuting persistence on NORMALIZED distance matrix")
    result_normalized = ripser(dist_mat_normalized, distance_matrix=True, maxdim=1)
    print(f"Saving persistence diagrams to {out_path}")
    persistence_diagram_normalized = result_normalized['dgms']
    f, ax = plt.subplots()
    plot_diagrams(persistence_diagram_normalized, show=False)
    f.savefig(out_path+'/persistence_diagram_normalized.png')

    print(f'Computing persistence statistics')
    stats_normalized = compute_persistence_statistics(persistence_diagram_normalized)
    print(stats_normalized)

    features += stats2features(stats_normalized, 'space_normalized')
    ################################################################################
    ################################################################################

    ################################################################################
    ######### Compare with the Ideal Graph in the Normalized Space Domain ##########
    ################################################################################
    ideal_dist_mat_normalized = ideal_dist_mat / normalizer
    print(f"Comuting persistence on Ideal NORMALIZED distance matrix")
    ideal_result_normalized = ripser(ideal_dist_mat_normalized , distance_matrix=True)
    print(f"Saving persistence diagrams to {out_path}")
    ideal_persistence_diagram_normalized = ideal_result_normalized['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_persistence_diagram_normalized, show=False)
    f.savefig(out_path+'/persistence_diagram_ideal_normalized.png')

    print(f'Computing persistence statistics')
    ideal_stats_normalized = compute_persistence_statistics(ideal_persistence_diagram_normalized)
    print(ideal_stats_normalized)

    print(f"Computing bottleneck distance between the normalized persistence diagrams")
    H1_bottleneck_distance_normalized = persim.bottleneck(persistence_diagram_normalized[1], ideal_persistence_diagram_normalized[1])
    print(f"Bottleneck distance H1: {H1_bottleneck_distance_normalized}")
    H0_bottleneck_distance_normalized = persim.bottleneck(persistence_diagram_normalized[0], ideal_persistence_diagram_normalized[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance_normalized}")

    features += [['H1_bn_space_ideal_normalized',H1_bottleneck_distance_normalized]]
    features += [['H0_bn_space_ideal_normalized',H0_bottleneck_distance_normalized]]
    # features += stats2features(ideal_stats_normalized, 'space_ideal_normalized')
    ################################################################################
    ################################################################################

    ################################################################################
    ####### Compare with the Min. Span. Tree in the Normalized Space Domain ########
    ################################################################################
    print(f"Computing dist matrix for the spanning tree normalized")
    dist_mat_mst_normalized = dist_mat_mst / normalizer

    print(f"Comuting persistence on Min Spanning Tree distance matrix")
    ideal_mst_result_normalized = ripser(dist_mat_mst_normalized, distance_matrix=True, maxdim=0)
    
    print(f"Saving persistence diagrams to {out_path}")
    ideal_mst_persistence_diagram_normalized = ideal_mst_result_normalized['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_mst_persistence_diagram_normalized, show=False)
    f.savefig(out_path+'/persistence_diagram_mst_normalized.png')

    print(f'Computing persistence statistics on the min spanning tree')
    ideal_mst_stats_normalized = compute_persistence_statistics(ideal_mst_persistence_diagram_normalized)
    print(ideal_mst_stats_normalized)

    print(f"Computing bottleneck distance between real and mst diagrams")
    H0_bottleneck_distance_mst_normalized = persim.bottleneck(persistence_diagram_normalized[0], ideal_mst_persistence_diagram_normalized[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance_mst_normalized}")
    # H1_bottleneck_distance_mst_normalized = persim.bottleneck(persistence_diagram_normalized[1], ideal_mst_persistence_diagram_normalized[1])
    # print(f"Bottleneck distance H1: {H1_bottleneck_distance_mst_normalized}")

    features += [['H0_bn_space_mst_normalized',H0_bottleneck_distance_mst_normalized]]
    # features += [['H1_bn_space_mst_normalized',H1_bottleneck_distance_mst_normalized]]
    # features += stats2features(ideal_mst_stats_normalized, 'space_mst_normalized')
    ################################################################################
    ################################################################################

    ################################################################################
    ############### Compute Homology in the Normalized Time Domain #################
    ################################################################################
    print()
    print(f"Moving into the normalized time domain")
    normalizer_time = np.mean(ideal_time_mat)
    print(f"Normalizer Time: {normalizer_time}")
    print(f"Median: {np.median(ideal_time_mat)}")
    time_mat_normalized = time_dist_mat / normalizer_time

    print(f"Comuting persistence on NORMALIZED distance matrix")
    result_normalized_time = ripser(time_mat_normalized, distance_matrix=True, maxdim=1)
    print(f"Saving persistence diagrams to {out_path}")
    persistence_diagram_normalized_time = result_normalized_time['dgms']
    f, ax = plt.subplots()
    plot_diagrams(persistence_diagram_normalized_time, show=False)
    f.savefig(out_path+'/persistence_diagram_normalized_time.png')

    print(f'Computing persistence statistics')
    stats_normalized_time = compute_persistence_statistics(persistence_diagram_normalized_time)
    print(stats_normalized_time)

    features += stats2features(stats_normalized_time, 'time_normalized')
    ################################################################################
    ################################################################################
    
    ################################################################################
    ######### Compare with the Ideal Graph in the Normalized Time Domain ###########
    ################################################################################
    ideal_time_mat_normalized = ideal_time_mat / normalizer_time
    print(f"Comuting persistence on Ideal NORMALIZED time matrix")
    ideal_result_normalized_time = ripser(ideal_time_mat_normalized , distance_matrix=True)
    print(f"Saving persistence diagrams to {out_path}")
    ideal_persistence_diagram_normalized_time = ideal_result_normalized_time['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_persistence_diagram_normalized_time, show=False)
    f.savefig(out_path+'/persistence_diagram_ideal_normalized_time.png')

    print(f'Computing persistence statistics')
    ideal_stats_normalized_time = compute_persistence_statistics(ideal_persistence_diagram_normalized_time)
    print(ideal_stats_normalized_time)

    print(f"Computing bottleneck distance between the normalized persistence diagrams")
    H1_bottleneck_distance_normalized_time = persim.bottleneck(persistence_diagram_normalized_time[1], ideal_persistence_diagram_normalized_time[1])
    print(f"Bottleneck distance H1: {H1_bottleneck_distance_normalized_time}")
    H0_bottleneck_distance_normalized_time = persim.bottleneck(persistence_diagram_normalized_time[0], ideal_persistence_diagram_normalized_time[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance_normalized_time}")

    features += [['H1_bn_time_ideal_normalized',H1_bottleneck_distance_normalized_time]]
    features += [['H0_bn_time_ideal_normalized',H0_bottleneck_distance_normalized_time]]
    # features += stats2features(ideal_stats_normalized_time, 'time_ideal_normalized')
    ################################################################################
    ################################################################################

    ################################################################################
    ####### Compare with the Min. Span. Tree in the Normalized Time Domain #########
    ################################################################################
    print(f"Computing time matrix for the spanning tree normalized")
    dist_mat_mst_time_normalized = dist_mat_mst_time / normalizer_time

    print(f"Comuting persistence on Min Spanning Tree distance matrix")
    ideal_mst_result_time_normalized = ripser(dist_mat_mst_time_normalized, distance_matrix=True, maxdim=0)
    
    print(f"Saving persistence diagrams to {out_path}")
    ideal_mst_persistence_diagram_time_normalized = ideal_mst_result_time_normalized['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_mst_persistence_diagram_time_normalized, show=False)
    f.savefig(out_path+'/persistence_diagram_mst_normalized.png')

    print(f'Computing persistence statistics on the min spanning tree')
    ideal_mst_stats_time_normalized = compute_persistence_statistics(ideal_mst_persistence_diagram_time_normalized)
    print(ideal_mst_stats_time_normalized)

    print(f"Computing bottleneck distance between real and mst diagrams")
    H0_bottleneck_distance_mst_time_normalized = persim.bottleneck(persistence_diagram_normalized_time[0], ideal_mst_persistence_diagram_time_normalized[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance_mst_time_normalized}")
    # H1_bottleneck_distance_mst_time_normalized = persim.bottleneck(persistence_diagram_normalized_time[1], ideal_mst_persistence_diagram_time_normalized[1])
    # print(f"Bottleneck distance H1: {H1_bottleneck_distance_mst_time_normalized}")

    features += [['H0_bn_time_mst_normalized',H0_bottleneck_distance_mst_time_normalized]]
    # features += [['H1_bn_time_mst_normalized',H1_bottleneck_distance_mst_time_normalized]]
    # features += stats2features(ideal_mst_stats_normalized, 'space_mst_normalized')
    ################################################################################
    ################################################################################

    with open(out_path+f'/{city}_features.csv', 'w') as file:
        file.write(features2str(features))

if __name__ == '__main__':

    city = 'Lausanne'
    gtfs_path = 'Data/filtered/lausanne_coords.zip'
    out_path = 'Data/outputs/test'

    main(gtfs_path, out_path, city)