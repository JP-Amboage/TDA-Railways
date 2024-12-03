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

def main(gtfs_path: str, out_path: str):
    
    print(f"Reading GTFS feed from {gtfs_path}")
    feed = pt.get_representative_feed(gtfs_path)

    print(f"Loading GTFS as graph")
    start = 00*60*60  # 7:00 AM
    end = 24*60*60
    G = pt.load_feed_as_graph(feed, start, end)
    
    print(f"Adding earth distance as edge attribute")
    add_earth_distnace(G)
    print(f'Edges attributes: {list(G.edges(data=True))[0]}')
    print(f'Node attributes: {list(G.nodes(data=True))[0]}')
    
    print(f"Saving graph plot to {out_path}")
    graph_plot = pt.generate_plot(G)
    #graph_plot.show()
    graph_plot[0].savefig(out_path+'/regular_graph.png')

    #dataset = MyOwnDataset(root=out_path+'/regular_graph', nx_graphs=[G])
    print(f"Computing Floyd-Warshall distance matrix")
    dist_mat = nx.floyd_warshall_numpy(G.to_undirected(), weight='earth_distance')
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

    print(f"Comuting Ideal distance matrix")
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

    print(f"Computing speed")
    speed = compute_speed(G)
    print(f"Speed: {speed}")

    print(f"Computing ideal graph")
    G_ideal = compute_ideal_graph(G, speed)

    print(f"Computing minimum spanning tree on ideal graph")
    G_ideal_mst = nx.minimum_spanning_tree(G_ideal)

    print(f"Computing distance matrix for the spanning tree on ideal graph")
    dist_mat_mst = nx.floyd_warshall_numpy(G_ideal_mst.to_undirected(), weight='earth_distance')

    print(f"Comuting persistence on Min Spanning Tree distance matrix")
    ideal_mst_result= ripser(dist_mat_mst, distance_matrix=True, maxdim=1)
    
    print(f"Saving persistence diagrams to {out_path}")
    ideal_mst_persistence_diagram = ideal_mst_result['dgms']
    f, ax = plt.subplots()
    plot_diagrams(ideal_mst_persistence_diagram, show=False)
    f.savefig(out_path+'/persistence_diagram_mst.png')

    print(f'Computing persistence statistics on the min spanning tree')
    ideal_mst_stats = compute_persistence_statistics(ideal_mst_persistence_diagram)
    print(ideal_mst_stats)

    print(f"Computing bottleneck distance between real and mst diagrams")
    H1_bottleneck_distance = persim.bottleneck(persistence_diagram[1], ideal_mst_persistence_diagram[1])
    print(f"Bottleneck distance H1: {H1_bottleneck_distance}")
    H0_bottleneck_distance = persim.bottleneck(persistence_diagram[0], ideal_mst_persistence_diagram[0])
    print(f"Bottleneck distance H0: {H0_bottleneck_distance}")

if __name__ == '__main__':
    gtfs_path = 'Data/zurich_filtered_coords.zip'
    out_path = 'Data/outputs/zurich'
    main(gtfs_path, out_path)