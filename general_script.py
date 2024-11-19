import pandas as pd
import os
import peartree as pt
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import numpy as np
from ripser import ripser
from persim import plot_diagrams

from utils import nx_to_pyg_data
from utils import MyOwnDataset
from utils import compute_persistence_statistics
from utils import compute_ideal_matrix

import persim

def main(gtfs_path: str, out_path: str):
    
    print(f"Reading GTFS feed from {gtfs_path}")
    feed = pt.get_representative_feed(gtfs_path)

    print(f"Loading GTFS as graph")
    start = 00*60*60  # 7:00 AM
    end = 24*60*60
    G = pt.load_feed_as_graph(feed, start, end)

    print(f"Saving graph to {out_path}")
    graph_plot = pt.generate_plot(G)
    #graph_plot.show()
    graph_plot[0].savefig(out_path+'/regular_graph.png')

    #dataset = MyOwnDataset(root=out_path+'/regular_graph', nx_graphs=[G])
    print(f"Comuting Floyd-Warshall distance matrix")
    dist_mat = nx.floyd_warshall_numpy(G.to_undirected())
    has_infinity = np.isinf(dist_mat).any()
    print(f"Distance matrix has infinity? {has_infinity}")
    print(f"Distance matrix is symmetric? {np.array_equal(dist_mat, dist_mat.T)}")

    print(f"Comuting persistence on Floyd-Warshall distance matrix")
    result = ripser(dist_mat, distance_matrix=True, maxdim=2)
    
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


if __name__ == '__main__':
    gtfs_path = 'Data/lucerne_filtered_coords.zip'
    out_path = 'Data/output_lucerne'
    main(gtfs_path, out_path)