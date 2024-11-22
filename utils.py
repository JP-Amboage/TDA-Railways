import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset, download_url
import numpy as np
from ripser import ripser
from persim import PersImage, plot_diagrams
import networkx as nx
from math import radians, sin, cos, sqrt, atan2

def nx_to_pyg_data(graph):
    """
    Convert a networkx graph to a PyTorch Geometric Data object.
    """
    # Create a mapping from nodes to integers
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    # Get the adjacency information with remapped node indices
    edge_index = torch.tensor([(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]).t().contiguous()

    # Extract node features with remapped node indices
    x_list = []
    for node in node_mapping:
        node_data = graph.nodes[node]
        boarding_cost = node_data['boarding_cost']
        modes = len(node_data['modes'])  # Transforming 'modes' list to its length
        y = node_data['y']
        x_coord = node_data['x']
        x_list.append([boarding_cost, modes, y, x_coord])

    x = torch.tensor(x_list, dtype=torch.float)

    edge_features = []

    # Extract edge features
    edge_features = []
    for u, v, key in graph.edges(keys=True):
        edge_data = graph[u][v][key]
        length = float(edge_data.get('length', 0))  # convert to float, default to 0 if 'length' is not found
        mode = int(edge_data.get('mode', '0') == 'transit')  # convert 'mode' to binary representation (1 if 'transit', 0 otherwise)
        edge_features.append([length, mode])

    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Return as PyTorch Geometric Data
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, nx_graphs, transform=None, pre_transform=None, pre_filter=None):
        self.nx_graphs = nx_graphs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # We're directly passing in NetworkX graphs so we don't need raw file names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No downloading needed as we're directly passing in NetworkX graphs
        pass

    def process(self):
        # Convert the NetworkX graphs to PyTorch Geometric Data objects
        data_list = [nx_to_pyg_data(graph) for graph in self.nx_graphs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Function to compute persistence statistics
def compute_persistence_statistics(diagrams):
    stats = {}
    for dim, diagram in enumerate(diagrams):
        if len(diagram) > 0:
            lifetimes = diagram[:, 1] - diagram[:, 0]  # Death - Birth
            total_persistence = np.sum(lifetimes)
            avg_persistence = np.mean(lifetimes) if len(lifetimes) > 0 else 0
            max_persistence = np.max(lifetimes) if len(lifetimes) > 0 else 0
            
            stats[f"H{dim}"] = {
                "num_features": len(lifetimes),
                "total_persistence": total_persistence,
                "average_persistence": avg_persistence,
                "max_persistence": max_persistence,
            }
        else:
            stats[f"H{dim}"] = {
                "num_features": 0,
                "total_persistence": 0,
                "average_persistence": 0,
                "max_persistence": 0,
            }
    return stats


def print_stats(stats):
    for hom_dim, stat in stats.items():
        print(f"{hom_dim} Persistence Statistics:")
        for key, value in stat.items():
            print(f"  {key}: {value}")
        print()


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth using their latitude and longitude.
    """
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def compute_ideal_matrix(G: nx.Graph):
    """
    Compute the ideal distance matrix for a graph G based on Haversine distances.
    """
    # Get the list of nodes
    nodes = list(G.nodes)
    num_nodes = len(nodes)
    
    # Initialize the distance matrix
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    # Compute distances between all pairs of nodes
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i == j:
                distance_matrix[i][j] = 0  # Distance to itself is zero
            else:
                # Get latitude and longitude of both nodes
                # print(G.nodes[node1].keys())
                lat1, lon1 = G.nodes[node1]['x'], G.nodes[node1]['y']
                lat2, lon2 = G.nodes[node2]['x'], G.nodes[node2]['y']
                
                # Compute Haversine distance
                distance_matrix[i][j] = haversine(lat1, lon1, lat2, lon2)
    
    return distance_matrix

def make_symmetric(matrix, method='min'):
    if method == 'min':
        return np.minimum(matrix, matrix.T)
    elif method == 'max':
        return np.maximum(matrix, matrix.T)
    elif method == 'mean':
        return (matrix + matrix.T) / 2
    else:
        raise ValueError("Method should be either 'min', 'max' or 'mean'")


def normalize(matrix, method='average'):
    if method == 'average':
        return matrix / np.mean(matrix[np.triu_indices(matrix.shape[0])])
    elif method == 'max':
        return matrix / np.max(matrix)
    else:
        raise ValueError("Method should be either 'average' or 'max'")
