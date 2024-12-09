import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset, download_url
import numpy as np
from ripser import ripser
from persim import PersImage, plot_diagrams
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
from geopy import distance
from networkx import set_edge_attributes
from typing import Optional

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



def _compute_var_statistics(data: list):
    """
    Compute statistics for a list of values.
    """
    if len(data) == 0:
        return {
            "mean": 0,
            "std": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "p10": 0,
            "p25": 0,
            "p75": 0,
            "p90": 0,
            "iqr": 0
        }
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    minimum = np.min(data)
    maximum = np.max(data)
    p10 = np.percentile(data, 10)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    p90 = np.percentile(data, 90)
    iqr = p75 - p25

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": minimum,
        "max": maximum,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "iqr": iqr
    }

# Function to compute persistence statistics
def compute_persistence_statistics(diagrams):
    stats = {}
    for dim, diagram in enumerate(diagrams):
        birth_times = []
        death_times = []
        midpoints = []
        lifespans = []
        num_non_inf = 0
        num_inf = 0
        total_count = 0
        for point in diagram:
            if np.isinf(point[0]) or np.isinf(point[1]):
                num_inf += 1
                continue
            birth_times.append(point[0])
            death_times.append(point[1])
            midpoints.append((point[0] + point[1]) / 2)
            lifespans.append(point[1] - point[0])
            num_non_inf += 1

        total_count = num_non_inf + num_inf
        stats[f"H{dim}"] = {
                "count": total_count,
                "num_non_inf": num_non_inf,
                "num_inf": num_inf,
                "births": _compute_var_statistics(birth_times),
                "deaths": _compute_var_statistics(death_times),
                "midpoints": _compute_var_statistics(midpoints),
                "lifespans": _compute_var_statistics(lifespans)
            }
    return stats


def stats2features(stats, id):
    '''
    Convert persistence statistics to features
    '''
    features = []
    for dim in stats:
        for key in ['count', 'num_non_inf', 'num_inf']:
            feature = [f'{id}_{dim}_{key}', stats[dim][key]]
            features.append(feature)

        for key in ['births', 'deaths', 'midpoints', 'lifespans']:
            for stat in stats[dim][key]:
                feature = [f'{id}_{dim}_{key}_{stat}', stats[dim][key][stat]]
                features.append(feature)
    return features

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
                coords_1 = (G.nodes[node1]['y'], G.nodes[node1]['x'])
                coords_2 = (G.nodes[node2]['y'], G.nodes[node2]['x'])
                # Compute Haversine distance
                distance_matrix[i][j] = distance.geodesic(coords_1, coords_2).meters
    
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

def add_earth_distnace(G: nx.Graph):
    """
    Add an 'earth_distance' attribute to each edge in the graph G.
    """
    set_edge_attributes(G, None, 'earth_distance')
    if type(G) == nx.MultiGraph:
        for u, v, key in G.edges(keys=True):
            coords_u = (G.nodes[u]['y'], G.nodes[u]['x'])
            coords_v = (G.nodes[v]['y'], G.nodes[v]['x'])
            G[u][v][key]['earth_distance'] = distance.geodesic(coords_u, coords_v).meters
    else: 
        for u, v in list(G.edges()):
            coords_u = (G.nodes[u]['y'], G.nodes[u]['x'])
            coords_v = (G.nodes[v]['y'], G.nodes[v]['x'])
            G[u][v]['earth_distance'] = distance.geodesic(coords_u, coords_v).meters

    

def compute_speed(G: nx.Graph, mode: str = 'max')->float:
    """
    Compute the maximum speed of the edges in the graph G.
    """
    if not _compute_speed_check(G, mode):
        print("length/earth_distance attributes are missing/have invalid values or mode is not valid (max/min/avg)!")
        return float('nan')
    
    if len(G.edges) == 0:
        return 0
    
    speeds = []
    mode_function = _speed_mode2function(mode)

    if type(G) == nx.MultiGraph:
        for u, v, key in G.edges(keys=True):
            if u != v:
                # if G[u][v][key]['length'] == 0:
                #     print(G[u][v][key])
                #     print(G.nodes[u])
                #     print(G.nodes[v])
                if G[u][v][key]['length'] != 0:
                    # Length 0 seems to be related to walking edges so they are not considered
                    speeds.append(G[u][v][key]['earth_distance'] / G[u][v][key]['length'])
    else:
        for u, v in list(G.edges()):
            if u != v:
                if G[u][v]['length'] != 0:
                    speeds.append(G[u][v]['earth_distance'] / G[u][v]['length'])    
    
    return mode_function(speeds)


def _compute_speed_check(G: nx.Graph, mode) -> bool:
    """
    Compute the number of edges in the graph G that exceed the given speed limit.
    """
    return not (list(G.edges(data=True))[0][2].get('length') is None 
            or list(G.edges(data=True))[0][2].get('length') == 0 
            or list(G.edges(data=True))[0][2].get('earth_distance') is None
            or mode not in ['max', 'min', 'avg'])

def _speed_mode2function(mode: str):
    """
    Return the appropriate function to compute speed based on the mode.
    """
    if mode == 'max':
        return np.max
    elif mode == 'min':
        return np.min
    elif mode == 'avg':
        return np.mean
    else:
        raise ValueError("Mode should be either 'max', 'min' or 'avg'")

def compute_undirected_graph_no_multi(G: nx.Graph, mode: str = 'min') -> nx.Graph:
    """
    Compute the undirected version of the graph G that also stops being a multigraph.
    """
    G_prime = nx.Graph()

    # preserve all noges and their attributes
    for node, data in G.nodes(data=True):
        G_prime.add_node(node, **data)

    # add edges
    for u, v in G.edges(keys=False):
        edges = G[u][v].values()
        
        if mode == 'max':
            chosen_edge = max(edges, key=lambda x: x['length'])
        elif mode == 'min':
            chosen_edge = min(edges, key=lambda x: x['length'])
        elif mode == 'mean':
            mean_value = sum(edge['length'] for edge in edges) / len(edges)
            chosen_edge = {'length': mean_value}
            if 'earth_distance' in G[u][v][0]:
                # If 'earth_distance' attribute is present all edges should have the same value
                chosen_edge['earth_distance'] = G[u][v][0]['earth_distance']
        else:
            raise ValueError("Invalid mode. Choose from 'max', 'min', or 'mean'.")


        G_prime.add_edge(u, v, **chosen_edge)

    G_prime = G_prime.to_undirected()
    return G_prime

def compute_ideal_graph(G: nx.Graph, speed: Optional[float] = None) -> nx.Graph:
    """
    Compute the ideal graph based on Haversine distances.
    """
    G_prime = nx.Graph()

    # preserve all nodes and their attributes
    for node, data in G.nodes(data=True):
        G_prime.add_node(node, **data)

    # add edges
    for u in G_prime.nodes:
        for v in G_prime.nodes:
            if u != v:
                coords_u = (G.nodes[u]['y'], G.nodes[u]['x'])
                coords_v = (G.nodes[v]['y'], G.nodes[v]['x'])
                earth_distance = distance.geodesic(coords_u, coords_v).meters

                if speed is not None:
                    length = earth_distance / speed
                    G_prime.add_edge(u, v, earth_distance=earth_distance, length=length)
                else:
                    G_prime.add_edge(u, v, earth_distance=earth_distance)

    G_prime = G_prime.to_undirected()
    return G_prime

def column2name(column: str) -> str:
    """
    Convert a column name to a more readable format.
    """
    name = column.replace('_', ' ')
    return name

def features2str(features: list) -> str:
    """
    Convert a list of features to a string.
    """
    header = ','.join([feature[0] for feature in features])
    values = ','.join([str(feature[1]) for feature in features])
    return header + '\n' + values