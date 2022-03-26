"""
Data loader of PPI dataset, for efficient GAT implementation.
"""

import os
import pickle
import zipfile
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import torch
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader, Dataset



def load_graph_data(config, device):
    """
    Protein-Protein Interaction dataset loader
    """
    # Instead of checking PPI in, I'd rather download it on-the-fly the first time it's needed
    PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'  # preprocessed PPI data from Deep Graph Library

    if not os.path.exists(config['ppi_path']):  # download the first time this is ran
        os.makedirs(config['ppi_path'])
        # Step 1: Download the ppi.zip (contains the PPI dataset)
        zip_tmp_path = os.path.join(config['ppi_path'], 'ppi.zip')
        download_url_to_file(PPI_URL, zip_tmp_path)
        # Step 2: Unzip it
        with zipfile.ZipFile(zip_tmp_path) as zf:
            zf.extractall(path=config['ppi_path'])
        print('Unzipping to: {} finished.'.format(config['ppi_path']))
        # Step3: Remove the temporary resource file
        os.remove(zip_tmp_path)
        print(f'Removing tmp file {zip_tmp_path}.')

    # Collect train/val/test graphs here
    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    # Dynamically determine how many graphs we have per split (avoid using constants when possible)
    num_graphs_per_split_cumulative = [0]

    splits = ['train', 'valid', 'test']
    for split in splits:
        # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
        # Note: node features are already preprocessed
        node_features = np.load(os.path.join(config['ppi_path'], f'{split}_feats.npy'))

        # PPI has 121 labels and each node can have multiple labels associated (gene ontology stuff)
        # shape = (NS, 121)
        node_labels = np.load(os.path.join(config['ppi_path'], f'{split}_labels.npy'))

        # Graph topology stored in a special nodes-links NetworkX format
        nodes_links_dict = json.load(open(os.path.join(config['ppi_path'], f'{split}_graph.json'), 'r'))
        # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
        # Use NetworkX's directed graph to explicitly model both directions
        collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
        # For each node in the above collection, ids specify to which graph the node belongs to
        graph_ids = np.load(os.path.join(config['ppi_path'], F'{split}_graph_id.npy'))
        num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))

        # Split the collection of graphs into separate PPI graphs
        for graph_id in range(np.min(graph_ids), np.max(graph_ids) + 1):
            mask = graph_ids == graph_id  # find the nodes which belong to the current graph (identified via id)
            graph_node_ids = np.asarray(mask).nonzero()[0]
            graph = collection_of_graphs.subgraph(graph_node_ids)  # returns the induced subgraph over these nodes
            print(f'Loading {split} graph {graph_id} to CPU. '
                  f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')

            # shape = (2, E) - where E is the number of edges in the graph
            edge_index = torch.tensor(list(graph.edges), dtype=torch.long).transpose(0, 1).contiguous()
            edge_index = edge_index - edge_index.min()  # bring the edges to [0, num_of_nodes] range
            edge_index_list.append(edge_index)
            # shape = (N, 50) - where N is the number of nodes in the graph
            node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float))
            # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
            node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float))


    data_loader_train = GraphDataLoader(
        node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
        node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
        edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
        batch_size=config['batch_size'],
        shuffle=True
    )

    data_loader_val = GraphDataLoader(
        node_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
        node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
        edge_index_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
        batch_size=config['batch_size'],
        shuffle=False  # no need to shuffle the validation and test graphs
    )

    data_loader_test = GraphDataLoader(
        node_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
        node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
        edge_index_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
        batch_size=config['batch_size'],
        shuffle=False
    )

    return data_loader_train, data_loader_val, data_loader_test


class GraphDataset(Dataset):
    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]


class GraphDataLoader(DataLoader):
    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range,
    Otherwise the scatter functions in the GAT model will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    return node_features, node_labels, edge_index
