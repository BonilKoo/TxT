import networkx as nx
import pandas as pd

from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomLinkSplit

def load_network(network_file, val_ratio, test_ratio):
    network_df = pd.read_csv(network_file)
    print(f'\nNetwork file {network_file} is loaded.')
    
    network_nx = nx.from_pandas_edgelist(network_df, source=network_df.columns[0], target=network_df.columns[1])
    print(f'\n# of nodes = {len(network_nx)}')
    print(f'# of edges = {len(network_nx.edges())}\n')
    gene_list = network_nx.nodes()
    
    network_PyG = from_networkx(network_nx)
    
    random_link_split = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True)
    network_PyG = random_link_split(network_PyG)
    train_data = network_PyG[0]
    val_data = network_PyG[1]
    test_data = network_PyG[2]
    
    return gene_list, train_data, val_data, test_data