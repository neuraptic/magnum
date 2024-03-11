import torch
from torch_geometric.data import Data, Batch

def get_batched_data(x, edge_index, edge_type=None):
    '''
    x: [batch_size, n_nodes, n_features]
    edge_index: [batch_size, 2, k]
    edge_type: [batch_size, 1, k]
    '''
    bs = len(x)
    if edge_type is not None:
        batch = Batch.from_data_list([Data(x=x[i], edge_index=edge_index[i], edge_type=edge_type[i]) for i in range(bs)])
        return batch.x, batch.edge_index, batch.edge_type
    else:
        batch = Batch.from_data_list([Data(x=x[i], edge_index=edge_index[i]) for i in range(bs)])
        return batch.x, batch.edge_index
    
def edge_to_adj(edge_index, n_nodes, edge_type=None, edge_w=None):
    assert edge_index.size(0) == 2
    a = torch.zeros(n_nodes, n_nodes, device=edge_index.device)

    if edge_type is not None:
        for t in edge_type.unique():
            idx = edge_index[:, edge_type==t]
            a[idx[0,:], idx[1,:]] = t.item()    
        return a
    
    a[edge_index[0,:], edge_index[1,:]] = 1 if edge_w is None else edge_w
    return a