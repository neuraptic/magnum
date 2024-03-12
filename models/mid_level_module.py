import torch
import torch.nn as nn
from models.utils import get_batched_data

class GraphPooling(nn.Module):
    def __init__(self, d_model, knn_k):
        super().__init__()
        self.d_model = d_model
        self.k = knn_k
        self.edge_pool = gnn.pool.EdgePooling(self.d_model)
        
    def forward(self, x: torch.Tensor):
        '''
        x: [batch_size, n_nodes, n_features]
        '''
        x_list = []
        edge_index_list = []
        batch_idx_list = []
        for i in range(x.size(0)):
            x_ = x[i]
            edge_index_ = tc.knn_graph(x_, k=self.k, loop=True)
            #edge_index_ = torch.ones(x_.size(0), x_.size(0)).nonzero().T
            x_, edge_index_, batch_idx_, _ = self.edge_pool(x[i], edge_index_, batch=torch.zeros(x.size(1), device=x_.device).long())
            x_list.append(x_)
            edge_index_list.append(edge_index_)
            batch_idx_list.append(batch_idx_+i)
        batch_idx = torch.cat(batch_idx_list)
        
        batch_x, batch_edge_index = get_batched_data(x_list, edge_index_list)
        
        return batch_x, batch_edge_index, batch_idx         
    
class Mix(nn.Module):
    def __init__(self, d_model, d_hidden, n_attn_heads):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_attn_heads = n_attn_heads
        self.layer = gnn.GATv2Conv(self.d_model, self.d_hidden, heads=self.n_attn_heads, concat=False)
        
    def forward(self, x, edge_index, batch_idx):
        out = self.layer(x=x, edge_index=edge_index)
        return [out[batch_idx==i] for i in batch_idx.unique()]