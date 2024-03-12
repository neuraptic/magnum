import torch
import torch.nn as nn
from models.mid_level_module import Mix, GraphPooling
from models.high_level_module import MultimodalGatedFusion

class BottomLevelModule(nn.Module):
    def __init__(self,
        d_model: int,
        language_model: torch.nn.Module = None,
        vision_model: torch.nn.Module = None,
        tabular_model: torch.nn.Module = None,
        language_mapper: torch.nn.Module = None,
        vision_mapper: torch.nn.Module = None,
        tabular_mapper: torch.nn.Module = None
    ):
        super().__init__()
        self.d_model = d_model
        self.language_model = language_model
        self.vision_model = vision_model
        self.tabular_model = tabular_model
        self.language_mapper = language_mapper if language_mapper is not None else nn.Identity()
        self.vision_mapper = vision_mapper if vision_mapper is not None else nn.Identity()
        self.tabular_mapper = tabular_mapper if tabular_mapper is not None else nn.Identity()
        
    def forward(self, tab_data=None, vis_data=None, lan_data=None):
        tab = self.tabular_mapper(self.tabular_model(**tab_data)) if tab_data is not None else None
        vis = self.vision_mapper(self.vision_model(vis_data, add_cls_token_output=True)) if vis_data is not None else None
        lan = self.language_mapper(self.language_model(**lan_data, add_cls_token_output=True)) if lan_data is not None else None
        return tab, vis, lan
    
class TopLevelModule(nn.Module):
    def __init__(self, d_model, hidden_size, gate_input_type, gate_output_type, k, n_output_classes, modalities=["tabular", "vision", "language"]):
        super().__init__()
        if "tabular" in modalities:
            self.tab_graph_pooling = GraphPooling(d_model=d_model, knn_k=k)
            self.tab_mix = Mix(d_model=d_model, d_hidden=d_model, n_attn_heads=1)
        if "vision" in modalities:
            self.vis_graph_pooling = GraphPooling(d_model=d_model, knn_k=k)
            self.vis_mix = Mix(d_model=d_model, d_hidden=d_model, n_attn_heads=1)
        if "language" in modalities:
            self.lan_graph_pooling = GraphPooling(d_model=d_model, knn_k=k)
            self.lan_mix = Mix(d_model=d_model, d_hidden=d_model, n_attn_heads=1)
        
        self.gate = MultimodalGatedFusion(d_model, len(modalities), hidden_size, gate_input_type, gate_output_type)
        
        self.classification_head = nn.Linear(hidden_size, n_output_classes)
        
        self.n_output_classes = n_output_classes
        
    def forward(self, tab_nodes=None, vis_nodes=None, lan_nodes=None):
        
        if tab_nodes is not None:
            tab_pool_out = self.tab_graph_pooling(tab_nodes)
            tab_out = self.tab_mix(*tab_pool_out)
        else:
            tab_out = None
            
        if vis_nodes is not None:
            vis_pool_out = self.vis_graph_pooling(vis_nodes)
            vis_out = self.vis_mix(*vis_pool_out)
        else:
            vis_out = None
            
        if lan_nodes is not None:
            lan_pool_out = self.lan_graph_pooling(lan_nodes)
            lan_out = self.lan_mix(*lan_pool_out)
        else:
            lan_out = None
            
        if tab_out is None:
            vis = torch.cat([v.mean(dim=0)[None,:] for v in vis_out], dim=0)
            lan = torch.cat([l.mean(dim=0)[None,:] for l in lan_out], dim=0)
            x = (vis, lan)
        elif vis_out is None:
            tab = torch.cat([t.mean(dim=0)[None,:] for t in tab_out], dim=0)
            lan = torch.cat([l.mean(dim=0)[None,:] for l in lan_out], dim=0)
            x = (tab, lan)
        elif lan_out is None:
            tab = torch.cat([t.mean(dim=0)[None,:] for t in tab_out], dim=0)
            vis = torch.cat([v.mean(dim=0)[None,:] for v in vis_out], dim=0)
            x = (tab, vis)
        else:
            tab = torch.cat([t.mean(dim=0)[None,:] for t in tab_out], dim=0)
            vis = torch.cat([v.mean(dim=0)[None,:] for v in vis_out], dim=0)
            lan = torch.cat([l.mean(dim=0)[None,:] for l in lan_out], dim=0)
            x = (tab, vis, lan)

        x = self.gate(*x)
        x = self.classification_head(x)
        
        return x if self.n_output_classes > 1 else x.view(-1)
    
class Magnum(nn.Module):  
    def __init__(self,
        bottom_level_module: torch.nn.Module,
        high_level_module: torch.nn.Module,
    ):
        super().__init__()
        self.bottom_level_module = bottom_level_module
        self.high_level_module = high_level_module
        
    def forward(self, tab_data=None, vis_data=None, lan_data=None):
        tab, vis, lan = self.bottom_level_module(tab_data, vis_data, lan_data)
        out = self.high_level_module(tab, vis, lan)
        return out
