import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, RobertaModel
from transformers.pytorch_utils import apply_chunking_to_forward

##############
### Vision ###
##############

class ViT(nn.Module):
    def __init__(self,
        from_pretrained: str = 'google/vit-base-patch16-224-in21k',
        frozen: bool = True
    ):
        super().__init__()
        self.model = ViTModel.from_pretrained(from_pretrained)
        self.d_model = self.model.config.hidden_size
        if frozen:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self,
        image: torch.Tensor            
    ):
        '''
        image: [batch_size, 3, 224, 224]
        '''
        return self.model(image).last_hidden_state

class ViTForPromptTuning(nn.Module):
    def __init__(self, 
        from_pretrained: str = 'google/vit-base-patch16-224-in21k',
        deep: int = 1, 
        deep_replace_method: str = "replace",
        frozen: bool = True
    ):
        super().__init__()
        self.model = ViT(from_pretrained=from_pretrained, frozen=frozen).model
        self.d_model = self.model.config.hidden_size
        self.deep = deep
        self.deep_replace_method = deep_replace_method

    def forward(self, 
        image: torch.Tensor, 
        prompt: torch.Tensor
    ):
        '''
        image: [batch_size, 3, 224, 224]
        prompt: [batch_size, n_deep, n_prompts, d_model]
        '''
        x = self.model.embeddings(image)
        
        g = prompt[:, 0]

        x = torch.cat([g, x], dim=1)
        L = g.size(1)
        
        for i, l in enumerate(self.model.encoder.layer):
            
            if i > 0:
                if i < self.deep:
                    if self.deep_replace_method == "replace":
                        g = prompt[:, i]
                    elif self.deep_replace_method == "accumulate":
                        previous_g_out = x[:,-L:,:]
                        g = torch.cat([previous_g_out, prompt[:, i]], dim=1)
                    x = torch.cat([g, x[:, L:, :]], dim=1)
                    L = g.size(1)
                    
            input = x
                    
            x = l.layernorm_before(x)
            q = l.attention.attention.query(x)
            k = l.attention.attention.key(x)
            v = l.attention.attention.value(x)

            q = l.attention.attention.transpose_for_scores(q)
            k = l.attention.attention.transpose_for_scores(k)
            v = l.attention.attention.transpose_for_scores(v)
            w = q @ k.transpose(-1, -2)
            w = w / np.sqrt(int(self.model.config.hidden_size/self.model.config.num_attention_heads))
            attention_probs = torch.softmax(w, dim=-1)
            if i == len(self.model.encoder.layer)-1:
                self.attention_matrix = w
            attention_probs = l.attention.attention.dropout(attention_probs)
            v = (attention_probs @ v).permute(0, 2, 1, 3).contiguous()
            v = v.view(x.size(0), x.size(1), -1)

            attention_output = l.attention.output(v, None)
            x = attention_output + input

            layer_output = l.layernorm_after(x)
            layer_output = l.intermediate(layer_output)

            x = l.output(layer_output, x)

        return x
    
class ViTPromptBottleneck(nn.Module):
    def __init__(self,
        L: int,
        deep: int = 1,
        deep_replace_method: str = "replace"
    ):
        super().__init__()
        self.L = L
        self.model = ViTForPromptTuning(deep=deep, deep_replace_method=deep_replace_method)
        self.d_model = self.model.d_model
        self.prompts = nn.Parameter(torch.zeros(deep, L, self.model.d_model))
        nn.init.xavier_uniform_(self.prompts.data)
        
    def forward(self,
        image: torch.Tensor,
        add_cls_token_output: bool = False,
        get_attention_matrix: bool = False
    ):
        bs = image.size(0)
        prompts = self.prompts.repeat(bs,1,1,1)
        out = self.model(image, prompts)
        
        idx = self.L+1 if add_cls_token_output else self.L
        
        if get_attention_matrix:
            return out[:,:idx,:], self.model.attention_matrix[:, :, :idx, :idx].mean(dim=1).softmax(dim=-1)
            
        return out[:,:idx,:]
    
################
### Language ###
################

class RoBERTaPromptTuning(nn.Module):
    def __init__(self, 
        from_pretrained: str = 'roberta-large', 
        deep: int = 1,
        deep_replace_method: str = "replace",
        frozen: bool = True
    ):
        super().__init__()
        self.model = RobertaModel.from_pretrained(from_pretrained)
        self.embedding_layer = self.model.embeddings
        self.d_model = self.model.config.hidden_size
        if frozen:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        self.deep = deep
        self.deep_replace_method = deep_replace_method
        self.layers = self.model.encoder.layer
        
    def forward_layer(self, x: torch.Tensor, attn_mask: torch.Tensor, layer_id: int = None):
        '''
        x: latent embedding state [batch_size, max_length, d_model]
        attn_mask: boolean mask [batch_size, max_length]
        layer_id: encoder layer to use
        '''
        l = self.layers[layer_id]
            
        q = l.attention.self.query(x)
        k = l.attention.self.key(x)
        v = l.attention.self.value(x)
        
        input_shape = x.size()
        extended_attention_mask = self.model.get_extended_attention_mask(attn_mask, input_shape)

        q = l.attention.self.transpose_for_scores(q)
        k = l.attention.self.transpose_for_scores(k)
        v = l.attention.self.transpose_for_scores(v)
        w = q @ k.transpose(-1, -2)
        w = w / np.sqrt(int(self.model.config.hidden_size/self.model.config.num_attention_heads))
        w = w + extended_attention_mask
        attention_probs = torch.softmax(w, dim=-1)
        if layer_id == len(self.layers)-1:
            self.attention_matrix = w
        attention_probs = l.attention.self.dropout(attention_probs)
        v = (attention_probs @ v).permute(0, 2, 1, 3).contiguous()
        v = v.view(x.size(0), x.size(1), -1)

        attention_output = l.attention.output(v, x)

        x = apply_chunking_to_forward(l.feed_forward_chunk, l.chunk_size_feed_forward, l.seq_len_dim, attention_output)
        
        return x
        
    def forward(self, text_tokens: torch.LongTensor, attn_mask: torch.Tensor, prompt: torch.Tensor):
        '''
        text_tokens: [batch_size,  max_length]
        attn_mask: [batch_size, max_length]
        prompt: [batch_size, n_deep, n_prompts, d_model]
        '''
        
        x = self.embedding_layer(text_tokens)
        g = prompt[:, 0]
        x = torch.cat([g, x], dim=1)
        L = g.size(-2)
        attn_mask = torch.cat([torch.ones(x.size(0), g.size(-2), device=x.device), attn_mask], dim=-1)
        
        for i in range(len(self.layers)):
            
            if i > 0:
                if i < self.deep:
                    if self.deep_replace_method == "replace":
                        g = prompt[:, i]
                    x = torch.cat([g, x[:,L:,:]], dim=1)
            x = self.forward_layer(x, attn_mask, layer_id=i)
                
        return x
    
class RoBERTaPromptBottleneck(nn.Module):
    def __init__(self,
        L: int,
        deep: int = 1,
        deep_replace_method: str = "replace"
    ):
        super().__init__()
        self.L = L
        self.model = RoBERTaPromptTuning(deep=deep, deep_replace_method=deep_replace_method)
        self.d_model = self.model.d_model
        self.prompts = nn.Parameter(torch.zeros(deep, L, self.model.d_model))
        nn.init.xavier_uniform_(self.prompts.data)
        
    def forward(self,
        text_tokens: torch.LongTensor, 
        attn_mask: torch.Tensor,
        add_cls_token_output: bool = False,
        get_attention_matrix: bool = False
    ):
        bs = len(text_tokens)
        prompts = self.prompts.repeat(bs,1,1,1)
        out = self.model(text_tokens, attn_mask, prompts)
            
        idx = self.L+1 if add_cls_token_output else self.L
        
        if get_attention_matrix:
            return out[:,:idx,:], self.model.attention_matrix[:, :, :idx, :idx].mean(dim=1).softmax(dim=-1)
            
        return out[:,:idx,:]

###############
### Tabular ###
###############

class TabularMapper(nn.Module):
    def __init__(self, 
        d_model: float,
        n_num_vars: int = None,
        n_cat_vars: int = None,
        num_cat_vars_classes: int = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_num_vars = n_num_vars
        self.n_cat_vars = n_cat_vars
        self.num_cat_vars_classes = num_cat_vars_classes
        if n_num_vars is not None:
            self.num_proj = nn.ModuleList([nn.Linear(1, self.d_model) for _ in range(n_num_vars)])
        if n_cat_vars is not None:
            self.cat_proj = nn.ModuleList([nn.Embedding(num_cat_vars_classes[i], self.d_model) for i in range(n_cat_vars)])
        
    def forward(self, x_num: torch.Tensor = None, x_cat: torch.LongTensor = None, **kwargs):
        device = x_num.device if x_num is not None else x_cat.device
        x_num_proj = torch.cat([self.num_proj[i](x_num[:,i].view(-1,1)).unsqueeze(1) for i in range(self.n_num_vars)], dim=1) if self.n_num_vars is not None else torch.tensor([], device=device)
        x_cat_proj = torch.cat([self.cat_proj[i](x_cat[:,i]).unsqueeze(1) for i in range(self.n_cat_vars)], dim=1) if self.n_cat_vars is not None else torch.tensor([], device=device)            
        x = torch.cat([x_num_proj, x_cat_proj], dim=1)
        
        return x