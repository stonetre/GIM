import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter
from model.GIM.GIM_utils import *
import numpy as np
from mamba_ssm import Mamba
from itertools import permutations
from multiprocessing import Pool
from model.GVAE.Invertible import RevIN
from model.GVAE.GVAE_modules import *
from model.GIM.GIM_models import *

class GNN(nn.Module):
    """
    Reimplementaion of the Message-Passing class in torch-geometric to allow more flexibility.
    """
    def __init__(self):
        super(GNN, self).__init__()
        
    def forward(self, *input):
        raise NotImplementedError

    def propagate(self, x: Tensor, es: Tensor, agg: str='mean') -> Tensor:
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            x: [node, ..., dim], node embeddings 
        """
        msg, idx, size = self.message(x, es)
        x = self.aggregate(msg, idx, size, agg)                          
        return x

    def aggregate(self, msg: Tensor, idx: Tensor, size: int, agg: str='mean') -> Tensor:
        """
        Args:
            msg: [E, ..., dim * 2]
            idx: [E]
            size: number of nodes
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            aggregated node embeddings
        """
        assert agg in {'add', 'mean', 'max'}
        return scatter(msg, idx, dim_size=size, dim=0, reduce=agg)

    def node2edge(self, x_i: Tensor, x_o: Tensor) -> Tensor:
        """
        Args:
            x_i: [E, ..., dim], embeddings of incoming nodes
            x_o: [E, ..., dim], embeddings of outcoming nodes

        Return:
            edge embeddings
        """
        return torch.cat([x_i, x_o], dim=-1)

    def message(self, x: Tensor, es: Tensor, option: str='o2i'):
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            option: default: 'o2i'
                'o2i': collecting incoming edge embeddings
                'i2o': collecting outcoming edge embeddings

        Return:
            mgs: [E, ..., dim * 2], edge embeddings
            col: [E], indices of 
            size: number of nodes
        """
        if option == 'i2o':
            row, col = es
        if option == 'o2i':
            col, row = es
        else:
            raise ValueError('i2o or o2i')
        x_i, x_o = x[row], x[col]
        msg = self.node2edge(x_i, x_o)
        return msg, col, len(x)

    def update(self, x):
        return x


class MLP4d(nn.Module):
    def __init__(self, node_num, n_in, feature_len, n_hid, n_out, 
                 act_fn=torch.nn.ELU(), conv_kernel=(1,1), use_layer_norm=True, dropout=0.):
        super(MLP4d, self).__init__()
        self.node_num = node_num
        self.dim_fc = nn.Linear(n_in, n_hid)
        self.fc1 = nn.Linear(feature_len, feature_len*2)
        self.fc2 = nn.Linear(feature_len*2, feature_len)
        self.bn = nn.BatchNorm1d(feature_len)
        self.ln_node = nn.LayerNorm([node_num, feature_len])
        self.ln_edge = nn.LayerNorm([node_num*(node_num-1), feature_len])
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.use_layer_norm = use_layer_norm
    
    def layer_norm(self, inputs):
        if inputs.shape[2]==self.node_num:
            x = self.ln_node(inputs)
        else:
            x = self.ln_edge(inputs)
        return x
    
    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1)*inputs.size(2), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), inputs.size(2), -1)

    def forward(self, inputs, t=None):
        # inputs:[E/N, B, L, D * 2]
        x = self.act_fn(self.dim_fc(inputs)) # [E, B, L, D]
        x = x.permute(1,3,0,2)              # [B, D, E, L]
        x = self.act_fn(self.fc1(x))
        x = self.dropout_fn(x)
        x = self.act_fn(self.fc2(x)) # [B, D, E, L]
        if self.use_layer_norm:
            x = self.layer_norm(x)
        else:
            x = self.batch_norm(x)
        return x


class MM4d(nn.Module):
    def __init__(self, node_num, n_in, feature_len, n_hid, n_out, act_fn=nn.SiLU(), 
                 conv_kernel=2, use_layer_norm=True, dropout=0.):
        super(MM4d, self).__init__()
        self.node_num = node_num
        self.dim_layer = nn.Linear(n_in, n_hid)
        self.mamba = Mamba(d_model=n_hid, d_state=n_hid*2, d_conv=conv_kernel, expand=1)
        self.dim_out = nn.Linear(n_hid, n_out)
        self.ln = nn.LayerNorm([feature_len, n_out])
        self.bn = nn.BatchNorm1d(n_out)
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.use_layer_norm = use_layer_norm
    
    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs, t=None):
        # inputs:[E/N, B, L, D * 2]
        x = inputs.reshape(inputs.size(0)*inputs.size(1), inputs.size(2), inputs.size(3)) # N*B, L, D
        x = self.dim_layer(x) #[N*B, L, D]
        x = self.act_fn(self.mamba(x))
        x = self.dropout_fn(x)
        x = self.dim_out(x)
        if self.use_layer_norm:
            x = self.ln(x)  # for L,D
        else:
            x = self.batch_norm(x) # for D  
        x = x.view(inputs.size(0), inputs.size(1), inputs.size(2), -1).permute(1,3,0,2)  # [N, B, L, D]->[B, D, N, L]
        return x


class Interaction_MLP4d_layer(GNN):
    def __init__(self, node_num, input_dim, feature_len, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, use_edge_feature = False,
                 attention=False, drop_rate=0.):
        super(Interaction_MLP4d_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type

        edge_dim = input_dim*2
        edge_nn = MLP4d(node_num=node_num, n_in=edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                      n_out=hidden_dim, act_fn=act_fn, conv_kernel=(1,1), use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.edge_block = nn.ModuleList([edge_nn for _ in range(edge_types)])

        node_edge_dim = hidden_dim*2
        self.node_block = MLP4d(node_num=node_num, n_in=hidden_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                n_out=hidden_dim, act_fn=act_fn, conv_kernel=(1,1), use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.node_edge_block = MLP4d(node_num=node_num, n_in=node_edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                    n_out=output_dim, act_fn=act_fn, conv_kernel=(1,1), use_layer_norm=use_layer_norm, dropout=drop_rate)
        
        if self.attention:
            self.att_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)),
                nn.Sigmoid())

        edge_index = np.array(list(permutations(range(node_num), 2))).T
        self.edge_index = torch.LongTensor(edge_index)

    # Node to edge
    def forward(self, node_feature, edge_feature, nodes_mask, edges_mask, real_edges = None):
        # node_feature : [B, D, N, L], real_edges:[B, E, K]
        x = node_feature.permute(2,0,3,1) # x :[N, B, L, D]
        edge_index = self.edge_index.to(x.device)
        # real_edges: [B, E, K] -> [L, B, E, K] -> [B, L, E, K]
        if real_edges is not None:
            real_edges = real_edges.repeat(x.size(2), 1, 1, 1).permute(1, 0, 2, 3).contiguous()
        msg, col, size = self.message(x, edge_index)  # node to edge
        # mgs: [E, B, L, dim * 2]
        # col:[E], size: number of nodes
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        norm = len(self.edge_block)
        if self.skip_first_edge_type:
            norm -= 1
        if real_edges is not None: 
            msgs = sum(self.edge_block[i](msg).permute(0,3,2,1) * torch.select(real_edges, -1, i).unsqueeze(-1) / norm
                    for i in range(start_idx, len(self.edge_block))).permute(0,3,2,1)
        else:
            msgs = sum(self.edge_block[i](msg) / norm
                    for i in range(start_idx, len(self.edge_block)))   # # [B, D, E, L]
        if edges_mask is not None:
            msgs = torch.einsum('abcd,ab->abcd', msgs.permute(0,2,1,3), edges_mask)

        if self.attention:
            att_val = self.att_conv(msgs)
            msgs = msgs * att_val

        x_adj = self.aggregate(msgs.permute(1,0,3,2), col, size)  # edge to node [E, ..., dim * 2]
        node_x = torch.cat([x, x_adj], dim=-1)
        out = self.node_edge_block(node_x)
        # out = node_feature + delta  # [B,D,N,L]
        if nodes_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        return out, msgs.permute(1,0,3,2)



class Interaction_MM4d_layer(GNN):
    def __init__(self, node_num, input_dim, feature_len, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, use_edge_feature=False,
                 attention=False, drop_rate=0.):
        super(Interaction_MM4d_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type
        self.use_edge_feature = use_edge_feature

        edge_dim = input_dim*(2+1) if use_edge_feature else input_dim*2
        edge_nn = MM4d(node_num=node_num, n_in=edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                      n_out=hidden_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.edge_block = nn.ModuleList([edge_nn for _ in range(edge_types)])

        node_edge_dim = hidden_dim*2
        self.node_block = MM4d(node_num=node_num, n_in=hidden_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                n_out=hidden_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.node_edge_block = MM4d(node_num=node_num, n_in=node_edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                    n_out=output_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        
        if self.attention:
            self.att_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)),
                nn.Sigmoid())

        edge_index = np.array(list(permutations(range(node_num), 2))).T
        self.edge_index = torch.LongTensor(edge_index)

    # Node to edge
    def forward(self, node_feature, edge_feature, nodes_mask, edges_mask, real_edges = None):
        # node_feature : [B, D, N, L], real_edges:[B, E, K]
        x = node_feature.permute(2,0,3,1) # x :[N, B, L, D]
        edge_index = self.edge_index.to(x.device)
        # real_edges: [B, E, K] -> [L, B, E, K] -> [B, L, E, K]
        if real_edges is not None:
            real_edges = real_edges.repeat(x.size(2), 1, 1, 1).permute(1, 0, 2, 3).contiguous()
        msg, col, size = self.message(x, edge_index)  # node to edge
        # mgs: [E, B, L, dim * 2]
        if self.use_edge_feature and edge_feature is not None:
            msg = torch.cat([msg, edge_feature], dim=-1)
        # col:[E], size: number of nodes
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        norm = len(self.edge_block)
        if self.skip_first_edge_type:
            norm -= 1
        if real_edges is not None: # b,L,E,D
            msgs = sum(self.edge_block[i](msg).permute(0,3,2,1) * torch.select(real_edges, -1, i).unsqueeze(-1) / norm
                    for i in range(start_idx, len(self.edge_block))).permute(0,3,2,1)
        else:
            msgs = sum(self.edge_block[i](msg) / norm
                    for i in range(start_idx, len(self.edge_block)))   # # [B, D, E, L]
        if edges_mask is not None:
            msgs = torch.einsum('abcd,ab->abcd', msgs.permute(0,2,1,3), edges_mask)

        if self.attention:
            att_val = self.att_conv(msgs)
            msgs = msgs * att_val

        x_adj = self.aggregate(msgs.permute(1,0,3,2), col, size)  # edge to node [E, ..., dim * 2]
        node_x = torch.cat([x, x_adj], dim=-1)
        delta = self.node_edge_block(node_x)
        out = node_feature + delta  # [B,D,N,L]
        if nodes_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        return out, msgs.permute(1,0,3,2)


class SCM_MM4d_layer(GNN):
    def __init__(self, node_num, input_dim, feature_len, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, use_edge_feature=False,
                 attention=False, t_node_guide=False, t_edge_guide=False, drop_rate=0.):
        super(SCM_MM4d_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type
        self.t_node_guide = t_node_guide
        self.t_edge_guide = t_edge_guide
        self.input_dim = input_dim
        self.edge_types = edge_types
        self.use_edge_feature = use_edge_feature
        self.node_block = MM4d(node_num=node_num, n_in=hidden_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                n_out=hidden_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        if self.attention:
            self.att_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)),
                nn.Sigmoid())

        edge_index = np.array(list(permutations(range(node_num), 2))).T
        self.edge_index = torch.LongTensor(edge_index)

    # Node to edge
    def forward(self, node_feature, edge_feature, nodes_mask, edges_mask, real_edges = None):
        # node_feature : [B, D, N, L], real_edges:[B, E, K]
        x = node_feature.permute(2,0,3,1) # x :[N, B, L, D]
        edge_index = self.edge_index.to(x.device)
        # real_edges: [B, E, K] -> [L, B, E, K] -> [B, L, E, K]
        if real_edges is not None:
            real_edges = real_edges.repeat(x.size(2), 1, 1, 1).permute(1, 0, 2, 3).contiguous()
        msg, col, size = self.message(x, edge_index)  # node to edge
        # mgs: [E, B, L, dim * 2]  
        if self.use_edge_feature:
            msg[:,:,:,:self.input_dim] = (msg[:,:,:,:self.input_dim] + edge_feature)/2
        # col:[E], size: number of nodes
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        norm = self.edge_types
        if self.skip_first_edge_type:
            norm -= 1
            
        if real_edges is not None: 
            msgs = sum(msg[:,:,:,:self.input_dim].permute(1,2,0,3) * torch.select(real_edges, -1, i).unsqueeze(-1)/ norm 
                        for i in range(start_idx, self.edge_types)).permute(0,3,2,1)
        else:
            msgs = msg[:,:,:,:self.input_dim].permute(0,3,2,1) # [B, D, E, L]

        if edges_mask is not None:
            msgs = torch.einsum('abcd,ab->abcd', msgs.permute(0,2,1,3), edges_mask)

        if self.attention:
            att_val = self.att_conv(msgs)
            msgs = msgs * att_val

        x_adj = self.aggregate(msgs.permute(1,0,3,2), col, size)  # edge to node 

        node_x = x_adj
        delta = self.node_block(node_x)
        out = node_feature + delta
        if nodes_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        return out, msgs.permute(1,0,3,2)  #[E, B, L, dim * 2]


class GSL_gim(nn.Module):
    def __init__(self, node_num, 
                 node_feature_dim, 
                 node_feature_len, 
                 node_prior_len,
                 node_description_dim, 
                 node_description_len, 
                 edge_types, 
                 hidden_dim, 
                 skip_first_edge_type, 
                 interact_type = 'mlp', 
                 edge_concat_methad = 'cat',
                 act_fn=nn.SiLU(), 
                 feature_layers=4, 
                 description_layers=2, 
                 attention=False,
                 out_edge_types=2,                   
                 use_layer_norm=False, 
                 use_description_learner=False, 
                 use_prior_learner=False, 
                 drop_rate=0., 
                 device='cpu'):
        super(GSL_gim, self).__init__()

        self.hidden_dim = hidden_dim
        self.device = device
        self.edge_concat_methad = edge_concat_methad
        self.use_description_learner = use_description_learner
        self.node_feature_dim = node_feature_dim
        self.node_feature_len = node_feature_len
        # node feature Encoder
        self.feature_layers = feature_layers
        self.interact_type = interact_type
        self.use_prior_learner = use_prior_learner

        kernel_size=1
        self.embedding_feature = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                               nn.Conv2d(node_feature_dim, hidden_dim, kernel_size=(kernel_size, 1)))  
        
        if interact_type == 'mlp':
            Interaction_nn = Interaction_MLP4d_layer
        elif interact_type == 'mamba':                   
            Interaction_nn = Interaction_MM4d_layer
        else:
            raise KeyError
        
        des_Interaction_nn = Interaction_MLP4d_layer

        for i in range(0, feature_layers):
            self.add_module("gcl_feature_%d" % i, Interaction_nn(
                node_num = node_num, 
                input_dim = hidden_dim, 
                feature_len = node_feature_len,
                output_dim =hidden_dim, 
                hidden_dim=hidden_dim,
                act_fn=act_fn, 
                edge_types=edge_types, 
                skip_first_edge_type=skip_first_edge_type, 
                use_layer_norm = use_layer_norm, 
                attention=attention, 
                drop_rate=drop_rate))
            
        self.dim_fc = nn.Sequential(
                    nn.Linear(hidden_dim*feature_layers, hidden_dim),
                    act_fn,
                    nn.Linear(hidden_dim, hidden_dim),
                    act_fn,
                )
        self.fc_out = nn.Linear(hidden_dim*node_feature_len, out_edge_types)

        # Node description Encoder
        self.description_layers=description_layers
        if use_description_learner:
            kernel_size = 1
            self.embedding_len = nn.Sequential(nn.Linear(node_description_dim*node_description_len, node_feature_len),
                                                    act_fn)
            self.embedding_description = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                                    nn.Conv2d(node_feature_dim, hidden_dim, kernel_size=(kernel_size, 1)))  
            for i in range(0, description_layers):
                self.add_module("gcl_description_%d" % i, des_Interaction_nn(
                node_num = node_num, 
                input_dim = hidden_dim, 
                feature_len = node_feature_len,
                output_dim =hidden_dim, 
                hidden_dim=hidden_dim,
                act_fn=act_fn, 
                edge_types=edge_types, 
                skip_first_edge_type=skip_first_edge_type, 
                use_layer_norm = use_layer_norm, 
                attention=attention, 
                drop_rate=drop_rate))
        
            self.dim_fc_des = nn.Sequential(
                        nn.Linear(hidden_dim*description_layers, hidden_dim),
                        act_fn,
                        nn.Linear(hidden_dim, hidden_dim),
                        act_fn,
                    )
            self.fc_out_des = nn.Linear(hidden_dim*node_feature_len, out_edge_types)

        if use_prior_learner:
            kernel_size=1
            self.embedding_prior = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                                nn.Conv2d(node_feature_dim, hidden_dim, kernel_size=(kernel_size, 1)))  
            for i in range(0, feature_layers):
                self.add_module("gcl_prior_%d" % i, Interaction_nn(
                    node_num = node_num, 
                    input_dim = hidden_dim, 
                    feature_len = node_feature_len,
                    output_dim =hidden_dim, 
                    hidden_dim=hidden_dim,
                    act_fn=act_fn, 
                    edge_types=edge_types, 
                    skip_first_edge_type=skip_first_edge_type, 
                    use_layer_norm = use_layer_norm, 
                    attention=attention, 
                    drop_rate=drop_rate))
        
            self.dim_fc_prior = nn.Sequential(
                        nn.Linear(hidden_dim*feature_layers, hidden_dim),
                        act_fn,
                        nn.Linear(hidden_dim, hidden_dim),
                        act_fn,
                    )
            self.fc_out_prior = nn.Linear(hidden_dim*node_feature_len, out_edge_types)
        self.to(self.device)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
        

    def forward(self, x, p, h, nodes_mask, edges_mask):
        # node feature Encoder
        # x: [B, N, D, L]
        z_g_out = []
        edge_list = []
        x = x.permute(0, 2, 1, 3).to(torch.float32)
        x = self.embedding_feature(x)
        if nodes_mask is not None:
            x = torch.einsum('abcd,ab->abcd', x.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        for i in range(0, self.feature_layers):
            x, g2 = self._modules["gcl_feature_%d" % i](x, None, nodes_mask, edges_mask) # g2: [B, D, E, L]
            edge_list.append(g2)

        g = torch.cat(edge_list, dim=-1).permute(1,0,2,3) # B,E,L,D
        g = self.dim_fc(g)
        g = g.reshape(g.size(0), g.size(1), -1).contiguous()
        z_g = self.fc_out(g)
        z_g = (z_g.permute(2,0,1) * edges_mask).permute(1,2,0)
        z_g_out.append(z_g)

        # node description Encoder
        if self.use_description_learner:
            des_edge_list = []
            h = self.embedding_len(h.to(torch.float32))
            h = h.permute(0, 2, 1, 3)
            h = self.embedding_description(h)
            if nodes_mask is not None:
                h = torch.einsum('abcd,ab->abcd', h.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
            for i in range(0, self.description_layers):
                h, g2 = self._modules["gcl_description_%d" % i](h, None, nodes_mask, edges_mask) # g2: [B, D, E, L]
                des_edge_list.append(g2)

            g_des = torch.cat(des_edge_list, dim=-1).permute(1,0,2,3) # B,E,L,D
            g_des = self.dim_fc_des(g_des)
            g_des = g_des.reshape(g_des.size(0), g_des.size(1), -1).contiguous()
            z_g_des = self.fc_out_des(g_des)
            z_g_des = (z_g_des.permute(2,0,1) * edges_mask).permute(1,2,0)
            z_g_out.append(z_g_des)

        # node prior Encoder
        if self.use_prior_learner:
            prior_edge_list = []
            p = p.permute(0, 2, 1, 3).to(torch.float32)
            p = self.embedding_prior(p)
            if nodes_mask is not None:
                p = torch.einsum('abcd,ab->abcd', p.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
            for i in range(0, self.feature_layers):
                p, g3 = self._modules["gcl_prior_%d" % i](p, None, nodes_mask, edges_mask) # g2: [B, D, E, L]
                prior_edge_list.append(g3)

            g_prior = torch.cat(prior_edge_list, dim=-1).permute(1,0,2,3) # B,E,L,D
            g_prior = self.dim_fc_prior(g_prior)
            g_prior = g_prior.reshape(g_prior.size(0), g_prior.size(1), -1).contiguous()
            z_g_prior = self.fc_out_prior(g_prior)
            z_g_prior = (z_g_prior.permute(2,0,1) * edges_mask).permute(1,2,0)
            z_g_out.append(z_g_prior)

        return z_g_out

class Node_encoder(nn.Module):
    def __init__(self, node_num, node_feature_len, 
                 node_feature_dim, hidden_dim, 
                 feature_layer, layer_type,
                 out_dim, out_len,
                 act_fn, use_layer_norm, drop_rate):
        super(Node_encoder, self).__init__()
        self.use_independ_encoder = False
        self.use_norm = False
        self.layer_type = layer_type
        self.feature_layer = feature_layer
        self.hidden_dim = hidden_dim
        self.out_len = out_len
        self.node_num = node_num
        # node feature Encoder
        kernel_size = 1
        self.embedding_feature = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                               nn.Conv2d(node_feature_dim, hidden_dim, kernel_size=(kernel_size, 1)))
        nn_layer = nn.Sequential()
        for i in range(0, feature_layer):
                nn_layer.add_module("input_enc_%d" % i, MM4d(node_num=node_num, n_in=hidden_dim, feature_len=node_feature_len, n_hid=hidden_dim, 
                                    n_out=hidden_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate))

        if self.use_independ_encoder: 
            for i in range(0, node_num):
                self.add_module("input_enc_%d" % i, nn_layer)
                self.add_module("input_mu_%d" % i, nn.Sequential(act_fn, 
                                                                nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                                                nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))
                                                                ))
                self.add_module("input_sigma_%d" % i, nn.Sequential(act_fn,
                                                                    nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                                                    nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))
                                                                    ))
        else:
            self.input_enc = nn_layer
            self.input_mu = nn.Sequential(
                                        nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                        nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1)),
                                        act_fn
                                        )
            self.input_sigma = nn.Sequential(
                                            nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                            nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1)),
                                            act_fn
                                            )    

    def forward(self, x, nodes_mask, kl_mask):
    
        x = x.permute(0, 2, 1, 3).to(torch.float32)   # B,D,N,L
        x = self.embedding_feature(x)    
        batch_size, dim, node, time = x.size()

        if not self.use_independ_encoder:
            x = self.input_enc(x.permute(2,0,3,1))   #： [B, D, N, L]
            x = x[:,:,:,-self.out_len:] 
            x_mu = self.input_mu(x)
            x_sigma = self.input_sigma(x)

        else:
            x_mu_list=[]
            x_sigma_list=[]
            for i in range(self.node_num):
                x_ = self._modules["input_enc_%d" % i](x[:,:,i,:].unsqueeze(2).permute(2,0,3,1)) 
                x_sigma = self._modules["input_sigma_%d" % i](x_)  #BDN1
                x_mu_list.append(x_)
                x_sigma_list.append(x_sigma)
            x_mu = torch.cat(x_mu_list, dim=2)
            x_sigma = torch.cat(x_sigma_list, dim=2)
            x_mu = x_mu[:,:,:,-self.out_len:]
            x_sigma = x_sigma[:,:,:,-self.out_len:]

        # reparameterization trick
        if self.training:
            z_x = repara_trick(x_mu, x_sigma, kl_mask)
        else:
            z_x = x_mu
        

        if nodes_mask is not None:
            z_x = torch.einsum('abcd,ab->abcd', z_x.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        return z_x, x_mu, x_sigma


class GIM_decoder(nn.Module):
    def __init__(self, node_num, node_feature_dim, node_feature_len, edge_types, 
                 out_node_feature_dim, out_node_feature_len,
                 hidden_dim, skip_first_edge_type, interact_type = 'mlp',
                 act_fn=nn.SiLU(), feature_layers=2, attention=False,
                 use_layer_norm=False,  use_edge_feature=True, drop_rate=0., device='cpu'):
        super(GIM_decoder, self).__init__()
        if out_node_feature_dim is None:
            out_node_feature_dim = node_feature_dim
        self.node_feature_dim = node_feature_dim
        self.interact_type = interact_type
        self.feature_layers = feature_layers
        self.use_edge_feature = use_edge_feature
        self.node_feature_len = node_feature_len
        self.node_num = node_num
        self.out_node_feature_len = out_node_feature_len

        kernel_size = 1
        self.embedding_feature = nn.Sequential(nn.ZeroPad2d(padding=(0, 0, kernel_size-1, 0)),
                                               nn.Conv2d(node_feature_dim, hidden_dim, kernel_size=(kernel_size, 1)))
        if interact_type == 'mlp':                                               
            Interaction_nn = Interaction_MLP4d_layer
        elif interact_type == 'mamba':                              
            Interaction_nn = Interaction_MM4d_layer
        elif interact_type == 'scm': 
            Interaction_nn = SCM_MM4d_layer
        else:
            raise KeyError
        for i in range(0, feature_layers):
            self.add_module("gcl_feature_%d" % i, Interaction_nn(
                node_num = node_num, 
                input_dim = hidden_dim, 
                feature_len = node_feature_len,
                output_dim =hidden_dim, 
                hidden_dim=hidden_dim,
                act_fn=act_fn, 
                edge_types=edge_types, 
                skip_first_edge_type=skip_first_edge_type, 
                use_layer_norm = use_layer_norm, 
                use_edge_feature = use_edge_feature,
                attention=attention, 
                drop_rate=drop_rate))

        self.feature_out = nn.Sequential(
                MLP4d(node_num=node_num, n_in=hidden_dim*feature_layers, feature_len = node_feature_len, n_hid=out_node_feature_dim, 
                                 n_out=out_node_feature_dim, act_fn=act_fn, conv_kernel=(1,1), use_layer_norm=use_layer_norm, dropout=drop_rate),
                )
        
        self.fc_out = nn.Linear(node_feature_len, out_node_feature_len)

    def forward(self, x, g, nodes_mask, edges_mask):
        # B,D,N,L
        x = self.embedding_feature(x)
        feature_list = [] 
        if nodes_mask is not None:
            x = torch.einsum('abcd,ab->abcd', x.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)

        if self.use_edge_feature:
            f_e = torch.zeros([x.shape[2]*(x.shape[2]-1), x.shape[0], x.shape[3], x.shape[1]], dtype=torch.float).to(x.device) #[E, B, L, D]
        else:
            f_e = None
        for i in range(0, self.feature_layers):
            x, f_e = self._modules["gcl_feature_%d" % i](x, f_e, 
                                                        nodes_mask=nodes_mask, edges_mask=edges_mask, real_edges=g)
            feature_list.append(x)
        
        x = torch.cat(feature_list, dim=1).permute(2,0,3,1) # N,B,L,D

        out = self.fc_out(self.feature_out(x))
        out = out.permute(0,2,1,3)  # # B,N,D,P

        if nodes_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out, nodes_mask) 

        return out


def get_model(args, dataset_info):
    histogram = dataset_info['subsets_nodes_vs_samples'] 
    nodes_dist = DistributionNodes(histogram)

    gsl_act_fn = get_act_fn(args.gsl_act_fn)
    encoder_act_fn = get_act_fn(args.encoder_act_fn)
    pred_decoder_act_fn = get_act_fn(args.pred_decoder_act_fn)

    device = args.device
    node_num = args.max_node_num
    edge_types = args.edge_types
    kl_weight = args.kl_weight
    temp = args.temp
    hard = args.hard
    sib_k = args.sib_k
    use_graph_prior = args.use_graph_prior

    gsl_module_type = args.gsl_module_type
    gsl_dropout = args.gsl_dropout
    gsl_feature_len = args.gsl_feature_len
    gsl_feature_dim = args.gsl_feature_dim
    gsl_feature_hidden_dim = args.gsl_feature_hidden_dim
    gsl_description_hidden_dim = args.gsl_description_hidden_dim
    gsl_description_dim = args.gsl_description_dim
    gsl_description_len = args.gsl_description_len
    gsl_skip_first = args.gsl_skip_first
    gsl_feature_layers = args.gsl_feature_layers #
    gsl_description_layers = args.gsl_description_layers
    gsl_attention = args.gsl_attention
    gsl_edge_fusion_methad = args.gsl_edge_fusion_methad
    gsl_use_layer_norm = args.gsl_use_layer_norm
    gsl_use_description_learner = args.gsl_use_description_learner
    gsl_interact_type = args.gsl_interact_type
    gsl_use_prior_learner = args.gsl_use_prior_learner

    encoder_feature_len = args.encoder_feature_len
    encoder_feature_dim = args.encoder_feature_dim
    encoder_hidden_dim = args.encoder_hidden_dim
    encoder_feature_layer = args.encoder_feature_layer
    encoder_use_layer_norm = args.encoder_use_layer_norm
    encoder_layer_type = args.encoder_layer_type
    encoder_out_dim = args.encoder_out_dim
    encoder_out_len = args.encoder_out_len
    encoder_dropout = args.encoder_dropout

    pred_decoder_type = args.pred_decoder_type
    pred_decoder_dropout = args.pred_decoder_dropout
    pred_decoder_feature_len = encoder_out_len
    pred_decoder_predict_len = args.pred_decoder_predict_len
    pred_decoder_feature_dim = encoder_out_dim
    pred_decoder_predict_dim = args.pred_decoder_predict_dim
    pred_decoder_feature_hidden_dim = args.pred_decoder_feature_hidden_dim
    pred_decoder_feature_layers = args.pred_decoder_feature_layers 
    pred_decoder_attention = args.pred_decoder_attention
    pred_decoder_use_layer_norm = args.pred_decoder_use_layer_norm
    pred_decoder_interact_type = args.pred_decoder_interact_type
    pred_decoder_skip_first = args.pred_decoder_skip_first
    pred_decoder_use_edge_feature = args.pred_decoder_use_edge_feature


    ###### GSL ######
    if gsl_module_type == 'mlp':
        graph_learner = GSL_mlp(node_num=node_num, 
                                node_feature_dim=gsl_feature_dim,
                                node_feature_len=gsl_feature_len, 
                                node_description_dim=gsl_description_dim, 
                                node_description_len=gsl_description_len,
                                node_feature_hidden_dim=gsl_feature_hidden_dim, 
                                node_description_hidden_dim = gsl_description_hidden_dim,
                                edge_output_dim=edge_types, 
                                concat_method=gsl_edge_fusion_methad, 
                                dropout=gsl_dropout, 
                                act_fn=gsl_act_fn, 
                                device=device)
    elif gsl_module_type == 'cnn':
        graph_learner = GSL_cnn(node_num=node_num, 
                                node_feature_dim=gsl_feature_dim,
                                node_feature_len=gsl_feature_len, 
                                node_description_dim=gsl_description_dim, 
                                node_description_len=gsl_description_len,
                                node_feature_hidden_dim=gsl_feature_hidden_dim, 
                                node_description_hidden_dim = gsl_description_hidden_dim,
                                output_dim=edge_types, 
                                concat_method=gsl_edge_fusion_methad, 
                                dropout=gsl_dropout, 
                                act_fn=gsl_act_fn)
    elif gsl_module_type == 'gim':
        graph_learner = GSL_gim(node_num=node_num, 
                                node_feature_dim=gsl_feature_dim, 
                                node_feature_len=gsl_feature_len,
                                node_prior_len = pred_decoder_predict_len,
                                node_description_dim=gsl_description_dim,
                                node_description_len=gsl_description_len,
                                edge_types=edge_types, 
                                hidden_dim=gsl_feature_hidden_dim, 
                                skip_first_edge_type=gsl_skip_first,
                                interact_type=gsl_interact_type,
                                edge_concat_methad=gsl_edge_fusion_methad, 
                                act_fn=gsl_act_fn, 
                                feature_layers=gsl_feature_layers, 
                                description_layers=gsl_description_layers,
                                attention=gsl_attention,
                                out_edge_types=edge_types,
                                use_layer_norm=gsl_use_layer_norm, 
                                use_description_learner=gsl_use_description_learner, 
                                use_prior_learner=gsl_use_prior_learner,
                                drop_rate=gsl_dropout,
                                device=device)

    else:
        raise ValueError('Incorrect GSL type!')

    ###### node encoder ######
    encoder = Node_encoder(node_num=node_num, 
                           node_feature_len=encoder_feature_len, 
                           node_feature_dim=encoder_feature_dim, 
                           hidden_dim=encoder_hidden_dim, 
                           feature_layer=encoder_feature_layer,
                           layer_type=encoder_layer_type,
                           out_dim=encoder_out_dim,
                           out_len=encoder_out_len,
                           act_fn=encoder_act_fn, 
                           use_layer_norm=encoder_use_layer_norm, 
                           drop_rate=encoder_dropout
                            )

    ##### pred decoder ########
    if pred_decoder_type == 'mlp':
        pred_decoder = MLP_decoder(node_num=node_num, 
                                node_feature_dim=pred_decoder_feature_dim, 
                                edge_types=edge_types,
                                node_feature_len=pred_decoder_feature_len,
                                node_feature_hidden_dim=pred_decoder_feature_hidden_dim, 
                                output_dim=pred_decoder_predict_dim, 
                                prediction_horizon=pred_decoder_predict_len,
                                skip_first=pred_decoder_skip_first,
                                dropout=pred_decoder_dropout, 
                                act_fn=pred_decoder_act_fn,
                                device=device)
    elif pred_decoder_type == 'rnn':
        pred_decoder = RNN_decoder(node_num=node_num, 
                                n_in_node=pred_decoder_feature_dim, 
                                edge_types=edge_types, 
                                msg_hid=pred_decoder_feature_hidden_dim, 
                                msg_out=pred_decoder_feature_hidden_dim, 
                                n_hid=pred_decoder_feature_hidden_dim, 
                                pred_len=pred_decoder_predict_len,
                                skip_first=pred_decoder_skip_first)     
    elif pred_decoder_type == 'gim':
        pred_decoder = GIM_decoder(node_num=node_num, 
                                    node_feature_dim=pred_decoder_feature_dim, 
                                    node_feature_len=pred_decoder_feature_len,
                                    edge_types=edge_types, 
                                    out_node_feature_dim=pred_decoder_predict_dim, 
                                    out_node_feature_len=pred_decoder_predict_len,
                                    hidden_dim=pred_decoder_feature_hidden_dim, 
                                    skip_first_edge_type=pred_decoder_skip_first,
                                    interact_type=pred_decoder_interact_type,
                                    act_fn=pred_decoder_act_fn, 
                                    feature_layers=pred_decoder_feature_layers,
                                    attention=pred_decoder_attention,
                                    use_layer_norm=pred_decoder_use_layer_norm, 
                                    use_edge_feature=pred_decoder_use_edge_feature,
                                    drop_rate=pred_decoder_dropout,
                                    device=device)
    else:
        raise ValueError('Incorrect pred_decoder type!')

    GIM_model = GIM(graph_learner=graph_learner,
                    encoder=encoder,
                    pred_decoder=pred_decoder,
                    gsl_feature_len=gsl_feature_len,
                    gsl_feature_dim=gsl_feature_dim,
                    gsl_use_prior_learner=gsl_use_prior_learner,
                    encoder_feature_len=encoder_feature_len,
                    encoder_feature_dim=encoder_feature_dim,
                    pred_decoder_predict_len=pred_decoder_predict_len,
                    pred_decoder_predict_dim=pred_decoder_predict_dim,
                    use_graph_prior=use_graph_prior, 
                    kl_weight=kl_weight,
                    temp = temp,
                    hard = hard,
                    sib_k = sib_k,
                    device=device
                    )

    return GIM_model, nodes_dist
