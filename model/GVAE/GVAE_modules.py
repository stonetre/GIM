import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from mamba_ssm import Mamba
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from model.GIM.GIM_utils import *
from torch import Tensor
from torch_scatter import scatter
from itertools import permutations


_EPS = 1e-10

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # value_len, key_len, query_len = 1,1,1

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with each head (N, query_len, heads, head_dim) * (N, key_len, heads, head_dim) -> (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out).squeeze(1)
        return out



class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, dropout=0., act_fn=torch.nn.ELU()):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = dropout
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [batch, node, num_features]
        x = self.act_fn(self.fc1(inputs))
        x = self.dropout_fn(x)
        x = self.act_fn(self.fc2(x))
        return self.batch_norm(x)


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0., act_fn=torch.nn.ReLU()):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = dropout
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = self.act_fn(self.conv1(inputs))
        x = self.bn1(x)
        x = self.dropout_fn(x)
        x = self.pool(x)
        x = self.act_fn(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob



class GSL_mlp(nn.Module):
    def __init__(self, node_num, node_feature_dim, node_feature_len, node_description_dim, 
                 node_feature_hidden_dim, node_description_hidden_dim, node_description_len, 
                 edge_output_dim, concat_method, dropout=0., factor=True, 
                 act_fn=torch.nn.ReLU(), device='cpu'):
        super(GSL_mlp, self).__init__()

        self.factor = factor
        self.concat_method = concat_method
        # full graph construction
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)
        self.act_fn = act_fn

        # node feature encoder
        self.mlp1 = MLP(node_feature_dim*node_feature_len, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        self.mlp2 = MLP(node_feature_hidden_dim * 2, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        self.mlp3 = MLP(node_feature_hidden_dim, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        if self.factor:
            self.mlp4 = MLP(node_feature_hidden_dim * 3, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(node_feature_hidden_dim * 2, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
            print("Using MLP encoder.")

        # node description encoder
        self.mlp5 = MLP(node_description_dim*node_description_len, node_description_hidden_dim, node_description_hidden_dim, dropout, act_fn)
        self.mlp6 = MLP(node_description_hidden_dim*2, node_description_hidden_dim, node_description_hidden_dim, dropout, act_fn)
        self.mlp7 = MLP(node_description_hidden_dim, node_description_hidden_dim, node_description_hidden_dim, dropout, act_fn)
        if self.factor:
            self.mlp8 = MLP(node_description_hidden_dim * 3, node_description_hidden_dim, node_description_hidden_dim, dropout, act_fn)
        else:
            self.mlp8 = MLP(node_description_hidden_dim * 2, node_description_hidden_dim, node_description_hidden_dim, dropout, act_fn)
        
        # concat 
        if concat_method == 'cat':
            self.fc_graph = nn.Linear(node_feature_hidden_dim + node_description_hidden_dim, node_feature_hidden_dim)
        elif concat_method == 'mean':
            self.fc_graph = nn.Linear(node_feature_hidden_dim, node_feature_hidden_dim)
        elif concat_method == 'max':
            self.fc_graph = nn.Linear(node_feature_hidden_dim, node_feature_hidden_dim)
        elif concat_method == 'min':
            self.fc_graph = nn.Linear(node_feature_hidden_dim, node_feature_hidden_dim)
        elif concat_method == 'attention':
            self.fc_graph = nn.Linear(node_feature_hidden_dim, node_feature_hidden_dim)
        else:
            raise NotImplementedError
        
        self.fc_out = nn.Linear(node_feature_hidden_dim, edge_output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(self.rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, x, h, t_nodes, t_edges, nodes_mask, edges_mask):

        x = x.reshape(x.size(0), x.size(1), -1).to(torch.float32)
        # now x has shape: [batch, nodes, num_timesteps * num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        z_x = x
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        if self.factor:
            x = self.edge2node(x)
            x = self.mlp3(x)
            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        z_g = self.fc_out(x)
        return z_g


class GSL_cnn(nn.Module):
    """
    Note: This code is derived from NRI (Neural Relational Inference). https://github.com/ethanfetaya/NRI
    
    """
    def __init__(self, node_num, node_feature_dim, node_feature_len, node_description_dim, 
                 node_feature_hidden_dim, node_description_hidden_dim, node_description_len, 
                 output_dim, concat_method, dropout=0., factor=True, act_fn=torch.nn.ReLU()):
        super(GSL_cnn, self).__init__()

        self.dropout_prob = dropout
        self.factor = factor
        self.concat_method = concat_method
        # full graph construction
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

        self.cnn = CNN(node_feature_dim * 2, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        self.mlp1 = MLP(node_feature_hidden_dim, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        self.mlp2 = MLP(node_feature_hidden_dim, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        if self.factor:
            self.mlp3 = MLP(node_feature_hidden_dim * 3, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        else:
            self.mlp3 = MLP(node_feature_hidden_dim * 2, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
       
        self.fc_out = nn.Linear(node_feature_hidden_dim, output_dim)
        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        self.rel_rec = self.rel_rec.to(inputs.device)
        self.rel_send = self.rel_send.to(inputs.device)
        receivers = torch.matmul(self.rel_rec, x)
        receivers = receivers.view(inputs.size(0) * receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(self.rel_send, x)
        senders = senders.view(inputs.size(0) * senders.size(1),
                               inputs.size(2),
                               inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(self.rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, h, t_nodes, t_edges, nodes_mask, edges_mask):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.permute(0,1,3,2).to(torch.float32)
        edges = self.node2edge_temporal(x)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x)
            x = self.mlp2(x)

            x = self.node2edge(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)
        
        z_g = self.fc_out(x)

        return x_skip, z_g


class MLP_decoder(nn.Module):
    """MLP decoder module."""
# node_feature_dim=node_features_dim, edge_types=edge_types,
#                         node_feature_len=node_features_len, 
#                         node_feature_hidden_dim=node_feature_hidden_dim, 
#                         node_description_hidden_dim = node_description_hidden_dim,
#                         output_dim=encoder_output_dim,
#                         dropout=encoder_dropout
    def __init__(self, node_num, node_feature_dim, edge_types, node_feature_len, node_feature_hidden_dim, output_dim, 
                 prediction_horizon, dropout=0., skip_first=False, act_fn=torch.nn.ReLU(), device='cpu'):
        super(MLP_decoder, self).__init__()
        # edge mlp
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * node_feature_hidden_dim, node_feature_hidden_dim) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(node_feature_hidden_dim, node_feature_hidden_dim) for _ in range(edge_types)])
        self.skip_first_edge_type = skip_first
        # node mlp
        self.mlp1 = MLP(node_feature_dim*node_feature_len, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn)
        # output node mlp
        self.out_fc = nn.ModuleList([ MLP(node_feature_hidden_dim*2, node_feature_hidden_dim, node_feature_hidden_dim, dropout, act_fn) for _ in range(output_dim)])
        # output linear mlp
        self.output_fc = nn.ModuleList([nn.Linear(node_feature_hidden_dim, prediction_horizon) for _ in range(output_dim)])
        print('Using learned interaction net decoder.')

        self.dropout_prob = dropout

        # full graph construction
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)

    def edge2node(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(self.rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_type, t_nodes, t_edges, node_mask, edge_mask):
        # inputs has shape: [batch_size, node_num, encoder_output_dim]
        # rel_type has shape: [batch_size, num_atoms*num_atoms-num_atoms, num_edge_types]

        # sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
        #          rel_type.size(2)]
        # rel_type = rel_type.unsqueeze(1).expand(sizes)


        x = inputs.reshape(inputs.size(0), inputs.size(1), -1).to(torch.float32)
        # now x has shape: [batch, nodes, num_timesteps * num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = (x.permute(2,0,1) * node_mask).permute(1,2,0)
        z_x = x

        x = self.node2edge(x)
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        all_msgs = []
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](x))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = (msg.permute(2,0,1) * edge_mask).permute(1,2,0)
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs.append(msg)

        x = torch.stack(all_msgs).sum(dim=0)
        # Aggregate all msgs to receiver
        x = self.edge2node(x)
        x = torch.cat([x, z_x], dim=-1)
        out =[]
        for i in range(len(self.out_fc)):
            out_ = self.out_fc[i](x)
            out_ = self.output_fc[i](out_)
            out_ = (out_.permute(2,0,1) * node_mask).permute(1,2,0)
            out.append(out_)
        out = torch.stack(out).permute(1,2,0,3).squeeze(-1)
        return out


class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, node_num, output_len, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first
        self.output_len = output_len

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

                # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_type, hidden):

        # node2edge
        rel_rec = self.rel_rec.to(inputs.device)
        rel_send = self.rel_send.to(inputs.device)
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, t_nodes, t_edges, nodes_mask, edges_mask, mask,
                burn_in=True):

        inputs = data.permute(0,3,1,2).contiguous().to(torch.float32)

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) + self.output_len):

            if burn_in:
                if step <= time_steps-1:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            pred, hidden = self.single_step_forward(ins, rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.permute(0,2,3,1)[:,:,:,-self.output_len:].contiguous()

class GRUX(nn.Module):
    """
    GRU from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, dim_in: int, dim_hid: int, bias: bool=True):
        """
        Args:
            dim_in: input dimension
            dim_hid: dimension of hidden layers
            bias: adding a bias term or not, default: True
        """
        super(GRUX, self).__init__()
        self.hidden = nn.ModuleList([
            nn.Linear(dim_hid, dim_hid, bias)
            for _ in range(3)
        ])
        self.input = nn.ModuleList([
            nn.Linear(dim_in, dim_hid, bias)
            for _ in range(3)
        ])

    def forward(self, inputs: Tensor, hidden: Tensor, state: Tensor=None) -> Tensor:
        """
        Args:
            inputs: [..., dim]
            hidden: [..., dim]
            state: [..., dim], default: None
        """
        r = torch.sigmoid(self.input[0](inputs) + self.hidden[0](hidden))
        i = torch.sigmoid(self.input[1](inputs) + self.hidden[1](hidden))
        n = torch.tanh(self.input[2](inputs) + r * self.hidden[2](hidden))
        if state is None:
            state = hidden
        output = (1 - i) * n + i * state
        return output

class GNN(nn.Module):
    """
    Reimplementaion of the Message-Passing class in torch-geometric to allow more flexibility.
    """
    def __init__(self):
        super(GNN, self).__init__()
        
    def forward(self, *input):
        raise NotImplementedError

    def propagate(self, x: Tensor, es: Tensor, f_e: Tensor=None, agg: str='mean') -> Tensor:
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            x: [node, ..., dim], node embeddings 
        """
        msg, idx, size = self.message(x, es, f_e)
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

    def node2edge(self, x_i: Tensor, x_o: Tensor, f_e: Tensor) -> Tensor:
        """
        Args:
            x_i: [E, ..., dim], embeddings of incoming nodes
            x_o: [E, ..., dim], embeddings of outcoming nodes
            f_e: [E, ..., dim * 2], edge embeddings

        Return:
            edge embeddings
        """
        return torch.cat([x_i, x_o], dim=-1)

    def message(self, x: Tensor, es: Tensor, f_e: Tensor=None, option: str='o2i'):
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
        msg = self.node2edge(x_i, x_o, f_e)
        return msg, col, len(x)

    def update(self, x):
        return x

class RNN_decoder(GNN):
    """
    RNN decoder with spatio-temporal message passing mechanisms.
    """
    def __init__(self, node_num, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int, pred_len: int,
                 do_prob: float=0., skip_first: bool=False, option='both'):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
            option: default: 'both'
                'both': using both node-level and edge-level spatio-temporal message passing operations
                'node': using node-level the spatio-temporal message passing operation
                'edge': using edge-level the spatio-temporal message passing operation
        """
        super(RNN_decoder, self).__init__()
        self.option = option
        self.n_in_node = n_in_node
        self.pred_len = pred_len
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )
        self.gru_edge = GRUX(n_hid, n_hid)
        self.gru_node = GRUX(n_hid + n_in_node, n_hid + n_in_node)
        self.msg_out = msg_out
        self.skip_first = skip_first
        es = np.array(list(permutations(range(node_num), 2))).T
        self.es = torch.LongTensor(es)
        print('Using learned interaction net decoder.')

    def move(self, x: Tensor, es: Tensor, z: Tensor, h_node: Tensor=None, h_edge: Tensor=None):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]
            h_node: [node, batch, step, dim], hidden states of nodes, default: None
            h_edge: [E, batch, step, dim], hidden states of edges, default: None

        Return:
            x: [node, batch, step, dim], future node states
            msgs: [E, batch, step, dim], hidden states of edges
            cat: [node, batch, step, dim], hidden states of nodes
        """
        
        # z: [E, batch, K] -> [E, batch, step, K]
        # z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)
        if self.skip_first:
            norm -= 1
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))
        if h_edge is not None and self.option in {'edge', 'both'}:
            msgs = self.gru_edge(msgs, h_edge)
        # aggregate all msgs from the incoming edges
        msg = self.aggregate(msgs, col, size)
        # skip connection
        cat = torch.cat([x, msg], dim=-1)
        if h_node is not None and self.option in {'node', 'both'}:
            cat = self.gru_node(cat, h_node)
        delta = self.out(cat)
        if self.option == 'node':
            msgs = None
        if self.option == 'edge':
            cat = None
        return x + delta, cat, msgs

    def forward(self, x: Tensor, z: Tensor, nodes_mask, edges_mask, mask) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """
        es = self.es.to(x.device)
        M = 10
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        # x = x.permute(2, 0, 1, -1).contiguous()
        x = x.permute(1,0,3,2).contiguous().to(torch.float32)
        z = z.permute(1, 0, 2).contiguous()
        x_len = x.size(2)
        # assert (M <= x.shape[2])
        # # only take m-th timesteps as starting points (m: pred_steps)
        # x_m = x[:, :, 0::M, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict m steps.
        xs = []
        h_node, h_edge = None, None
        for i in range(0, x_len + self.pred_len):
            if i < x_len:
                x_in = x[:, :, i, :]
            else:
                x_in = x_m
            x_m, h_node, h_edge = self.move(x_in, es, z, h_node, h_edge)
            xs.append(x_m)

        node, batch, dim = xs[0].shape
        sizes = [node, batch, len(xs), dim]
        x_hat = Variable(torch.zeros(sizes))
        if x.is_cuda:
            x_hat = x_hat.cuda()
        # Re-assemble correct timeline
        for i in range(len(xs)):
            x_hat[:, :, i, :] = xs[i]
        x_hat = x_hat.permute(1, 0, 3, 2).contiguous()  # B,N,D,L
        return x_hat[:, :, :, -(self.pred_len):], x_hat, x_hat


# class RNNDEC(GNN):
#     """
#     RNN decoder with spatio-temporal message passing mechanisms.
#     """
#     def __init__(self, node_num, n_in_node: int, edge_types: int,
#                  msg_hid: int, msg_out: int, n_hid: int, pred_len: int,
#                  do_prob: float=0., skip_first: bool=False, option='both'):
#         """
#         Args:
#             n_in_node: input dimension
#             edge_types: number of edge types
#             msg_hid, msg_out, n_hid: dimension of different hidden layers
#             do_prob: rate of dropout, default: 0
#             skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
#             option: default: 'both'
#                 'both': using both node-level and edge-level spatio-temporal message passing operations
#                 'node': using node-level the spatio-temporal message passing operation
#                 'edge': using edge-level the spatio-temporal message passing operation
#         """
#         super(RNNDEC, self).__init__()
#         self.option = option
#         self.n_in_node = n_in_node
#         self.pred_len = pred_len
#         self.msgs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(2 * n_in_node, msg_hid),
#                 nn.ReLU(),
#                 nn.Dropout(do_prob),
#                 nn.Linear(msg_hid, msg_out),
#                 nn.ReLU(),
#             )
#             for _ in range(edge_types)
#         ])

#         self.out = nn.Sequential(
#             nn.Linear(n_in_node + msg_out, n_hid),
#             nn.ReLU(),
#             nn.Dropout(do_prob),
#             nn.Linear(n_hid, n_hid),
#             nn.ReLU(),
#             nn.Dropout(do_prob),
#             nn.Linear(n_hid, n_in_node)
#         )
#         self.gru_edge = GRUX(n_hid, n_hid)
#         self.gru_node = GRUX(n_hid + n_in_node, n_hid + n_in_node)
#         self.msg_out = msg_out
#         self.skip_first = skip_first
#         es = np.array(list(permutations(range(node_num), 2))).T
#         self.es = torch.LongTensor(es)
#         print('Using learned interaction net decoder.')

#     def move(self, x: Tensor, es: Tensor, z: Tensor, h_node: Tensor=None, h_edge: Tensor=None):
#         """
#         Args:
#             x: [node, batch, step, dim]
#             es: [2, E]
#             z: [E, batch, K]
#             h_node: [node, batch, step, dim], hidden states of nodes, default: None
#             h_edge: [E, batch, step, dim], hidden states of edges, default: None

#         Return:
#             x: [node, batch, step, dim], future node states
#             msgs: [E, batch, step, dim], hidden states of edges
#             cat: [node, batch, step, dim], hidden states of nodes
#         """
        
#         # z: [E, batch, K] -> [E, batch, step, K]
#         z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
#         msg, col, size = self.message(x, es)  # node to edge
#         idx = 1 if self.skip_first else 0
#         norm = len(self.msgs)
#         if self.skip_first:
#             norm -= 1
#         msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
#                    for i in range(idx, len(self.msgs)))
#         if h_edge is not None and self.option in {'edge', 'both'}:
#             msgs = self.gru_edge(msgs, h_edge)
#         # aggregate all msgs from the incoming edges
#         msg = self.aggregate(msgs, col, size)  # edge to node
#         # skip connection
#         cat = torch.cat([x, msg], dim=-1)
#         if h_node is not None and self.option in {'node', 'both'}:
#             cat = self.gru_node(cat, h_node)
#         delta = self.out(cat)
#         if self.option == 'node':
#             msgs = None
#         if self.option == 'edge':
#             cat = None
#         return x + delta, cat, msgs

#     def forward(self, x: Tensor, z: Tensor, t_nodes, t_edges, nodes_mask, edges_mask) -> Tensor:
#         """
#         Args:
#             x: [batch, step, node, dim], historical node states
#             z: [E, batch, K], distribution of edge types
#             es: [2, E], edge list
#             M: number of steps to predict

#         Return:
#             future node states
#         """
#         es = self.es.to(x.device)
#         M = 10
#         # x: [batch, step, node, dim] -> [node, batch, step, dim]
#         # x = x.permute(2, 0, 1, -1).contiguous()
#         x = x.permute(1,0,3,2).contiguous().to(torch.float32)
#         z = z.permute(1, 0, 2).contiguous()
#         assert (M <= x.shape[2])
#         # only take m-th timesteps as starting points (m: pred_steps)
#         x_m = x[:, :, 0::M, :]
#         # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
#         # predict m steps.
#         xs = []
#         h_node, h_edge = None, None
#         for _ in range(0, M):
#             x_m, h_node, h_edge = self.move(x_m, es, z, h_node, h_edge)
#             xs.append(x_m)
#         node, batch, skip, dim = xs[0].shape
#         sizes = [node, batch, skip * M, dim]
#         x_hat = Variable(torch.zeros(sizes))
#         if x.is_cuda:
#             x_hat = x_hat.cuda()
#         # Re-assemble correct timeline
#         for i in range(M):
#             x_hat[:, :, i::M, :] = xs[i]
#         x_hat = x_hat.permute(1, 0, 3, 2).contiguous()  # B,N,D,L
#         return x_hat[:, :, :, -(self.pred_len):]


class MLPs(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, node_num, n_in, n_hid, n_out, act_fn=torch.nn.ELU(), use_layer_norm=True, dropout=0.):
        super(MLPs, self).__init__()
        self.node_num = node_num
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.ln_node = nn.LayerNorm([node_num, n_hid])
        self.ln_edge = nn.LayerNorm([node_num*(node_num-1), n_hid])
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.use_layer_norm = use_layer_norm
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def layer_norm(self, inputs):
        if inputs.shape[1]==self.node_num:
            x = self.ln_node(inputs)
        else:
            x = self.ln_edge(inputs)
        return x
    
    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs, t=None):
        # Input shape: [batch, node, num_features]
        x = self.act_fn(self.fc1(inputs))
        x = self.dropout_fn(x)
        x = self.act_fn(self.fc2(x))
        if self.use_layer_norm:
            x = self.layer_norm(x)
        else:
            x = self.batch_norm(x)
        return x


class MLP4d(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, node_num, n_in, feature_len, n_hid, n_out, 
                 act_fn=torch.nn.ELU(), conv_kernel=(1,1), use_layer_norm=True, dropout=0.):
        super(MLP4d, self).__init__()
        self.node_num = node_num
        self.conv1 = nn.Conv2d(n_in, n_hid, kernel_size=conv_kernel, stride=1, padding=0)
        self.fc1 = nn.Linear(feature_len, feature_len)
        self.fc2 = nn.Linear(feature_len, feature_len)
        self.bn = nn.BatchNorm1d(feature_len)
        self.ln_node = nn.LayerNorm([node_num, feature_len])
        self.ln_edge = nn.LayerNorm([node_num*(node_num-1), feature_len])
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.use_layer_norm = use_layer_norm
    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal(m.weight.data)
    #             m.bias.data.fill_(0.1)
    
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
        # Input shape: [B, D, N/E, L]
        x = self.conv1(inputs)
        x = self.act_fn(self.fc1(x))
        x = self.dropout_fn(x)
        x = self.act_fn(self.fc2(x))
        if self.use_layer_norm:
            x = self.layer_norm(x)
        else:
            x = self.batch_norm(x)
        return x


class Casual_CNNs(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, node_num, n_in, feature_len, n_hid, n_out, 
                 act_fn=torch.nn.ELU(), conv_kernel=5, use_layer_norm=True, dropout=0.):
        super(Casual_CNNs, self).__init__()
        self.node_num = node_num
        self.conv1 = nn.Sequential( nn.ReplicationPad2d((conv_kernel-1,0,0,0)),
                                    nn.Conv2d(n_in, n_hid, kernel_size=(1,conv_kernel), stride=1),
                                    nn.BatchNorm2d(n_hid),
                                    act_fn)
        self.conv2 = nn.Sequential( nn.ReplicationPad2d((conv_kernel-1,0,0,0)),
                                    nn.Conv2d(n_hid, n_out, kernel_size=(1,conv_kernel), stride=1),
                                    nn.BatchNorm2d(n_out),
                                    act_fn)
        self.fc1 = nn.Linear(feature_len, feature_len)
        self.fc2 = nn.Linear(feature_len, feature_len)
        self.bn = nn.BatchNorm1d(feature_len)
        self.ln_node = nn.LayerNorm([node_num, feature_len])
        self.ln_edge = nn.LayerNorm([node_num*(node_num-1), feature_len])
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.use_layer_norm = use_layer_norm
    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal(m.weight.data)
    #             m.bias.data.fill_(0.1)

    
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
        # Input shape: [B, D, N/E, L]
        x = self.conv1(inputs)
        x = self.dropout_fn(x)
        x = self.conv2(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        else:
            x = self.batch_norm(x)
        return x


class MMs(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, node_num, feature_len, n_in, n_hid, n_out, act_fn=nn.SiLU(), 
                 mm_conv=2, use_layer_norm=True, dropout=0.):
        super(MMs, self).__init__()
        assert mm_conv %2 == 0
        self.node_num = node_num
        self.mamba = Mamba(d_model=n_in, d_state=n_hid, d_conv=mm_conv, expand=int(mm_conv/2))
        self.dim_layer = nn.Linear(n_in, n_out)
        self.ln_node = nn.LayerNorm([node_num, n_out])
        self.ln_edge = nn.LayerNorm([feature_len, n_out])
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.use_layer_norm = use_layer_norm

    
    def layer_norm(self, inputs):
        if inputs.shape[1]==self.node_num:
            x = self.ln_node(inputs)
        else:
            x = self.ln_edge(inputs)
        return x
    
    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs, t=None):
        # Input shape: [batch, node, num_features]
        x = self.act_fn(self.mamba(inputs))
        x = self.dropout_fn(x)
        x = self.dim_layer(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        else:
            x = self.batch_norm(x)
        return x

# class MMs4d(nn.Module):
#     """Two-layer fully-connected ELU net with batch norm."""

#     def __init__(self, node_num, n_in, feature_len, n_hid, n_out, act_fn=nn.SiLU(), 
#                  conv_kernel=2, use_layer_norm=True, dropout=0.):
#         super(MMs4d, self).__init__()
#         assert conv_kernel %2 == 0
#         self.node_num = node_num
#         # self.mamba = Mamba(d_model=n_in, d_state=n_hid, d_conv=conv_kernel, expand=int(conv_kernel/2))
#         self.mamba = Mamba(d_model=feature_len, d_state=feature_len, d_conv=conv_kernel, expand=int(conv_kernel/2))
#         self.dim_layer = nn.Linear(n_in, n_out)
#         self.ln = nn.LayerNorm([feature_len, n_out])
#         self.bn = nn.BatchNorm1d(n_out)
#         self.act_fn = act_fn
#         self.dropout_fn = nn.Dropout(p=dropout)
#         self.use_layer_norm = use_layer_norm

    
#     def batch_norm(self, inputs):
#         x = inputs.view(inputs.size(0) * inputs.size(1), -1)
#         x = self.bn(x)
#         return x.view(inputs.size(0), inputs.size(1), -1)

#     def forward(self, inputs, t=None):
#         # Input shape: [B, D, N/E, L]
#         # x = inputs.permute(0,2,3,1)
#         # x = x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3))
#         x = inputs.permute(0,2,1,3)  # B, N, D, L
#         x = x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3))
#         x = self.act_fn(self.dim_layer(x.permute(0,2,1))).permute(0,2,1) #[B*N, D, L]
#         x = self.dropout_fn(x)
#         x = self.act_fn(self.mamba(x)).permute(0,2,1) #[B*N, L, D]
#         # x = self.dim_layer(x) #[B*N, L, D]
#         if self.use_layer_norm:
#             x = self.ln(x)  # for L,D
#         else:
#             x = self.batch_norm(x) # for D
#         x = x.view(inputs.size(0), inputs.size(2), inputs.size(3), -1).permute(0,3,1,2)
#         return x


class MMs4d(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, node_num, n_in, feature_len, n_hid, n_out, act_fn=nn.SiLU(), 
                 conv_kernel=2, use_layer_norm=True, dropout=0.):
        super(MMs4d, self).__init__()
        self.node_num = node_num
        # self.mamba = Mamba(d_model=n_in, d_state=n_hid, d_conv=conv_kernel, expand=int(conv_kernel/2))
        self.dim_layer = nn.Linear(n_in, n_hid)
        self.mamba = Mamba(d_model=n_hid, d_state=n_hid*2, d_conv=conv_kernel, expand=1)
        self.dim_out = nn.Linear(n_hid, n_out)
        self.ln = nn.LayerNorm([feature_len, n_out])
        self.bn = nn.BatchNorm1d(n_out)
        self.act_fn = act_fn
        self.dropout_fn = nn.Dropout(p=dropout)
        self.use_layer_norm = use_layer_norm

        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.fill_(0.1)

    
    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs, t=None):
        # Input shape: [B, D, N/E, L]
        x = inputs.permute(0,2,3,1)
        x = x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3)) # B*N, L, D
        x = self.dim_layer(x) #[B*N, L, D]
        x = self.act_fn(self.mamba(x))
        x = self.dropout_fn(x)
        x = self.dim_out(x)
        if self.use_layer_norm:
            x = self.ln(x)  # for L,D
        else:
            x = self.batch_norm(x) # for D
        x = x.view(inputs.size(0), inputs.size(2), inputs.size(3), -1).permute(0,3,1,2)
        return x



class Interaction_mlp_layer(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, 
                 attention=False, drop_rate=0.):
        super(Interaction_mlp_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type
        # 先edge_mlp再node_mlp
        self.edge_nn = nn.ModuleList([MLPs(node_num, input_dim * 2, hidden_dim, 
                                           hidden_dim, act_fn, 
                                           use_layer_norm=use_layer_norm, 
                                           dropout=drop_rate) for _ in range(edge_types)])
        self.node_nn = MLPs(node_num, hidden_dim, hidden_dim, hidden_dim, 
                            act_fn, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.node_edge_nn = MLPs(node_num, input_dim + hidden_dim, hidden_dim, output_dim, 
                                 act_fn, use_layer_norm=use_layer_norm, dropout=drop_rate)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())
            
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # 列为发送节点
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)


    def edge_model(self, node_feature, t_edges, nodes_mask, edges_mask, real_edges = None):
        # full_edges: [2, full_num_edges], real_edges: [full_num_edges, edge_types]
        self.rel_rec = self.rel_rec.to(node_feature.device)
        self.rel_send =  self.rel_send.to(node_feature.device)
        receivers = torch.matmul(self.rel_rec, node_feature)
        senders = torch.matmul(self.rel_send, node_feature)
        edges_feature = torch.cat([senders, receivers], dim=2)
        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.edge_nn)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.edge_nn))
        all_msgs = []
        for i in range(start_idx, len(self.edge_nn)):
            msg = self.edge_nn[i](edges_feature, t_edges)
            msg = (msg.permute(2,0,1) * edges_mask).permute(1,2,0)
            if real_edges is not None:
                msg = msg * real_edges[:, :, i:i + 1]
            all_msgs.append(msg)
        mij = torch.stack(all_msgs).sum(dim=0)/norm

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edges_mask is not None:
            out = (out.permute(2,0,1) * edges_mask).permute(1,2,0)
            mij = (mij.permute(2,0,1) * edges_mask).permute(1,2,0)
        return out, mij

    def node_model(self, edge_feature, node_feature, t_nodes, nodes_mask, edges_mask):
        # x:[batch*node*node, 256]
        self.rel_rec = self.rel_rec.to(node_feature.device)
        incoming = torch.matmul(self.rel_rec.t(), edge_feature) # 必须是先edge_mlp再node_mlp
        incoming = incoming / incoming.size(1)
        incoming = self.node_nn(incoming, t_nodes)
        incoming = (incoming.permute(2,0,1) * nodes_mask).permute(1,2,0)
        node_edge_feature = torch.cat([node_feature, incoming], dim=2)
        out = self.node_edge_nn(node_edge_feature, t_nodes)
        # out = self.node_edge_mlp(incoming)
        if nodes_mask is not None:
            out = (out.permute(2,0,1) * nodes_mask).permute(1,2,0)
        return out

    def forward(self, node_feature, t_nodes, t_edges, nodes_mask, edges_mask, real_edges = None):
        edge_feature_att, edge_feature = self.edge_model(node_feature, t_edges, nodes_mask, edges_mask, real_edges)
        out_node = self.node_model(edge_feature_att, node_feature, t_nodes, nodes_mask, edges_mask)
        return out_node, edge_feature


class Interaction_MLP4d_layer(nn.Module):
    def __init__(self, node_num, input_dim, feature_len, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, 
                 attention=False, t_node_guide=False, t_edge_guide=False, drop_rate=0.):
        super(Interaction_MLP4d_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type
        self.t_node_guide = t_node_guide
        self.t_edge_guide = t_edge_guide
        # 先edge_cnn再node_cnn
        edge_dim = input_dim*2+1 if t_edge_guide else input_dim*2
        edge_nn = MLP4d(node_num=node_num, n_in=edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                      n_out=hidden_dim, act_fn=act_fn, conv_kernel=(1,1), use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.edge_block = nn.ModuleList([edge_nn for _ in range(edge_types)])

        node_edge_dim = hidden_dim*2+1 if t_node_guide else hidden_dim*2
        self.node_block = MLP4d(node_num=node_num, n_in=hidden_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                n_out=hidden_dim, act_fn=act_fn, conv_kernel=(1,1), use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.node_edge_block = MLP4d(node_num=node_num, n_in=node_edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                    n_out=output_dim, act_fn=act_fn, conv_kernel=(1,1), use_layer_norm=use_layer_norm, dropout=drop_rate)
        
        if self.attention:
            self.att_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)),
                nn.Sigmoid())
            
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # 列为发送节点
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

    def edge_model(self, node_feature, t_edges, nodes_mask, edges_mask, real_edges = None):
        # full_edges: [2, full_num_edges], real_edges: [full_num_edges, edge_types] 
        self.rel_rec = self.rel_rec.to(node_feature.device)
        self.rel_send =  self.rel_send.to(node_feature.device)
        receivers = torch.matmul(self.rel_rec, node_feature)
        senders = torch.matmul(self.rel_send, node_feature)
        
        if self.t_edge_guide:
            edges_feature = torch.cat([senders, receivers, t_edges], dim=1)  # [B, D*2+1, E, L]
        else:
            edges_feature = torch.cat([senders, receivers], dim=1)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        all_msgs = []
        for i in range(start_idx, len(self.edge_block)):
            msg = self.edge_block[i](edges_feature) # B, D, E, L
            msg = torch.einsum('abcd,ab->abcd', msg.permute(0,2,1,3), edges_mask)
            if real_edges is not None:
                msg = torch.einsum('abcd,ab->abcd', msg, real_edges[:, :, i:i + 1].squeeze(-1)).permute(0,2,1,3)
            else:
                msg = msg.permute(0,2,1,3)
            all_msgs.append(msg)
        mij = torch.stack(all_msgs).sum(dim=0)

        if self.attention:
            att_val = self.att_conv(mij)
            out = mij * att_val
        else:
            out = mij

        if edges_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), edges_mask).permute(0,2,1,3)
            mij = torch.einsum('abcd,ab->abcd', mij.permute(0,2,1,3), edges_mask).permute(0,2,1,3)
        return out, mij

    def node_model(self, edge_feature, node_feature, t_nodes, nodes_mask, edges_mask):
        # x:[batch*node*node, 256]
        self.rel_rec = self.rel_rec.to(edge_feature.device)
        incoming = torch.matmul(self.rel_rec.t(), edge_feature) # 必须是先edge_mlp再node_mlp
        incoming = self.node_block(incoming / incoming.size(2))
        if self.t_node_guide:
            node_edge_feature = torch.cat([node_feature, incoming, t_nodes], dim=1)
        else:
            node_edge_feature = torch.cat([node_feature, incoming], dim=1)
        out = self.node_edge_block(node_edge_feature)
        if nodes_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        return out

    def forward(self, node_feature, t_nodes, t_edges, nodes_mask, edges_mask, real_edges = None):
        edge_feature_att, edge_feature = self.edge_model(node_feature, t_edges, nodes_mask, edges_mask, real_edges)
        out_node = self.node_model(edge_feature_att, node_feature, t_nodes, nodes_mask, edges_mask)
        return out_node, edge_feature


class Interaction_casual_cnn_layer(nn.Module):
    def __init__(self, node_num, input_dim, feature_len, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, 
                 attention=False, t_node_guide=False, t_edge_guide=False, drop_rate=0.):
        super(Interaction_casual_cnn_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type
        self.t_node_guide = t_node_guide
        self.t_edge_guide = t_edge_guide
        # 先edge_cnn再node_cnn
        edge_dim = input_dim*2+1 if t_edge_guide else input_dim*2
        edge_nn = Casual_CNNs(node_num=node_num, n_in=edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                      n_out=hidden_dim, act_fn=act_fn, conv_kernel=5, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.edge_block = nn.ModuleList([edge_nn for _ in range(edge_types)])

        node_edge_dim = hidden_dim*2+1 if t_node_guide else hidden_dim*2
        self.node_block = Casual_CNNs(node_num=node_num, n_in=hidden_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                n_out=hidden_dim, act_fn=act_fn, conv_kernel=5, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.node_edge_block = Casual_CNNs(node_num=node_num, n_in=node_edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                    n_out=output_dim, act_fn=act_fn, conv_kernel=5, use_layer_norm=use_layer_norm, dropout=drop_rate)
        
        if self.attention:
            self.att_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)),
                nn.Sigmoid())
            
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # 列为发送节点
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

    def edge_model(self, node_feature, t_edges, nodes_mask, edges_mask, real_edges = None):
        # full_edges: [2, full_num_edges], real_edges: [full_num_edges, edge_types] 
        self.rel_rec = self.rel_rec.to(node_feature.device)
        self.rel_send =  self.rel_send.to(node_feature.device)
        receivers = torch.matmul(self.rel_rec, node_feature)
        senders = torch.matmul(self.rel_send, node_feature)
        
        if self.t_edge_guide:
            edges_feature = torch.cat([senders, receivers, t_edges], dim=1)  # [B, D*2+1, E, L]
        else:
            edges_feature = torch.cat([senders, receivers], dim=1)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        all_msgs = []
        for i in range(start_idx, len(self.edge_block)):
            msg = self.edge_block[i](edges_feature) # B, D, E, L
            msg = torch.einsum('abcd,ab->abcd', msg.permute(0,2,1,3), edges_mask)
            if real_edges is not None:
                msg = torch.einsum('abcd,ab->abcd', msg, real_edges[:, :, i:i + 1].squeeze(-1)).permute(0,2,1,3)
            else:
                msg = msg.permute(0,2,1,3)
            all_msgs.append(msg)
        mij = torch.stack(all_msgs).sum(dim=0)

        if self.attention:
            att_val = self.att_conv(mij)
            out = mij * att_val
        else:
            out = mij

        if edges_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), edges_mask).permute(0,2,1,3)
            mij = torch.einsum('abcd,ab->abcd', mij.permute(0,2,1,3), edges_mask).permute(0,2,1,3)
        return out, mij

    def node_model(self, edge_feature, node_feature, t_nodes, nodes_mask, edges_mask):
        # x:[batch*node*node, 256]
        self.rel_rec = self.rel_rec.to(edge_feature.device)
        incoming = torch.matmul(self.rel_rec.t(), edge_feature) # 必须是先edge_mlp再node_mlp
        incoming = self.node_block(incoming / incoming.size(2))
        if self.t_node_guide:
            node_edge_feature = torch.cat([node_feature, incoming, t_nodes], dim=1)
        else:
            node_edge_feature = torch.cat([node_feature, incoming], dim=1)
        out = self.node_edge_block(node_edge_feature)
        if nodes_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        return out

    def forward(self, node_feature, t_nodes, t_edges, nodes_mask, edges_mask, real_edges = None):
        edge_feature_att, edge_feature = self.edge_model(node_feature, t_edges, nodes_mask, edges_mask, real_edges)
        out_node = self.node_model(edge_feature_att, node_feature, t_nodes, nodes_mask, edges_mask)
        return out_node, edge_feature


class Interaction_MM_layer(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, 
                 attention=False, t_node_guide=False, t_edge_guide=False, drop_rate=0.):
        super(Interaction_MM_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type
        self.t_node_guide = t_node_guide
        self.t_edge_guide = t_edge_guide
        # 先edge_mlp再node_mlp
        edge_dim = 3 if t_edge_guide else 2
        edge_nn = MMs(node_num=node_num, feature_len=input_dim, n_in=edge_dim, n_hid=hidden_dim, 
                      n_out=1, act_fn=act_fn, mm_conv=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.edge_block = nn.ModuleList([edge_nn for _ in range(edge_types)])

        node_dim = 3 if t_node_guide else 2
        self.node_block = MMs(node_num=node_num, feature_len=input_dim, n_in=hidden_dim, n_hid=hidden_dim, 
                              n_out=hidden_dim, act_fn=act_fn, mm_conv=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.node_edge_block = MMs(node_num=node_num, feature_len=input_dim, n_in=node_dim, n_hid=hidden_dim, 
                                    n_out=1, act_fn=act_fn, mm_conv=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())
            
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # 列为发送节点
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

    def edge_model(self, node_feature, t_edges, nodes_mask, edges_mask, real_edges = None):
        # full_edges: [2, full_num_edges], real_edges: [full_num_edges, edge_types] 
        self.rel_rec = self.rel_rec.to(node_feature.device)
        self.rel_send =  self.rel_send.to(node_feature.device)
        receivers = torch.matmul(self.rel_rec, node_feature)
        senders = torch.matmul(self.rel_send, node_feature)
        bs, n = receivers.shape[0], receivers.shape[1]
        receivers = receivers.view(bs*n,-1).contiguous().unsqueeze(2)
        senders = senders.view(bs*n,-1).contiguous().unsqueeze(2)
        
        if self.t_edge_guide:
            t_edges = t_edges.view(bs*n,-1).contiguous().unsqueeze(2)
            edges_feature = torch.cat([senders, receivers, t_edges], dim=2)  # [B*E, L, C=3]
        else:
            edges_feature = torch.cat([senders, receivers], dim=2)
        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        all_msgs = []

        for i in range(start_idx, len(self.edge_block)):
            msg = self.edge_block[i](edges_feature)
            msg = msg.squeeze(-1).view(bs,n,-1).contiguous()
            msg = (msg.permute(2,0,1) * edges_mask).permute(1,2,0)
            if real_edges is not None:
                msg = msg * real_edges[:, :, i:i + 1]
            all_msgs.append(msg)
        mij = torch.stack(all_msgs).sum(dim=0)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edges_mask is not None:
            out = (out.permute(2,0,1) * edges_mask).permute(1,2,0)
            mij = (mij.permute(2,0,1) * edges_mask).permute(1,2,0)
        return out, mij

    def node_model(self, edge_feature, node_feature, t_nodes, nodes_mask, edges_mask):
        # x:[batch*node*node, 256]
        self.rel_rec = self.rel_rec.to(edge_feature.device)
        incoming = torch.matmul(self.rel_rec.t(), edge_feature) # 必须是先edge_mlp再node_mlp
        bs, n = incoming.shape[0], incoming.shape[1]
        incoming = self.node_block(incoming / incoming.size(1))
        if self.t_node_guide:
            node_edge_feature = torch.cat([node_feature.unsqueeze(3), incoming.unsqueeze(3), t_nodes.unsqueeze(3)], dim=3)
        else:
            node_edge_feature = torch.cat([node_feature.unsqueeze(3), incoming.unsqueeze(3)], dim=3)
        node_edge_feature = node_edge_feature.view(bs*n,node_edge_feature.shape[2], -1).contiguous()
        out = self.node_edge_block(node_edge_feature)
        out = out.squeeze(-1).view(bs,n,-1).contiguous()
        if nodes_mask is not None:
            out = (out.permute(2,0,1) * nodes_mask).permute(1,2,0)
        return out

    def forward(self, node_feature, t_nodes, t_edges, nodes_mask, edges_mask, real_edges = None):
        edge_feature_att, edge_feature = self.edge_model(node_feature, t_edges, nodes_mask, edges_mask, real_edges)
        out_node = self.node_model(edge_feature_att, node_feature, t_nodes, nodes_mask, edges_mask)
        return out_node, edge_feature


class Interaction_MM4d_layer(nn.Module):
    def __init__(self, node_num, input_dim, feature_len, output_dim, hidden_dim,  
                 act_fn=nn.SiLU(), edge_types=2, 
                 skip_first_edge_type=True, use_layer_norm=False, 
                 attention=False, t_node_guide=False, t_edge_guide=False, drop_rate=0.):
        super(Interaction_MM4d_layer, self).__init__()
        self.attention = attention
        self.skip_first_edge_type = skip_first_edge_type
        self.t_node_guide = t_node_guide
        self.t_edge_guide = t_edge_guide
        # 先edge_cnn再node_cnn
        edge_dim = input_dim*2+1 if t_edge_guide else input_dim*2
        edge_nn = MMs4d(node_num=node_num, n_in=edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                      n_out=hidden_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.edge_block = nn.ModuleList([edge_nn for _ in range(edge_types)])

        node_edge_dim = hidden_dim*2+1 if t_node_guide else hidden_dim*2
        self.node_block = MMs4d(node_num=node_num, n_in=hidden_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                n_out=hidden_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        self.node_edge_block = MMs4d(node_num=node_num, n_in=node_edge_dim, feature_len=feature_len, n_hid=hidden_dim, 
                                    n_out=output_dim, act_fn=act_fn, conv_kernel=2, use_layer_norm=use_layer_norm, dropout=drop_rate)
        
        if self.attention:
            self.att_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, 1, kernel_size=(1,1)),
                nn.Sigmoid())
            
        # Generate off-diagonal interaction graph
        off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)  # 列为发送节点
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

    def edge_model(self, node_feature, t_edges, nodes_mask, edges_mask, real_edges = None):
        # full_edges: [2, full_num_edges], real_edges: [full_num_edges, edge_types] 
        self.rel_rec = self.rel_rec.to(node_feature.device)
        self.rel_send =  self.rel_send.to(node_feature.device)
        receivers = torch.matmul(self.rel_rec, node_feature)
        senders = torch.matmul(self.rel_send, node_feature)
        
        if self.t_edge_guide:
            edges_feature = torch.cat([senders, receivers, t_edges], dim=1)  # [B, D*2+1, E, L]
        else:
            edges_feature = torch.cat([senders, receivers], dim=1)

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        all_msgs = []
        for i in range(start_idx, len(self.edge_block)):
            msg = self.edge_block[i](edges_feature) # B, D, E, L
            msg = torch.einsum('abcd,ab->abcd', msg.permute(0,2,1,3), edges_mask)
            if real_edges is not None:
                msg = torch.einsum('abcd,ab->abcd', msg, real_edges[:, :, i:i + 1].squeeze(-1)).permute(0,2,1,3)
            else:
                msg = msg.permute(0,2,1,3)
            all_msgs.append(msg)
        mij = all_msgs[0]
        if len(all_msgs)>1:
            for i in range(1,len(self.edge_block)):
                mij = mij + all_msgs[i]
        mij = mij/len(self.edge_block)

        if self.attention:
            att_val = self.att_conv(mij)
            out = mij * att_val
        else:
            out = mij

        if edges_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), edges_mask).permute(0,2,1,3)
            mij = torch.einsum('abcd,ab->abcd', mij.permute(0,2,1,3), edges_mask).permute(0,2,1,3)
        return out, mij

    def node_model(self, edge_feature, node_feature, t_nodes, nodes_mask, edges_mask):
        # x:[batch*node*node, 256]
        self.rel_rec = self.rel_rec.to(edge_feature.device)
        incoming = torch.matmul(self.rel_rec.t(), edge_feature) # 必须是先edge_mlp再node_mlp
        incoming = self.node_block(incoming / incoming.size(2))
        if self.t_node_guide:
            node_edge_feature = torch.cat([node_feature, incoming, t_nodes], dim=1)
        else:
            node_edge_feature = torch.cat([node_feature, incoming], dim=1)
        out = self.node_edge_block(node_edge_feature)
        if nodes_mask is not None:
            out = torch.einsum('abcd,ab->abcd', out.permute(0,2,1,3), nodes_mask).permute(0,2,1,3)
        return out

    def forward(self, node_feature, t_nodes, t_edges, nodes_mask, edges_mask, real_edges = None):
        edge_feature_att, edge_feature = self.edge_model(node_feature, t_edges, nodes_mask, edges_mask, real_edges)
        out_node = self.node_model(edge_feature_att, node_feature, t_nodes, nodes_mask, edges_mask)
        return out_node, edge_feature



class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        if isinstance(histogram, np.ndarray):
            histogram = histogram.item()
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs