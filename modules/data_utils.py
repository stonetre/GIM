import torch
import numpy as np
from sklearn import metrics
import pandas as pd
from numpy.random import multivariate_normal as mnorm
import copent
from xgboost import XGBRegressor
from torch_geometric.data import Data
import torch_geometric as pyg

def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


class Preprocess_fn:
    def __init__(self):
        pass
    
    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : 是一个列表，每个元素是一个字典，每个字典代表一个分子
            The collated data.
        """
        # # 将一个batch按属性进行拼接
        # batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
        # assert batch['nodes_timeseries'].shape[2] == self.max_node_num 
        
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
        
        # pyg_datas= []
        # for sample in batch:
        #     sample['edge_index'] = sample['edge_index'].T
        #     sample['nodes_timeseries'] = sample['nodes_timeseries'].T
        #     sample['target'] = sample['target'].T
        #     sample['node_description_embed'] = sample['node_description_embed'].T
        #     sample['full_edge_index'] = sample['full_edge_index'].T
        #     pyg_data = Data(**sample)
        #     pyg_datas.append(pyg_data)

        # # pyg_datas = Data(**batch)
        # batched_data = pyg.data.Batch.from_data_list(pyg_datas)

        return batch


def edges_to_adj_matrix(edges, num_nodes):
    """将边索引矩阵转换为邻接矩阵。

    参数:
    edges -- 边的列表，每个边是一对顶点索引。
    num_nodes -- 图中真实顶点的数量。

    返回值:
    邻接矩阵，是一个二维numpy数组。
    """
    
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        i, j = edge
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # 如果是无向图，添加这行代码

    return adj_matrix

# simple graph
def adj_matrix_to_edge_onehot(adj_matrix, max_node):
    # adj_matrix 最大支持4维矩阵。每一个维度定义为[batch, node, node, edge_type]
    # 数据格式支持numpy和torch
    assert adj_matrix.shape[-2] == adj_matrix.shape[-1] 
    
    if isinstance(adj_matrix, np.ndarray):
        num_classes = int(np.max(adj_matrix) + 1)
        if num_classes<2:
            num_classes = 2
        if adj_matrix.ndim == 2: # 只有一个图
            _, nodes = adj_matrix.shape
            if nodes > max_node:
                max_node_num = nodes
            else:
                max_node_num = max_node
            non_diagonal_elements = adj_matrix[np.logical_not(np.eye(len(adj_matrix), dtype=bool))]
            adj_matrix_onehot = np.eye(num_classes)[non_diagonal_elements]
        elif adj_matrix.ndim == 3:  #每个batch一个图
            bs, nodes, _ = adj_matrix.shape
            if nodes > max_node:
                max_node_num = nodes
            else:
                max_node_num = max_node
            max_edge = max_node_num * max_node_num - max_node_num
            adj_matrix_onehot = np.zeros((bs, max_edge, 2))
            for i in range(bs):
                adj = adj_matrix[i,:,:]
                non_diagonal_elements = adj[np.logical_not(np.eye(len(adj), dtype=bool))]
                adj_matrix_onehot[i] = np.eye(num_classes)[non_diagonal_elements]

        else:
            raise ValueError('adj_matrix should be a 2D, 3D or 4D numpy array.')
        
    elif isinstance(adj_matrix, torch.Tensor):
        num_classes = torch.max(adj_matrix) + 1
        if adj_matrix.ndim == 2: # 只有一个图
            _, nodes = adj_matrix.shape
            if nodes > max_node:
                max_node_num = nodes
            else:
                max_node_num = max_node
            non_diagonal_elements = adj_matrix[torch.logical_not(torch.eye(len(adj_matrix), dtype=bool))]
            adj_matrix_onehot = torch.eye(num_classes)[non_diagonal_elements]
        elif adj_matrix.ndim == 3:  #每个batch一个图
            bs, nodes, _ = adj_matrix.shape
            if nodes > max_node:
                max_node_num = nodes
            else:
                max_node_num = max_node
            max_edge = max_node_num * max_node_num - max_node_num
            adj_matrix_onehot = torch.zeros((bs, max_edge, 2))
            for i in range(bs):
                adj = adj_matrix[i,:,:]
                non_diagonal_elements = adj[torch.logical_not(torch.eye(len(adj), dtype=bool))]
                adj_matrix_onehot[i] = torch.eye(num_classes)[non_diagonal_elements]

        else:
            raise ValueError('adj_matrix should be a 2D, 3D or 4D numpy array.')
        
    else:
        raise ValueError('adj_matrix should be a numpy array or a torch tensor.')
    
    return adj_matrix_onehot



# def adj_matrix_to_edge_list(adj_matrix):
#     edge_list = []
#     num_nodes = adj_matrix.shape[0]
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):  # 只遍历上三角来避免重复
#             if adj_matrix[i, j] != 0:  # 存在边
#                 edge_list.append([i, j])
#     return np.array(edge_list)

def adj_matrix_to_edge_list(adj_matrix):
    edge_list = []
    rows, cols = np.where(adj_matrix != 0)
    for row, col in zip(rows, cols):
        edge_list.append([row, col])
    return edge_list

def NMI_matrix(df, cut_off=0.5):  # 计算标准化互信息矩阵
    if df.columns[0] == 'DateTime': 
        df = df.loc[:, df.columns != 'DateTime'] 
    number = df.columns.size  # 获取df的列数
    List = []
    Name = []
    for n in range(number):
        Name.append(df.columns[n])  # 获取dataframe的索引
    for i in range(number):
        A = []
        X = df[df.columns[i]]  # df.columns[i]获取对应列的索引，df['索引']获取对应列的数值
        for j in range(number):
            Y = df[df.columns[j]]
            A.append(metrics.normalized_mutual_info_score(X, Y))  # 计算标准化互信息
        List.append(A)  # List是列表格式
    NMI_matrix = np.array(List)
    NMI_matrix = np.where(abs(NMI_matrix) < cut_off, 0, 1)
    for i in range(NMI_matrix.shape[0]):
        NMI_matrix[i, i] = 0
    edge_index = adj_matrix_to_edge_list(NMI_matrix)
    return NMI_matrix, edge_index

def pearson_corr_matrix(data, cut_off=0.5):
    if data.columns[0] == 'DateTime': 
        data = data.loc[:, data.columns != 'DateTime'] 
    PC_matrix = data.corr(method='pearson')
    PC_matrix = np.array((PC_matrix > cut_off).astype(int))
    np.fill_diagonal(PC_matrix, 0)
    edge_index = adj_matrix_to_edge_list(PC_matrix)
    return PC_matrix, edge_index

def copula_entropy_matrix(df, cut_off=0.5):
    if df.columns[0] == 'DateTime': 
        df = df.loc[:, df.columns != 'DateTime'] 
    number = df.columns.size  # 获取df的列数
    List = []
    Name = []
    for n in range(number):
        Name.append(df.columns[n])  # 获取dataframe的索引
    for i in range(number):
        A = []
        X = df[df.columns[i]]  # df.columns[i]获取对应列的索引，df['索引']获取对应列的数值
        for j in range(number):
            Y = df[df.columns[j]]
            value = pd.concat([X,Y],axis=1)
            A.append(copent.copent(value))  # 计算Copula互信息
            print('Copula_entropy: node {} to node {} .'.format(i,j))
        List.append(A)  # List是列表格式
    Copula_matrix = np.array(List)
    Copula_matrix = np.where(abs(Copula_matrix) < cut_off, 0, 1)
    for i in range(Copula_matrix.shape[0]):
        Copula_matrix[i, i] = 0
    edge_index = adj_matrix_to_edge_list(Copula_matrix)
    return Copula_matrix, edge_index

def transfer_entropy_matrix(df, max_lag=10, cut_off=0.5):
    number = df.columns.size  # 获取df的列数
    List = []
    Name = []
    for n in range(number):
        Name.append(df.columns[n])  # 获取dataframe的索引
    for i in range(number):
        A = []
        X = df[df.columns[i]]  # df.columns[i]获取对应列的索引，df['索引']获取对应列的数值
        for j in range(number):
            te = np.zeros(max_lag)
            Y = df[df.columns[j]]
            for lag in range(1,max_lag+1):
                te[lag-1] = copent.transent(X,Y,lag)
            A.append(np.max(te))  # 计算Copula互信息
            print('Transfer_entropy: node {} to node {} max lag {}'.format(i,j,np.where(te==np.max(te))))
        List.append(A)  # List是列表格式
    TE_matrix = np.array(List)
    TE_matrix = np.where(abs(TE_matrix) < cut_off, 0, 1)
    for i in range(TE_matrix.shape[0]):
        TE_matrix[i, i] = 0
    edge_index = adj_matrix_to_edge_list(TE_matrix)
    return TE_matrix, edge_index

def xgboost_score_matrix(df, cut_off=0.5):
    if df.columns[0] == 'DateTime': 
        df = df.loc[:, df.columns != 'DateTime'] 
    number = df.columns.size  # 获取df的列数
    List = []
    Name = []
    model = XGBRegressor(max_depth=200, learning_rate=0.15, n_estimators=100)
    for n in range(number):
        Name.append(df.columns[n])  # 获取dataframe的索引
    for i in range(number):
        A = []
        X = df.drop(Name[i],axis=1).values
        Y = df[df.columns[i]]
        model.fit(X,Y)
        k = list(model.feature_importances_)
        k.insert(i,1)
        List.append(k)  # List是列表格式
    XGB_matrix = np.array(List)
    XGB_matrix = np.where(abs(XGB_matrix) < cut_off, 0, 1)
    for i in range(XGB_matrix.shape[0]):
        XGB_matrix[i, i] = 0
    edge_index = adj_matrix_to_edge_list(XGB_matrix)
    return XGB_matrix, edge_index
