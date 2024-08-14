import os
import logging
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from modules.data_utils import pearson_corr_matrix, NMI_matrix, copula_entropy_matrix, transfer_entropy_matrix, xgboost_score_matrix
from modules.data_utils import adj_matrix_to_edge_list
from os.path import join as join

def normalize_timeseries(data, method='standard', save_params=False):

    epsion = 1e-8
    params = {}
    if method not in ['standard', 'minmax']:
        raise ValueError("method musk is 'standard' or 'minmax'")
    
    if method == 'standard':
        # 计算均值和标准差
        mean = data.mean()
        std = data.std()
        
        # 标准化
        normalized_data = (data - mean) / (std + epsion)
        
        if save_params:
            params['mean'] = mean
            params['std'] = std
    else:  # method == 'minmax'
        # 计算最大值和最小值
        min_val = data.min()
        max_val = data.max()
        
        # 最大最小化归一化
        normalized_data = (data - min_val) / (max_val - min_val)
        
        if save_params:
            params['min'] = min_val
            params['max'] = max_val
    
    if save_params:
        return normalized_data, params
    else:
        return normalized_data


def prepare_dataset(datadir, dataset, file_format='csv', has_graph=True, get_graph_method='copula',
                    has_node_description=True, time_downsample=1, slide_window_size=12, label_window_size=1,
                    slide_window_step=1, max_node=15, force_download=False):

    dataset_dir = [datadir, dataset]
    file_names = get_file_names(dataset, file_format, has_graph, has_node_description)

    datafiles = get_datafiles(dataset_dir, file_names, file_format)
    check_datafiles_exist(datafiles)

    new_download = check_dataset_availability(datafiles, force_download)
    if new_download or force_download:
        download_dataset(datadir, dataset)

    output = process_data(datadir, dataset, file_format, time_downsample, slide_window_size, label_window_size,
                          slide_window_step, has_graph, get_graph_method, has_node_description, max_node)
    return output

def get_file_names(dataset, file_format, has_graph, has_node_description):
    if file_format == 'csv':
        file_names = ['node_properties']
        if has_node_description:
            file_names.append('node_description')
        if has_graph:
            file_names.append('graph')
    if file_format == 'npy':
        file_names = get_npy_file_names(dataset)
    return file_names

def get_npy_file_names(dataset):
    if dataset == 'sim_springs10':
        return ['edges_test_springs10', 'edges_train_springs10', 'edges_valid_springs10',
                'loc_test_springs10', 'loc_train_springs10', 'loc_valid_springs10',
                'vel_test_springs10', 'vel_train_springs10', 'vel_valid_springs10']
    elif dataset == 'sim_springs5':
        return ['edges_test_springs5', 'edges_train_springs5', 'edges_valid_springs5',
                'loc_test_springs5', 'loc_train_springs5', 'loc_valid_springs5',
                'vel_test_springs5', 'vel_train_springs5', 'vel_valid_springs5']
    elif dataset == 'sim_kuramoto5':
        return ['edges_test_kuramoto5', 'edges_train_kuramoto5', 'edges_valid_kuramoto5',
                'loc_test_kuramoto5', 'loc_train_kuramoto5', 'loc_valid_kuramoto5',
                'vel_test_kuramoto5', 'vel_train_kuramoto5', 'vel_valid_kuramoto5']
    else:
        raise ValueError('Unsupported dataset for npy format!')

def get_datafiles(dataset_dir, file_names, file_format):
    return {name: os.path.join(*(dataset_dir + [name + f'.{file_format}'])) for name in file_names}

def check_datafiles_exist(datafiles):
    datafiles_checks = [os.path.exists(datafile) for datafile in datafiles.values()]
    if not all(datafiles_checks):
        raise ValueError('Dataset only partially processed. Try deleting and running again to download/process.')

def check_dataset_availability(datafiles, force_download):
    datafiles_checks = [os.path.exists(datafile) for datafile in datafiles.values()]
    if all(datafiles_checks):
        logging.info('Dataset exists and is processed.')
        return False
    elif any([not x for x in datafiles_checks]):
        logging.info('Dataset does not exist. Downloading!')
        return True
    return force_download

def download_dataset(datadir, dataset):
    if not os.path.exists(f'{datadir}/{dataset}'):
        os.makedirs(f'{datadir}/{dataset}')
    if dataset == 'swat':
        logging.info('Beginning download of %s dataset!', dataset)
        tar_data = os.path.join(datadir, dataset, 'swat.zip')
        url_data = 'https://figshare.com/ndownloader/files/44809780'
        urllib.request.urlretrieve(url_data, filename=tar_data)
        logging.info('%s dataset downloaded successfully!', dataset)
        with zipfile.ZipFile(tar_data, 'r') as zip_ref:
            zip_ref.extractall(datadir)
    else:
        logging.info('Dataset exists.')

def process_data(datadir, dataset, file_format, time_downsample, slide_window_size, label_window_size,
                 slide_window_step, has_graph, get_graph_method, has_node_description, max_node):
    if file_format == 'csv':
        return process_csv_data(datadir, dataset, time_downsample, slide_window_size, label_window_size,
                                slide_window_step, has_graph, get_graph_method, has_node_description, max_node)
    elif file_format == 'npy':
        return process_npy_data(datadir, dataset, slide_window_size, label_window_size, slide_window_step, max_node)
    raise ValueError('Incorrect file format! Must choose csv/npy!')

def process_csv_data(datadir, dataset, time_downsample, slide_window_size, label_window_size, slide_window_step,
                     has_graph, get_graph_method, has_node_description, max_node):
    """
    Process the csv data file. The csv file should be timeseries data. 
    The first column of the csv file must be DataTime. 
    Check if a specific name appears in the last column. 
    If the name is any of 'Label', 'Output', 'target', it is considered a soft measurement task. 
    If 'OT' appears, delete it.
    """
    output = {}
    df = pd.read_csv(os.path.join(datadir, dataset, 'node_properties.csv'), index_col=False, parse_dates=['DateTime'])
    df = df.iloc[:, 1:].astype('float32').iloc[::time_downsample, :]
    print('Downsample over, Data shape:', df.shape)
    df.ffill().bfill()
    if df.isnull().to_numpy().sum():
        raise ValueError('Check the data file!')
    
    # Check if the last column is a label for soft measurement tasks
    if df.columns[-1] in ['Label', 'Output', 'target']:
        label = df.iloc[:, -1]
        df = df.iloc[:, :-1]
    elif df.columns[-1] in ['OT']:
        df = df.iloc[:, :-1]
        label = np.zeros((df.shape[0], 1))
    else:
        label = np.zeros((df.shape[0], 1))

    df = normalize_timeseries(df, method='standard')
    real_node_num = df.shape[1]
    nodes_name = df.columns.values.tolist()
    label = np.array([label[i:i + label_window_size] for i in range(0, label.shape[0] - label_window_size + 1, slide_window_step)])
    output['label'] = label
    if df.shape[1] < max_node:
        df = pd.concat([df, pd.DataFrame(np.zeros((df.shape[0], max_node - df.shape[1])))], axis=1)
    nodes_timeseries = np.array(df)
    nodes_timeseries_samples, target = create_sliding_windows(nodes_timeseries, slide_window_size, label_window_size, slide_window_step)
    output['target'] = np.expand_dims(target, 2)
    output['nodes_timeseries'] = np.expand_dims(nodes_timeseries_samples, 2)
    output['sample_num'] = nodes_timeseries_samples.shape[0]

    if has_graph:
        output.update(process_graph_data(datadir, dataset, real_node_num, max_node, output['sample_num']))
    else:
        output.update(create_graph(df, get_graph_method, real_node_num, max_node, output['sample_num']))

    if has_node_description:
        output.update(process_node_description(datadir, dataset, nodes_name, max_node, output['sample_num']))
    else:
        output.update(np.zeros((output['sample_num'], 512, 1, max_node)))
        output['node_types'] = ['default']
    
    return output

def process_npy_data(datadir, dataset, slide_window_size, label_window_size, slide_window_step, max_node):
    output = {}
    loc_train, vel_train, edges_train = load_npy_files(datadir, dataset)
    nodes_timeseries = np.concatenate((loc_train, vel_train), axis=2)
    nodes_timeseries, label = create_sliding_windows(nodes_timeseries, slide_window_size, label_window_size, slide_window_step)
    nodes_timeseries = nodes_timeseries.squeeze(0) # [Batch, Time, Dim, Node]
    label = label.squeeze(0) # [Batch, Time, Dim, Node]
    real_node_num = nodes_timeseries.shape[3]
    sample_num = nodes_timeseries.shape[0]

    if nodes_timeseries.shape[3] < max_node:
        nodes_timeseries = np.concatenate((nodes_timeseries, np.zeros((nodes_timeseries.shape[0], nodes_timeseries.shape[1], nodes_timeseries.shape[2], max_node - nodes_timeseries.shape[3]))), axis=3)
        label = np.concatenate((label, np.zeros((label.shape[0], label.shape[1], label.shape[2], max_node - label.shape[3]))), axis=3)
    output['target'] = label  
    output['label'] = label 
    output['nodes_timeseries'] = nodes_timeseries 
    output['sample_num'] = nodes_timeseries.shape[0]
    edges_adj = edges_train
    edges_index = [adj_matrix_to_edge_list(edges_adj[i]) for i in range(edges_adj.shape[0])]

    non_diagonal_mask = ~np.eye(edges_adj.shape[1], dtype=bool)
    edge_attr = [edges_adj[i][non_diagonal_mask] for i in range(edges_adj.shape[0])]
    output['real_edge_index'] = edges_index.copy()
    output['real_adj_1d'] = np.array(edge_attr)

    node_mask = np.ones(max_node)
    node_mask[real_node_num:] = 0
    node_mask = np.tile(node_mask, (sample_num, 1))
    edge_mask = np.ones((max_node, max_node))
    edge_mask[real_node_num:, :] = 0
    edge_mask[:, real_node_num:] = 0
    np.fill_diagonal(edge_mask, 0)
    edge_mask = edge_mask[non_diagonal_mask]
    edge_mask = np.tile(edge_mask, (sample_num, 1))
    output['node_mask_1d'] = node_mask
    output['edge_mask_1d'] = edge_mask
    output['node_description_embed'] = np.zeros((nodes_timeseries.shape[0], 512, nodes_timeseries.shape[2], nodes_timeseries.shape[3]))
    output['node_types'] = np.zeros((nodes_timeseries.shape[0], 512, nodes_timeseries.shape[2], nodes_timeseries.shape[3]))
    return output

def load_npy_files(datadir, dataset):
    if dataset == 'sim_springs10':
        loc_train = np.load(os.path.join(datadir, dataset, 'loc_train_springs10.npy'))
        vel_train = np.load(os.path.join(datadir, dataset, 'vel_train_springs10.npy'))
        edges_train = np.load(os.path.join(datadir, dataset, 'edges_train_springs10.npy'))
        loc_valid = np.load(os.path.join(datadir, dataset, 'loc_valid_springs10.npy'))
        vel_valid = np.load(os.path.join(datadir, dataset, 'vel_valid_springs10.npy'))
        edges_valid = np.load(os.path.join(datadir, dataset, 'edges_valid_springs10.npy'))
        loc_test = np.load(os.path.join(datadir, dataset, 'loc_test_springs10.npy'))
        vel_test = np.load(os.path.join(datadir, dataset, 'vel_test_springs10.npy'))
        edges_test = np.load(os.path.join(datadir, dataset, 'edges_test_springs10.npy'))
    elif dataset == 'sim_springs5':
        loc_train = np.load(os.path.join(datadir, dataset, 'loc_train_springs5.npy'))
        vel_train = np.load(os.path.join(datadir, dataset, 'vel_train_springs5.npy'))
        edges_train = np.load(os.path.join(datadir, dataset, 'edges_train_springs5.npy'))
        loc_valid = np.load(os.path.join(datadir, dataset, 'loc_valid_springs5.npy'))
        vel_valid = np.load(os.path.join(datadir, dataset, 'vel_valid_springs5.npy'))
        edges_valid = np.load(os.path.join(datadir, dataset, 'edges_valid_springs5.npy'))
        loc_test = np.load(os.path.join(datadir, dataset, 'loc_test_springs5.npy'))
        vel_test = np.load(os.path.join(datadir, dataset, 'vel_test_springs5.npy'))
        edges_test = np.load(os.path.join(datadir, dataset, 'edges_test_springs5.npy'))
    elif dataset == 'sim_kuramoto5':
        loc_train = np.load(os.path.join(datadir, dataset, 'loc_train_kuramoto5.npy'))
        vel_train = np.load(os.path.join(datadir, dataset, 'vel_train_kuramoto5.npy'))
        edges_train = np.load(os.path.join(datadir, dataset, 'edges_train_kuramoto5.npy'))
        loc_valid = np.load(os.path.join(datadir, dataset, 'loc_valid_kuramoto5.npy'))
        vel_valid = np.load(os.path.join(datadir, dataset, 'vel_valid_kuramoto5.npy'))
        edges_valid = np.load(os.path.join(datadir, dataset, 'edges_valid_kuramoto5.npy'))
        loc_test = np.load(os.path.join(datadir, dataset, 'loc_test_kuramoto5.npy'))
        vel_test = np.load(os.path.join(datadir, dataset, 'vel_test_kuramoto5.npy'))
        edges_test = np.load(os.path.join(datadir, dataset, 'edges_test_kuramoto5.npy'))
    else:
        raise ValueError('Unsupported dataset for npy format!')

    loc_train = np.concatenate((loc_train, loc_valid, loc_test), axis=0)
    vel_train = np.concatenate((vel_train, vel_valid, vel_test), axis=0)
    edges_train = np.concatenate((edges_train, edges_valid, edges_test), axis=0)
    # Normalize to [-1, 1]
    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1
    return loc_train, vel_train, edges_train

def create_sliding_windows(df, slide_window_size, label_window_size, slide_window_step):
    nodes_timeseries = np.array(df)
    if nodes_timeseries.ndim ==4:
        label = np.array([nodes_timeseries[:,i + slide_window_size:i + slide_window_size + label_window_size, :, :]
                      for i in range(0, nodes_timeseries.shape[1] - slide_window_size - label_window_size + 1, slide_window_step)])
        nodes_timeseries_samples = np.array([nodes_timeseries[:, i:i + slide_window_size, :, :]
                                         for i in range(0, nodes_timeseries.shape[1] - slide_window_size - label_window_size + 1, slide_window_step)])
    else:
        label = np.array([nodes_timeseries[i + slide_window_size:i + slide_window_size + label_window_size, :]
                        for i in range(0, nodes_timeseries.shape[0] - slide_window_size - label_window_size + 1, slide_window_step)])
        nodes_timeseries_samples = np.array([nodes_timeseries[i:i + slide_window_size, :]
                                            for i in range(0, nodes_timeseries.shape[0] - slide_window_size - label_window_size + 1, slide_window_step)])
    return nodes_timeseries_samples, label

def process_graph_data(datadir, dataset, real_node_num, max_node, sample_num):
    output = {}
    graph = pd.read_csv(os.path.join(datadir, dataset, 'graph.csv'), header=None)
    graph_index = np.array(graph.iloc[:, 0:2])
    weights = np.array(graph.iloc[:, 2])
    adj = np.zeros((max_node, max_node))
    for i, weight in enumerate(weights):
        adj[graph_index[i, 0], graph_index[i, 1]] = weight
    non_diagonal_mask = ~np.eye(max_node, dtype=bool)
    real_adj_1d = adj[non_diagonal_mask]
    real_edge_index = np.tile(graph_index, (sample_num, 1, 1))
    real_weighted_adj = np.tile(adj, (sample_num, 1))
    real_adj_1d = np.tile(real_adj_1d, (sample_num, 1))
    # mask node
    node_mask = np.ones(max_node)
    node_mask[real_node_num:] = 0
    node_mask = np.tile(node_mask, (sample_num, 1))
    # mask edge
    edge_mask = np.ones((max_node, max_node))
    edge_mask[real_node_num:, :] = 0
    edge_mask[:, real_node_num:] = 0
    np.fill_diagonal(edge_mask, 0)
    edge_mask = edge_mask[non_diagonal_mask]
    edge_mask = np.tile(edge_mask, (sample_num, 1))
    output['node_mask_1d'] = node_mask
    output['edge_mask_1d'] = edge_mask
    output['real_edge_index'] = real_edge_index
    output['real_adj_2d'] = real_weighted_adj
    output['real_adj_1d'] = real_adj_1d
    return output

def create_graph(nodes_timeseries, get_graph_method, real_node_num, max_node, sample_num):
    output = {}
    if nodes_timeseries.shape[0] > 2000:
        input_adj = nodes_timeseries[:2000]
    else:
        input_adj = nodes_timeseries
    if get_graph_method == 'corr':
        adj, graph_index = pearson_corr_matrix(input_adj)
    elif get_graph_method == 'nmi':
        adj, graph_index = NMI_matrix(input_adj)
    elif get_graph_method == 'copula':
        adj, graph_index = copula_entropy_matrix(input_adj)
    elif get_graph_method == 'te':
        adj, graph_index = transfer_entropy_matrix(input_adj)
    elif get_graph_method == 'xgb':
        adj, graph_index = xgboost_score_matrix(input_adj)
    else:
        raise ValueError('Incorrect get_graph_method! Must choose corr/nmi/copula/te/xgb!')
    print('Get Graph adj sucess:', adj)
    non_diagonal_mask = ~np.eye(max_node, dtype=bool)
    real_adj_1d = adj[non_diagonal_mask]
    real_edge_index = np.tile(graph_index, (sample_num, 1, 1))
    real_weighted_adj = np.tile(adj, (sample_num, 1))
    real_adj_1d = np.tile(real_adj_1d, (sample_num, 1))
    # mask node
    node_mask = np.ones(max_node)
    node_mask[real_node_num:] = 0
    node_mask = np.tile(node_mask, (sample_num, 1))
    # mask edge
    edge_mask = np.ones((max_node, max_node))
    edge_mask[real_node_num:, :] = 0
    edge_mask[:, real_node_num:] = 0
    np.fill_diagonal(edge_mask, 0)
    edge_mask = edge_mask[non_diagonal_mask]
    edge_mask = np.tile(edge_mask, (sample_num, 1))
    output['node_mask_1d'] = node_mask
    output['edge_mask_1d'] = edge_mask
    output['real_edge_index'] = real_edge_index
    output['real_adj_2d'] = real_weighted_adj
    output['real_adj_1d'] = real_adj_1d
    return output

def process_node_description(datadir, dataset, nodes_name, max_node, sample_num):
    print('Process node description.....')
    output = {}
    node_description = pd.read_csv(os.path.join(datadir, dataset, 'node_description.csv'), index_col=False)
    node_sort_list = [node_description[node_description.iloc[:, 0] == node_name].index[0] for node_name in nodes_name]
    node_description_df = node_description.iloc[node_sort_list, :]
    node_types = node_description_df.iloc[:,1].values.tolist()
    node_description_list = [' '.join(node_description_df.iloc[i, :].values.tolist()) for i in range(node_description_df.shape[0])]
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('modules/sentence_transformers/sentence-transformers_distiluse-base-multilingual-cased-v1')
    node_description_embed = embedder.encode(node_description_list, convert_to_tensor=True).cpu().numpy()
    node_types_embed = embedder.encode(node_types, convert_to_tensor=True).cpu().numpy()
    # 扩展到最大节点数
    node_description_embed = np.concatenate((node_description_embed, np.zeros((max_node - node_description_embed.shape[0], 512))), axis=0)
    node_types_embed = np.concatenate((node_types_embed, np.zeros((max_node - node_types_embed.shape[0], 512))), axis=0)
    node_description_embed = np.tile(node_description_embed, (sample_num, 1, 1)).transpose(0, 2, 1)
    node_types_embed = np.tile(node_types_embed, (sample_num, 1, 1)).transpose(0, 2, 1)
    node_description_embed = np.expand_dims(node_description_embed, 2) # [Batch, 512, 1, Node]
    node_types_embed = np.expand_dims(node_types_embed, 2) # [Batch, 512, 1, Node]
    output['node_description_embed'] = node_description_embed
    output['node_types'] = node_types_embed
    print('Process node description sucess!')
    return output


