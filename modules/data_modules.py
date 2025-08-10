import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from os.path import join as join
from torch.utils.data import Dataset
from modules.data_utils import Preprocess_fn
from modules.datafile_preprocess import prepare_dataset


class ProcessedDataset(Dataset):

    def __init__(self, data, sample_normalize=False, normalize_method= 'std'):

        self.data = data
        self.normalize_method = normalize_method
        if sample_normalize:
            self.normalize_nodes_timeseries()

    def normalize_nodes_timeseries(self):
        # Normalize `node_properties` while retaining the normalization parameters for each sample and each time series.
        if self.normalize_method == 'std':
            nodes_timeseries = self.data['nodes_timeseries']
            # nodes_timeseries shape is [sample_num, slide_window_size, node_num]
            # normalize along the second axis
            nodes_timeseries_mean = torch.mean(nodes_timeseries, axis=1)
            nodes_timeseries_std = torch.std(nodes_timeseries, axis=1)
            self.data['mean_or_min'] = nodes_timeseries_mean
            self.data['std_or_max'] = nodes_timeseries_std
            nodes_timeseries_std[nodes_timeseries_std == 0] = 1
            nodes_timeseries = (nodes_timeseries - nodes_timeseries_mean.unsqueeze(1)) / nodes_timeseries_std.unsqueeze(1)
            self.data['nodes_timeseries'] = nodes_timeseries
            self.data['target'] = (self.data['target'] - nodes_timeseries_mean.unsqueeze(1)) / nodes_timeseries_std.unsqueeze(1)   
        elif self.normalize_method == 'max':
            nodes_timeseries = self.data['nodes_timeseries']
            nodes_timeseries_max = torch.max(nodes_timeseries, axis=1)
            nodes_timeseries_min = torch.min(nodes_timeseries, axis=1)
            self.data['mean_or_min'] = nodes_timeseries_min
            self.data['std_or_max'] = nodes_timeseries_max
            nodes_timeseries_max[nodes_timeseries_max == 0] = 1
            nodes_timeseries = (nodes_timeseries - nodes_timeseries_min.unsqueeze(1)) / nodes_timeseries_max.unsqueeze(1)
            self.data['nodes_timeseries'] = nodes_timeseries
            self.data['target'] = (self.data['target'] - nodes_timeseries_min.unsqueeze(1)) / nodes_timeseries_max.unsqueeze(1)            
        elif self.normalize_method == 'min_max':
            nodes_timeseries = self.data['nodes_timeseries']
            nodes_timeseries_max = torch.max(nodes_timeseries, axis=1)
            nodes_timeseries_min = torch.min(nodes_timeseries, axis=1)
            self.data['mean_or_min'] = nodes_timeseries_min
            self.data['std_or_max'] = nodes_timeseries_max
            nodes_timeseries_max[nodes_timeseries_max == 0] = 1
            nodes_timeseries = (nodes_timeseries - nodes_timeseries_min.unsqueeze(1)) / (nodes_timeseries_max.unsqueeze(1) - nodes_timeseries_min.unsqueeze(1))
            self.data['nodes_timeseries'] = nodes_timeseries
            self.data['target'] = (self.data['target'] - nodes_timeseries_min.unsqueeze(1)) / (nodes_timeseries_max.unsqueeze(1) - nodes_timeseries_min.unsqueeze(1))
            
        else:
            raise ValueError('Incorrect normalize method! Must chose std/max/min_max!')


    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.data['nodes_timeseries'].shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}


def initialize_datasets(args, force_download=False):

    datadir = args.datadir
    datasets = args.datasets                            # multi dataset, list type. such as ['swat','wadi']
    file_format = args.file_format                      # must according to dataset, such as ['csv', 'npy']
    has_graph = args.has_graph                          # must according to dataset, such as [True, False]
    has_node_description = args.has_node_description    # must according to dataset, such as [True, False]
    slide_window_size = args.encoder_feature_len
    label_window_size = args.pred_decoder_predict_len
    slide_window_step = args.slide_window_step    
    time_downsample = args.timeseries_downsample  
    max_node_num = args.max_node_num 
    get_graph_method = args.get_graph_method 

    if datasets in [['sim_springs10'], ['sim_springs5'], ['sim_charges5'], ['sim_kuramoto5']]:
        train_percent = 5/7
        valid_percent = 1/7
        test_percent = 1-train_percent-valid_percent 
    else:
        train_percent = args.train_percent
        valid_percent = args.valid_percent
        test_percent = args.test_percent   

    # Download and process dataset. Returns datafiles.
    all_trainset=[]
    all_validset=[]
    all_testset=[]
    dataset_info = {
        'subsets_num': len(datasets),
        'subsets_samples_num': [],
        'subsets_nodes_num': [],
        'subsets_edges_num': [],
        'subsets_nodes_vs_samples': {}
    }

    for i, dataset_name in enumerate(datasets):
        print('Processing dataset: ', dataset_name)
        datafiles = prepare_dataset(datadir, dataset_name, file_format=file_format[i], 
                                    has_graph=has_graph[i], get_graph_method=get_graph_method, 
                                    has_node_description= has_node_description[i], 
                                    slide_window_size=slide_window_size, label_window_size = label_window_size,
                                    slide_window_step=slide_window_step, time_downsample = time_downsample,
                                    max_node=max_node_num, force_download=force_download)
        node_num = int(sum(datafiles['node_mask_1d'][0]))
        dataset_info['subsets_samples_num'].append(datafiles['sample_num'])
        if node_num not in dataset_info['subsets_nodes_num']:
            dataset_info['subsets_nodes_vs_samples'][node_num] = datafiles['sample_num']
        else:
            dataset_info['subsets_nodes_vs_samples'][node_num] += datafiles['sample_num']
        dataset_info['subsets_nodes_num'].append(node_num)
        print('Dataset: {} samples:{} nodes:{}'.format(dataset_name, datafiles['sample_num'], node_num))
        # 划分数据集
        train_num = int(datafiles['sample_num']*train_percent)
        valid_num = int(datafiles['sample_num']*valid_percent)

        train_set = {}
        valid_set = {}
        test_set = {}
        for key in datafiles.keys():
            if key in ['nodes_timeseries', 'target', 'label', 'node_mask_1d', 'edge_mask_1d','real_adj_2d','real_adj_1d','node_description_embed','node_types']:
                train_set[key] = datafiles[key][:train_num]
                valid_set[key] = datafiles[key][train_num:train_num+valid_num]
                test_set[key] = datafiles[key][train_num+valid_num:]
        all_trainset.append(train_set)
        all_validset.append(valid_set)
        all_testset.append(test_set)
        # # 使用pickle分别保存
        # with open(datadir + '/' + dataset_name +'/' + 'train.pkl', 'wb') as f:
        #     pickle.dump(train_set, f)
        # with open(datadir + '/' + dataset_name  +'/' + 'valid.pkl', 'wb') as f:
        #     pickle.dump(valid_set, f)
        # with open(datadir + '/' + dataset_name  +'/' + 'test.pkl', 'wb') as f:
        #     pickle.dump(test_set, f)
        print('Dataset: {} trainset:{} validset:{} testset:{}'.format(dataset_name, len(train_set['nodes_timeseries']), len(valid_set['nodes_timeseries']), len(test_set['nodes_timeseries'])))


    # 合并训练集、验证集、测试集
    train_dataset = {}
    valid_dataset = {}
    test_dataset = {}
    for train_set in all_trainset:
        for key, val in train_set.items():
            if key in train_dataset:
                train_dataset[key] = np.concatenate((train_dataset[key], val), axis=0)
            else:
                train_dataset[key] = val
    for valid_set in all_validset:
        for key, val in valid_set.items():
            if key in valid_dataset:
                valid_dataset[key] = np.concatenate((valid_dataset[key], val), axis=0)
            else:
                valid_dataset[key] = val
    for test_set in all_testset:
        for key, val in test_set.items():
            if key in test_dataset:
                test_dataset[key] = np.concatenate((test_dataset[key], val), axis=0)
            else:
                test_dataset[key] = val

    # 保存数据集
    np.savez(join(datadir, 'train_dataset.npz'), **train_dataset)
    np.savez(join(datadir, 'valid_dataset.npz'), **valid_dataset)
    np.savez(join(datadir, 'test_dataset.npz'), **test_dataset)
    np.savez(join(datadir, 'dataset_info.npz'), **dataset_info)

    datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # 归一化、可迭代
    datasets = {split: ProcessedDataset(data) for split, data in datasets.items()}

    return datasets, dataset_info

def load_dataset(datadir):
    train_dataset = np.load(datadir +'/train_dataset.npz')
    valid_dataset = np.load(datadir +'/valid_dataset.npz')
    test_dataset = np.load(datadir +'/test_dataset.npz')
    dataset_info = np.load(datadir +'/dataset_info.npz', allow_pickle=True)
    print('train set length:', len(train_dataset['nodes_timeseries']))
    print('valid set length:', len(valid_dataset['nodes_timeseries']))
    print('test set length:', len(test_dataset['nodes_timeseries']))
    train_set = {}
    valid_set = {}
    test_set = {}
    data_info = {}
    for key in train_dataset.keys():
        train_set[key] = torch.from_numpy(train_dataset[key])
    for key in valid_dataset.keys():
        valid_set[key] = torch.from_numpy(valid_dataset[key])
    for key in test_dataset.keys():
        test_set[key] = torch.from_numpy(test_dataset[key])
    for key in dataset_info.keys():
        data_info[key] = dataset_info[key]
    datasets = {'train': train_set, 'valid': valid_set, 'test': test_set}
    datasets = {split: ProcessedDataset(data) for split, data in datasets.items()}
    return datasets, data_info


def retrieve_dataloaders(args, load=True, force_download=False):
    # Retrieve dataloaders
    batch_size = args.batch_size
    num_workers = args.num_workers
    input_sequential = args.input_sequential
    if load:
        datasets, dataset_info = load_dataset(args.datadir)
    else:
        datasets, dataset_info = initialize_datasets(args, force_download=force_download)
    
    # datasets是一个字典，key为train/valid/test，value为ProcessedDataset对象
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train') and not input_sequential
        # dataset 为ProcessedDataset对象，包含迭代器，迭代器返回一个字典，字典的key为node_properties等，value为tensor
        # 因此DataLoader的collate_fn函数需要对字典的value进行处理
        preprocess = Preprocess_fn()
        dataloaders[split] = DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        drop_last=True,
                                        collate_fn=preprocess.collate_fn)
    return dataloaders, dataset_info

 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='time series data process')
    parser.add_argument('--train_percent', type=float, default=0.7)
    parser.add_argument('--valid_percent', type=float, default=0.2)
    parser.add_argument('--test_percent', type=float, default=0.1)
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--datasets', type=list, default=['swat'])
    parser.add_argument('--file_format', type=list, default=['csv'])
    parser.add_argument('--has_graph', type=list, default=[True])
    parser.add_argument('--has_node_description', type=list, default=[True])
    parser.add_argument('--slide_window_size', type=int, default=12)
    parser.add_argument('--label_window_size', type=int, default=1)
    parser.add_argument('--slide_window_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_sequential', type=bool, default=False)
    args = parser.parse_args()

    dataloaders = retrieve_dataloaders(args, force_download=False)
    data_dummy = next(iter(dataloaders['train']))


    print(data_dummy['nodes_timeseries'].shape)
