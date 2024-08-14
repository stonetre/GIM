import argparse
from exp.exp_forcast import main
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
DATASETS: Specify all datasets for training. The names in the list need to be the same as the folder names under data.
FILEFORMAT: Specify the file format of the dataset. The length of the list should be the same as DATASETS.
DATA_WITH_GRAPH: Specify whether the dataset has a graph structure. The length of the list should be the same as DATASETS.
DATA_WITH_DESCRIP: Specify whether the dataset has node description. The length of the list should be the same as DATASETS.

Example:
# DATASETS = ['sim_springs5','swat2015']
# FILEFORMAT = ['npy','csv']
# DATA_WITH_GRAPH = [True, False]
# DATA_WITH_DESCRIP = [False, True]

If each dataset does not have a pre-specified reference graph structure, the data preprocessing process provides 
a variety of structure inference methods to pre-build a reference graph structure from the time series.
'''

DATASETS = ['sim_springs5']
FILEFORMAT = ['npy']
DATA_WITH_GRAPH = [True]
DATA_WITH_DESCRIP = [False]

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='sim_springs5', help='Experiment name')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--valid_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='How many batches to wait before logging.')
parser.add_argument('--edge_types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--lr_decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor')
parser.add_argument('--train_percent', type=float, default=0.7)
parser.add_argument('--valid_percent', type=float, default=0.1)
parser.add_argument('--test_percent', type=float, default=0.2)
parser.add_argument('--timeseries_downsample', type=int, default=1)
parser.add_argument('--datadir', type=str, default='data/sim/sim_springs5', help='')
parser.add_argument('--datasets', type=list, default=DATASETS, help='set on the top of this script')
parser.add_argument('--file_format', type=list, default=FILEFORMAT)
parser.add_argument('--has_graph', type=list, default=DATA_WITH_GRAPH)
parser.add_argument('--has_node_description', type=list, default=DATA_WITH_DESCRIP)
parser.add_argument('--slide_window_size', type=int, default=30)
parser.add_argument('--label_window_size', type=int, default=19)
parser.add_argument('--slide_window_step', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=200)  
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--input_sequential', type=bool, default=False)
parser.add_argument('--max_node_num', type=int, default=5)
parser.add_argument('--node_features_dim', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--kl_weight', type=float, default=1)
parser.add_argument('--use_graph_prior', action='store_true', default=False)
parser.add_argument('--temp',  type=float, default=0.5)
parser.add_argument('--hard', action='store_true', default=False)
parser.add_argument('--skip_first', action='store_true', default=True)
parser.add_argument('--get_graph_method', type=str, default='corr')
parser.add_argument('--dp', type=eval, default=True, help='True | False')
parser.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
parser.add_argument('--break_train_epoch', type=eval, default=True, help='True | False')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--clip_grad', type=eval, default=True, help='clip grad')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--sib_k', type=int, default=5, help='node selection number')

parser.add_argument('--gsl_module_type', type=str, default='gim', help='Type of path encoder model (mlp ,cnn, gim).')
parser.add_argument('--gsl_feature_len', type=int, default=49, help='')
parser.add_argument('--gsl_feature_dim', type=int, default=2, help='')
parser.add_argument('--gsl_act_fn', type=str, default='silu')
parser.add_argument('--gsl_dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--gsl_feature_layers', type=int, default=2, help='')
parser.add_argument('--gsl_description_layers', type=int, default=2, help='')
parser.add_argument('--gsl_attention', type=bool, default=False, help='')      
parser.add_argument('--gsl_edge_fusion_methad', type=str, default='cat', help='cat,sum,only_x')
parser.add_argument('--gsl_use_layer_norm', type=bool, default=False, help='')   
parser.add_argument('--gsl_use_description_learner', type=bool, default=False, help='') 
parser.add_argument('--gsl_use_prior_learner', type=bool, default=False, help='')  
parser.add_argument('--gsl_feature_hidden_dim', type=int, default=8)
parser.add_argument('--gsl_description_hidden_dim', type=int, default=8)
parser.add_argument('--gsl_interact_type', type=str, default='mlp', help='mlp, mamba')
parser.add_argument('--gsl_skip_first', action='store_true', default=True)
parser.add_argument('--gsl_description_dim', type=int, default=1)
parser.add_argument('--gsl_description_len', type=int, default=512)

parser.add_argument('--encoder_feature_len', type=int, default=30, help='')
parser.add_argument('--encoder_feature_dim', type=int, default=2, help='')
parser.add_argument('--encoder_hidden_dim', type=int, default=32, help='')
parser.add_argument('--encoder_feature_layer', type=int, default=1, help='')
parser.add_argument('--encoder_act_fn', type=str, default='silu')
parser.add_argument('--encoder_use_layer_norm', type=bool, default=True, help='')
parser.add_argument('--encoder_layer_type', type=str, default='mamba', help='rnn, mamba')
parser.add_argument('--encoder_out_dim', type=int, default=32, help='')
parser.add_argument('--encoder_out_len', type=int, default=30, help='')
parser.add_argument('--encoder_dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

parser.add_argument('--pred_decoder_type', type=str, default='gim', help='rnn, gim')
parser.add_argument('--pred_decoder_predict_len', type=int, default=19, help='')
parser.add_argument('--pred_decoder_predict_dim', type=int, default=2, help='')
parser.add_argument('--pred_decoder_dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--pred_decoder_act_fn', type=str, default='silu')
parser.add_argument('--pred_decoder_feature_layers', type=int, default=2, help='')
parser.add_argument('--pred_decoder_attention', type=bool, default=False, help='')
parser.add_argument('--pred_decoder_use_layer_norm', type=bool, default=True, help='') 
parser.add_argument('--pred_decoder_feature_hidden_dim', type=int, default=32)
parser.add_argument('--pred_decoder_interact_type', type=str, default='scm', help='mlp, scm, mamba, work with gim decoder')
parser.add_argument('--pred_decoder_skip_first', action='store_true', default=True)
parser.add_argument('--pred_decoder_use_edge_feature', action='store_true', default=False)

args = parser.parse_args()


print(args)

if __name__ == '__main__':
    main(args)
