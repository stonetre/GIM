import numpy as np
import getpass
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime

def create_folders(args):
    try:
        os.makedirs('outputs')
    except OSError:
        pass

    try:
        os.makedirs('outputs/' + args.exp_name)
    except OSError:
        pass

def get_optim(args, generative_model):
    optim = torch.optim.AdamW(
        generative_model.parameters(),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim

# Model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


#Gradient clipping
class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def gradient_clipping(flow, gradnorm_queue):
    # Allow gradient norm to be 150% + 2 * stdev of the recent history.
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    # Clips gradient and returns the norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f'Clipped gradient with value {grad_norm:.1f} '
              f'while allowed {max_grad_norm:.1f}')
    return grad_norm


def plot_trajectory(pred, gt, save_path):
    color_list = ['red', 'blue', 'green', 'orange', 'magenta', 'cyan', 'purple', 'brown', 'pink', 'gray']
    for j in range(10):
        x_gt = gt[j,:,:,0]
        y_gt = gt[j,:,:,1]
        x_pred = pred[j,:,:,0]
        y_pred = pred[j,:,:,1]
        N = x_gt.shape[0]

        fig, ax = plt.subplots()
        for i in range(N):
            ax.plot(x_gt[i], y_gt[i], color=color_list[i], alpha=0.2, linewidth=5)
            ax.plot(x_pred[i], y_pred[i], color=color_list[i], alpha=0.9, linewidth=5)
            ax.scatter(x_gt[i], y_gt[i], color=color_list[i], alpha=0.2, s=100, zorder=5, edgecolor='none')
            ax.scatter(x_pred[i], y_pred[i], color=color_list[i], alpha=0.7, s=100, zorder=5, edgecolor='none')

        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)
        save_name = save_path + '/trajectory' + str(j) + '.svg'
        plt.savefig(save_name)

def plot_line(pred, gt, save_path, num=10):
    color_list = ['red', 'blue', 'green', 'orange', 'magenta', 'cyan', 'purple', 'brown', 
                  'pink', 'gray', 'lime', 'teal', 'lavender', 'salmon', 'gold', 'black', 
                  'white', 'silver', 'olive', 'maroon', 'navy', 'aqua', 'fuchsia', 'lime', 
                  'teal', 'lavender', 'salmon', 'gold', 'black', 'white', 'silver', 'olive', 
                  'maroon', 'navy', 'aqua', 'fuchsia']
    # color_list = plt.cm.viridis(np.linspace(0, 1, gt.shape[1]))
    for j in range(num):
        if gt.shape[-1]==3:
            x_gt = gt[j,:,:,1]
            x_pred = pred[j,:,:,1]
        else:
            x_gt = gt[j,:,:,0]
            x_pred = pred[j,:,:,0]
        N = x_gt.shape[0]
        t = np.linspace(0, x_gt.shape[-1]-1, x_gt.shape[-1])
        t2 = np.linspace(x_gt.shape[-1]-x_pred.shape[-1], x_gt.shape[-1]-1, x_pred.shape[-1])

        fig, ax = plt.subplots(N,1)
        for i in range(N):
            ax[i].plot(t, x_gt[i], color=color_list[i], alpha=0.3, linewidth=3)
            ax[i].plot(t2, x_pred[i], color=color_list[i], alpha=0.9, linewidth=3)
            if i !=N-1:
                ax[i].tick_params(labelbottom=False)
            ax[i].tick_params(labelleft=False)

        # ax.set_xlim(-2, 2)
        # ax.set_ylim(-2, 2)
        save_name = save_path + '/trajectory' + str(j) + '.svg'
        plt.savefig(save_name)

def get_log_dir(args):
     # tensorboard config
    exp_name = args.exp_name
    now = datetime.now()
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minite = now.strftime("%M")

    gsl_layer = str(args.gsl_feature_layers)
    pred_dec_layer = str(args.pred_decoder_feature_layers)
    gsl_desp = "_UseDes" if args.gsl_use_description_learner else "_NotUseDes"
    gsl_norm = "_UseLN--" if args.gsl_use_layer_norm else "_UseBN--"
    pred_dec_norm = "_UseLN--" if args.pred_decoder_use_layer_norm else "_UseBN--"
    if args.gsl_interact_type=='mlp':
        enc_it = "_IntMLP"
    elif args.gsl_interact_type=='dmlp':
        enc_it = "_IntDMLP"
    elif args.gsl_interact_type=='dmlp_mm':
        enc_it = "_IntDMLPMM"
    elif args.gsl_interact_type=='cnn':
        enc_it = "_IntCNN"
    elif args.gsl_interact_type=='mamba':
        enc_it = "_IntMamba"
    elif args.gsl_interact_type=='scm':
        enc_it = "_IntSCM"
    else:
        raise ValueError('Invalid gsl_interact_type.')
    if args.pred_decoder_interact_type=='mlp':
        pred_dec_it = "_IntMLP"
    elif args.pred_decoder_interact_type=='dmlp':
        pred_dec_it = "_IntDMLP"
    elif args.pred_decoder_interact_type=='dmlp_mm':
        pred_dec_it = "_IntDMLPMM"
    elif args.pred_decoder_interact_type=='cnn':
        pred_dec_it = "_IntCNN"
    elif args.pred_decoder_interact_type=='mamba':
        pred_dec_it = "_IntMamba"
    elif args.pred_decoder_interact_type=='scm':
        pred_dec_it = "_IntScm"
    else:
        raise ValueError('Invalid pred_decoder_interact_type.')
    
    if args.gsl_module_type=='gim':
        gsl_model = "GSLGIM_L" + gsl_layer + enc_it + gsl_desp + gsl_norm
    elif args.gsl_module_type=='mlp':
        gsl_model = "GSLMLP_L" + gsl_layer + enc_it + gsl_desp + gsl_norm
    elif args.gsl_module_type=='cnn':
        gsl_model = "GSLCNN_L" + gsl_layer + enc_it + gsl_desp + gsl_norm
    else:
        raise ValueError('Invalid encoder type.')
    if args.pred_decoder_type=='gim':
        pred_dec_model = "PredDecGIM_L" + pred_dec_layer + pred_dec_it + pred_dec_norm
    elif args.pred_decoder_type=='mlp':
        pred_dec_model = "PredDecMLP_L" + pred_dec_layer + pred_dec_it + pred_dec_norm
    elif args.pred_decoder_type=='cnn':
        pred_dec_model = "PredDecCNN_L" + pred_dec_layer + pred_dec_it + pred_dec_norm
    elif args.pred_decoder_type=='rnn':
        pred_dec_model = "PredDecRNN_"
    else:
        raise ValueError('Invalid pred_decoder type.')

    exp_date = day + '_' + hour + '_' + minite   
    tb_save_dir = 'outputs/' + exp_name +'/'+ gsl_model + pred_dec_model + exp_date 
    print('log name:', tb_save_dir)
    return tb_save_dir
