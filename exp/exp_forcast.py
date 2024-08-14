import exp.exp_utils as exp_utils
import time
import argparse
import pickle
import random
import copy
import torch
from os.path import join
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from modules.data_modules import retrieve_dataloaders
from model.GIM.GIM_modules import get_model
from exp.train import train, test
from datetime import datetime
from exp.exp_utils import plot_trajectory, plot_line, get_log_dir

from torch.utils.tensorboard import SummaryWriter

def seed_everything(seed=42):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)

def main(args):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    # seed_everything(args.seed)

    exp_utils.create_folders(args)

    gradnorm_queue = exp_utils.Queue()
    gradnorm_queue.add(3000) 

    tb_save_dir = get_log_dir(args)
    writer = SummaryWriter(tb_save_dir)

    # Load data
    dataloaders, dataset_info = retrieve_dataloaders(args, load=True, force_download=False)
    data_dummy = next(iter(dataloaders['train']))

    # Create model
    model, nodes_dist = get_model(args, dataset_info)
    optimizer = exp_utils.get_optim(args, model)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

    # model = model.to(device)

    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()

    best_nll_val = 1e8
    best_nll_test = 1e8
    early_stopping = 0
    metrics_save = {'train_loss': [], 'valid_loss': [], 'test_loss': [], 
               'train_kl': [], 'valid_kl': [], 'test_kl': [],
               'train_pred_error': [], 'valid_pred_error': [], 'test_pred_error': [],
               'train_mse_error': [], 'valid_mse_error': [], 'test_mse_error': [],
               'train_mae_error': [], 'valid_mae_error': [], 'test_mae_error': [],
               'train_edge_acc': [], 'valid_edge_acc': [], 'test_edge_acc': [],
               'train_edge_auroc': [], 'valid_edge_auroc': [], 'test_edge_auroc': [],
               'train_edge_acc_one': [], 'valid_edge_acc_one': [], 'test_edge_acc_one': [],
               'train_edge_acc_zero': [], 'valid_edge_acc_zero': [], 'test_edge_acc_zero': []}
    for epoch in range(args.epochs):
        start_epoch = time.time()
        # dataloaders: dict_keys(['train', 'valid', 'test'])
        train_output = train(args=args, loader=dataloaders['train'], epoch=epoch, model=model,
                   nodes_dist=nodes_dist, scheduler=scheduler, 
                   optimizer=optimizer, gradnorm_queue=gradnorm_queue)
        print(f"Epoch {epoch} took {time.time() - start_epoch:.1f} seconds.")
        train_loss = torch.stack(train_output['loss']).mean(dim=0).cpu().detach().numpy()
        train_kl = torch.stack(train_output['kl']).mean(dim=0).cpu().detach().numpy()
        train_pred_error = torch.stack(train_output['pred_error']).mean(dim=0).cpu().detach().numpy()
        train_mse_error = torch.stack(train_output['mse_error']).mean(dim=0).cpu().detach().numpy()
        train_mae_error = torch.stack(train_output['mae_error']).mean(dim=0).cpu().detach().numpy()
        train_edge_acc = torch.stack(train_output['edge_acc']).mean(dim=0).cpu().detach().numpy()
        train_edge_acc_one = torch.stack(train_output['edge_acc_one']).mean(dim=0).cpu().detach().numpy()
        train_edge_acc_zero = torch.stack(train_output['edge_acc_zero']).mean(dim=0).cpu().detach().numpy()
        train_edge_acc_auroc = torch.stack(train_output['auroc']).mean(dim=0).cpu().detach().numpy()
        print(f"train Epoch: {epoch}, Loss: {train_loss:.3f}, kl: {train_kl:.3f}, \
pred_error: {train_pred_error:.3f}, mse_error: {train_mse_error:.3f}, mae_error: {train_mae_error:.3f},\
edge_auroc: {train_edge_acc_auroc:.3f}, edge_acc: {train_edge_acc:.3f},  \
edge_acc_one: {train_edge_acc_one:.3f}, edge_acc_zero: {train_edge_acc_zero:.3f}.")

        if epoch % args.valid_epoch == 0:
            valid_output = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model,
                           partition='Val', nodes_dist=nodes_dist
                           )
            valid_loss = torch.stack(valid_output['loss']).mean(dim=0).cpu().detach().numpy()
            valid_kl = torch.stack(valid_output['kl']).mean(dim=0).cpu().detach().numpy()
            valid_pred_error = torch.stack(valid_output['pred_error']).mean(dim=0).cpu().detach().numpy()
            valid_mse_error = torch.stack(valid_output['mse_error']).mean(dim=0).cpu().detach().numpy()
            valid_mae_error = torch.stack(valid_output['mae_error']).mean(dim=0).cpu().detach().numpy()
            valid_edge_acc = torch.stack(valid_output['edge_acc']).mean(dim=0).cpu().detach().numpy()
            valid_edge_acc_one = torch.stack(valid_output['edge_acc_one']).mean(dim=0).cpu().detach().numpy()
            valid_edge_acc_zero = torch.stack(valid_output['edge_acc_zero']).mean(dim=0).cpu().detach().numpy()
            valid_edge_acc_auroc = torch.stack(valid_output['auroc']).mean(dim=0).cpu().detach().numpy()
            print(f"valid Epoch: {epoch}, Loss: {valid_loss:.3f}, kl: {valid_kl:.3f}, \
pred_error: {valid_pred_error:.3f}, mse_error: {valid_mse_error:.3f}, mae_error: {valid_mae_error:.3f}, \
edge_auroc: {valid_edge_acc_auroc:.3f}, edge_acc: {valid_edge_acc:.3f},  \
edge_acc_one: {valid_edge_acc_one:.3f}, edge_acc_zero: {valid_edge_acc_zero:.3f}.")

            test_output = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model,
                            partition='Test', nodes_dist=nodes_dist)
            show_index = int(len(test_output['pred_edge'])/2)
            pred_edges = torch.cat(test_output['pred_edge'],dim=0).detach().cpu().numpy()
            real_edges = torch.cat(test_output['real_edges'],dim=0).detach().cpu().numpy()
            gt_y = np.concatenate(test_output['gt_y'],axis=0)
            pred_y = np.concatenate(test_output['pred_y'],axis=0)
            np.save(tb_save_dir + '/pred_edges.npy', pred_edges)
            np.save(tb_save_dir + '/real_edges.npy', real_edges)
            np.save(tb_save_dir + '/gt_y.npy', gt_y)
            np.save(tb_save_dir + '/pred_y.npy', pred_y)
            _, pred_edges = test_output['pred_edge'][show_index][0].max(-1)
            real_edges = test_output['real_edges'][show_index][0].to(dtype=torch.int)
            print(pred_edges)
            print(real_edges)
            test_loss = torch.stack(test_output['loss']).mean(dim=0).cpu().detach().numpy()
            test_kl = torch.stack(test_output['kl']).mean(dim=0).cpu().detach().numpy()
            test_pred_error = torch.stack(test_output['pred_error']).mean(dim=0).cpu().detach().numpy()
            test_mse_error = torch.stack(test_output['mse_error']).mean(dim=0).cpu().detach().numpy()
            test_mae_error = torch.stack(test_output['mae_error']).mean(dim=0).cpu().detach().numpy()
            test_edge_acc = torch.stack(test_output['edge_acc']).mean(dim=0).cpu().detach().numpy()
            test_edge_acc_one = torch.stack(test_output['edge_acc_one']).mean(dim=0).cpu().detach().numpy()
            test_edge_acc_zero = torch.stack(test_output['edge_acc_zero']).mean(dim=0).cpu().detach().numpy()
            test_edge_acc_auroc = torch.stack(test_output['auroc']).mean(dim=0).cpu().detach().numpy()
            print(f"test Epoch: {epoch}, Loss: {test_loss:.3f}, kl: {test_kl:.3f}, \
pred_error: {test_pred_error:.3f}, mse_error: {test_mse_error:.3f}, mae_error: {test_mae_error:.3f},\
edge_auroc: {test_edge_acc_auroc:.3f}, edge_acc: {test_edge_acc:.3f}, \
edge_acc_one: {test_edge_acc_one:.3f}, edge_acc_zero: {test_edge_acc_zero:.3f}.")
            
            if valid_loss < best_nll_val:
                best_nll_val = valid_loss
                best_nll_test = test_loss
                early_stopping = 0
                if args.save_model:
                    args.current_epoch = epoch + 1
                    exp_utils.save_model(optimizer, 'outputs/%s/optim.npy' % args.exp_name)
                    exp_utils.save_model(model, 'outputs/%s/best_model.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)
                    print('Model %s saved !' % args.exp_name)
            else:
                # early stopping
                early_stopping += 1
                if early_stopping >= args.early_stopping:
                    print('Early stopping at epoch %d' % epoch)
                    break

            writer.add_scalar('pred_error/train_pred_error', train_pred_error.item(), epoch)
            writer.add_scalar('pred_error/valid_pred_error', valid_pred_error.item(), epoch)
            writer.add_scalar('pred_error/test_pred_error', test_pred_error.item(), epoch)

            writer.add_scalar('mse_error/train_mse_error', train_mse_error.item(), epoch)
            writer.add_scalar('mse_error/valid_mse_error', valid_mse_error.item(), epoch)
            writer.add_scalar('mse_error/test_mse_error', test_mse_error.item(), epoch)

            writer.add_scalar('loss/train_loss', train_loss.item(), epoch) 
            writer.add_scalar('loss/valid_loss', valid_loss.item(), epoch) 
            writer.add_scalar('loss/test_loss', test_loss.item(), epoch)

            writer.add_scalar('kl/train_kl', train_kl.item(), epoch)
            writer.add_scalar('kl/valid_kl', valid_kl.item(), epoch)
            writer.add_scalar('kl/test_kl', test_kl.item(), epoch)

            writer.add_scalar('mae_error/train_mae_error', train_mae_error.item(), epoch)
            writer.add_scalar('mae_error/valid_mae_error', valid_mae_error.item(), epoch)
            writer.add_scalar('mae_error/test_mae_error', test_mae_error.item(), epoch)

            writer.add_scalar('edge_acc/train_edge_acc', train_edge_acc.item(), epoch)
            writer.add_scalar('edge_acc/valid_edge_acc', valid_edge_acc.item(), epoch)
            writer.add_scalar('edge_acc/test_edge_acc', test_edge_acc.item(), epoch)

            writer.add_scalar('best_loss/best_val_loss', best_nll_val, epoch)
            writer.add_scalar('best_loss/best_test_loss', best_nll_test, epoch)

            metrics_save['train_loss'].append(train_loss)
            metrics_save['valid_loss'].append(valid_loss)
            metrics_save['test_loss'].append(test_loss)
            metrics_save['train_kl'].append(train_kl)
            metrics_save['valid_kl'].append(valid_kl)
            metrics_save['test_kl'].append(test_kl)
            metrics_save['train_pred_error'].append(train_pred_error)
            metrics_save['valid_pred_error'].append(valid_pred_error)
            metrics_save['test_pred_error'].append(test_pred_error)
            metrics_save['train_mse_error'].append(train_mse_error)
            metrics_save['valid_mse_error'].append(valid_mse_error)
            metrics_save['test_mse_error'].append(test_mse_error)
            metrics_save['train_mae_error'].append(train_mae_error)
            metrics_save['valid_mae_error'].append(valid_mae_error)
            metrics_save['test_mae_error'].append(test_mae_error)
            metrics_save['train_edge_acc'].append(train_edge_acc)
            metrics_save['valid_edge_acc'].append(valid_edge_acc)
            metrics_save['test_edge_acc'].append(test_edge_acc)
            metrics_save['train_edge_auroc'].append(train_edge_acc_auroc)
            metrics_save['valid_edge_auroc'].append(valid_edge_acc_auroc)
            metrics_save['test_edge_auroc'].append(test_edge_acc_auroc)
            metrics_save['train_edge_acc_one'].append(train_edge_acc_one)
            metrics_save['valid_edge_acc_one'].append(valid_edge_acc_one)
            metrics_save['test_edge_acc_one'].append(test_edge_acc_one)
            metrics_save['train_edge_acc_zero'].append(train_edge_acc_zero)
            metrics_save['valid_edge_acc_zero'].append(valid_edge_acc_zero)
            metrics_save['test_edge_acc_zero'].append(test_edge_acc_zero)
            my_df = pd.DataFrame(metrics_save)
            my_df.to_csv(tb_save_dir + '/metrics.csv')

    print('The last test on the optimal model.')
    model = exp_utils.load_model(model, 'outputs/%s/best_model.npy' % args.exp_name)
    test_output = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model,
                            partition='Test', nodes_dist=nodes_dist)
    test_loss = torch.stack(test_output['loss']).mean(dim=0).cpu().detach().numpy()
    test_kl = torch.stack(test_output['kl']).mean(dim=0).cpu().detach().numpy()
    test_pred_error = torch.stack(test_output['pred_error']).mean(dim=0).cpu().detach().numpy()
    test_mse_error = torch.stack(test_output['mse_error']).mean(dim=0).cpu().detach().numpy()
    test_mae_error = torch.stack(test_output['mae_error']).mean(dim=0).cpu().detach().numpy()
    test_edge_acc = torch.stack(test_output['edge_acc']).mean(dim=0).cpu().detach().numpy()
    test_edge_acc_one = torch.stack(test_output['edge_acc_one']).mean(dim=0).cpu().detach().numpy()
    test_edge_acc_zero = torch.stack(test_output['edge_acc_zero']).mean(dim=0).cpu().detach().numpy()
    test_edge_acc_auroc = torch.stack(valid_output['auroc']).mean(dim=0).cpu().detach().numpy()
    print(f"test Epoch: {epoch}, Loss: {test_loss:.3f}, kl: {test_kl:.3f}, \
pred_error: {test_pred_error:.3f}, mse_error: {test_mse_error:.3f}, mae_error: {test_mae_error:.3f}, \
edge_auroc: {test_edge_acc_auroc:.3f}, edge_acc: {test_edge_acc:.3f}, \
edge_acc_one: {test_edge_acc_one:.3f}, edge_acc_zero: {test_edge_acc_zero:.3f}.")
    pred_edges = torch.cat(test_output['pred_edge'],dim=0).detach().cpu().numpy()
    real_edges = torch.cat(test_output['real_edges'],dim=0).detach().cpu().numpy()
    gt_y = np.concatenate(test_output['gt_y'],axis=0)
    pred_y = np.concatenate(test_output['pred_y'],axis=0)
    label = np.concatenate(test_output['label'],axis=0)
    np.save(tb_save_dir + '/last_pred_edges.npy', pred_edges)
    np.save(tb_save_dir + '/last_real_edges.npy', real_edges)
    np.save(tb_save_dir + '/last_gt_y.npy', gt_y)
    np.save(tb_save_dir + '/last_pred_y.npy', pred_y)
    np.save(tb_save_dir + '/last_label.npy', label)
    show_index = 0
    if args.exp_name == 'sim_springs5':
        plot_trajectory(test_output['pred_y'][show_index], test_output['gt_y'][show_index], tb_save_dir)
    else:
        plot_line(test_output['pred_y'][show_index], test_output['gt_y'][show_index], tb_save_dir, 15)

