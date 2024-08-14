from __future__ import division
from __future__ import print_function
import exp.exp_utils as exp_utils
import torch

def train(args, loader, epoch, model, nodes_dist, scheduler, optimizer, gradnorm_queue):

    loss_train = []
    kl_train = []
    pred_error_train = []
    mse_error_train = []
    mae_error_train = []
    edge_acc_train = []
    edge_acc_train_one = []
    edge_acc_train_zero = []
    auroc_train = []
    model.train()
    scheduler.step()

    for batch_idx, data in enumerate(loader):

        x = data['nodes_timeseries'].permute(0,3,2,1).to(args.device)  # [batch, nodes, features, timesteps]
        h = data['node_types'].permute(0,3,2,1).to(args.device)
        y = data['target'].permute(0,3,2,1).to(args.device)
        g_target = data['real_adj_1d'].to(args.device)
        label = data['label'].to(args.device)
        nodes_mask = data['node_mask_1d'].to(torch.float32).to(args.device)
        edges_mask = data['edge_mask_1d'].to(torch.float32).to(args.device)

        optimizer.zero_grad()
        loss_dict = model(x, h, y, g_target, nodes_mask, edges_mask, epoch)

        nll = loss_dict['loss']

        # Average over batch.
        loss = nll.mean(0)
        loss.backward()

        if args.clip_grad:
            _ = exp_utils.gradient_clipping(model, gradnorm_queue)

        optimizer.step()

        loss_kl = loss_dict['loss_kl']
        pred_error = loss_dict['pred_error']
        mse_error = loss_dict['mse_error']
        mae_error = loss_dict['mae_error']
        edge_acc_zero = loss_dict['edge_acc'][1]
        edge_acc_one = loss_dict['edge_acc'][0]
        edge_acc = loss_dict['edge_acc'][2]
        show_edges = loss_dict['edges']
        auroc = loss_dict['auroc']

        loss_train.append(loss.mean(0))
        kl_train.append(loss_kl.mean(0))
        pred_error_train.append(pred_error.mean(0))
        mse_error_train.append(mse_error.mean(0))
        mae_error_train.append(mae_error.mean(0))
        edge_acc_train.append(edge_acc)
        edge_acc_train_one.append(edge_acc_one)
        edge_acc_train_zero.append(edge_acc_zero)
        auroc_train.append(auroc.mean(0))

    return {'loss': loss_train, 'kl': kl_train, 
            'pred_error': pred_error_train, 
            'mse_error': mse_error_train,
            'mae_error': mae_error_train,
            'edge_acc': edge_acc_train,
            'edge_acc_one': edge_acc_train_one,
            'edge_acc_zero': edge_acc_train_zero,
            'edges': show_edges,
            'real_edges': g_target,
            'auroc': auroc_train}



def test(args, loader, epoch, eval_model, nodes_dist, partition='Test'):
    loss_test = []
    kl_test = []
    pred_error_test = []
    mse_error_test = []
    mae_error_test = []
    edge_acc_test = []
    edge_acc_test_one = []
    edge_acc_test_zero = []
    auroc_test = []
    pred_edge = []
    real_edge = []
    pred_y = []
    gt_y = []
    label_y = []
    eval_model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            x = data['nodes_timeseries'].permute(0,3,2,1).to(args.device)  # [batch, nodes, features, timesteps]
            h = data['node_types'].permute(0,3,2,1).to(args.device)
            y = data['target'].permute(0,3,2,1).to(args.device)
            label = data['label'].detach().cpu().numpy()
            g_target = data['real_adj_1d'].to(args.device)
            nodes_mask = data['node_mask_1d'].to(torch.float32).to(args.device)
            edges_mask = data['edge_mask_1d'].to(torch.float32).to(args.device)

            loss_dict = eval_model(x, h, y, g_target, nodes_mask, edges_mask, epoch)
            nll = loss_dict['loss']

            # Average over batch.
            loss = nll.mean(0)

            loss_kl = loss_dict['loss_kl']
            pred_error = loss_dict['pred_error']
            mse_error = loss_dict['mse_error']
            mae_error = loss_dict['mae_error']
            edge_acc_zero = loss_dict['edge_acc'][1]
            edge_acc_one = loss_dict['edge_acc'][0]
            edge_acc = loss_dict['edge_acc'][2]
            show_edges = loss_dict['edges']
            gt_x = loss_dict['gt_x'].permute(0,1,3,2).detach().cpu().numpy()
            pred_x = loss_dict['pred_x'].permute(0,1,3,2).detach().cpu().numpy()
            auroc = loss_dict['auroc']

            loss_test.append(loss.mean(0))
            kl_test.append(loss_kl.mean(0))
            pred_error_test.append(pred_error.mean(0))
            mse_error_test.append(mse_error.mean(0))
            mae_error_test.append(mae_error.mean(0))
            edge_acc_test.append(edge_acc)
            edge_acc_test_one.append(edge_acc_one)
            edge_acc_test_zero.append(edge_acc_zero)
            auroc_test.append(auroc.mean(0))
            pred_edge.append(show_edges)
            real_edge.append(g_target)
            pred_y.append(pred_x)
            gt_y.append(gt_x)
            label_y.append(label)

        return {'loss': loss_test,
                'kl': kl_test, 
                'pred_error': pred_error_test, 
                'mse_error': mse_error_test,
                'mae_error': mae_error_test,
                'edge_acc': edge_acc_test,
                'edge_acc_one': edge_acc_test_one,
                'edge_acc_zero': edge_acc_test_zero,
                'pred_edge': pred_edge,
                'real_edges': real_edge,
                'gt_y':gt_y,
                'pred_y':pred_y,
                'label':label_y,
                'auroc':auroc_test}


