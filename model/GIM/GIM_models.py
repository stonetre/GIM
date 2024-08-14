import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from model.GIM.GIM_utils import *

class GIM(torch.nn.Module):
    def __init__(
            self, graph_learner, encoder, pred_decoder,
            gsl_feature_len, gsl_feature_dim,
            gsl_use_prior_learner,
            encoder_feature_len, encoder_feature_dim, 
            pred_decoder_predict_len, pred_decoder_predict_dim,
            use_graph_prior, 
            kl_weight, temp, hard,
            sib_k, device
            ):
        super().__init__()
        self.graph_learner = graph_learner
        self.encoder = encoder
        self.pred_decoder = pred_decoder
        self.gsl_feature_len = gsl_feature_len
        self.gsl_feature_dim = gsl_feature_dim
        self.gsl_use_prior_learner = gsl_use_prior_learner
        self.encoder_feature_len = encoder_feature_len
        self.encoder_feature_dim = encoder_feature_dim
        self.pred_decoder_predict_len = pred_decoder_predict_len
        self.pred_decoder_predict_dim = pred_decoder_predict_dim
        self.kl_weight = kl_weight
        self.use_graph_prior = use_graph_prior
        self.temp = temp
        self.hard = hard
        self.sib_k = sib_k

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def compute_prediction_error(self, y_pred, y_label):
        """Computes reconstruction error."""
        # prediction error
        error_x = nll_gaussian(y_pred, y_label, 5e-5)
        # error_x = sum_except_batch(error_x)
        return error_x
    
    
    def compute_loss(self, x, h, y, g_target, nodes_mask, edges_mask, epoch):
        """Computes an estimator for the variational lower bound."""

        batch_input = torch.cat([x,y],dim=-1)
        # gsl 
        gsl_input = batch_input[:,:,:self.gsl_feature_dim,:self.gsl_feature_len]
        gsl_prior = batch_input[:,:,:self.gsl_feature_dim,-self.gsl_feature_len:]
        # prediction
        x_len = batch_input.shape[-1]
        input_start = x_len - self.encoder_feature_len - self.pred_decoder_predict_len
        input_stop = x_len - self.pred_decoder_predict_len
        encoder_input = batch_input[:,:,:self.encoder_feature_dim, input_start:input_stop]
        pred_decoder_y_label = batch_input[:,:,:self.pred_decoder_predict_dim,-self.pred_decoder_predict_len:]

        z_g_logits = self.graph_learner(gsl_input, gsl_prior, h, nodes_mask, edges_mask)
        if not self.training:
            z_g_logits[0] = torch.where(torch.isnan(z_g_logits[0]), torch.tensor(1.0), z_g_logits[0])
        prob = my_softmax(z_g_logits[0], -1)
        edges = gumbel_softmax(z_g_logits[0], tau=self.temp, hard=self.hard)
        edges = (edges.permute(2,0,1) * edges_mask).permute(1,2,0)

        if self.gsl_use_prior_learner:
            if not self.training:
                z_g_logits[-1] = torch.where(torch.isnan(z_g_logits[-1]), torch.tensor(1.0), z_g_logits[-1])
            prob_prior = my_softmax(z_g_logits[-1], -1)
        else:
            prob_prior = None

        # KL for graph.
        if self.use_graph_prior:
            loss_kl_g = kl_categorical(prob, g_target, edges_mask, nodes_mask)
        else:
            loss_kl_g = kl_categorical_uniform(prob, edges_mask, nodes_mask)

        if self.gsl_use_prior_learner:
            loss_kl_g_prior = kl_categorical(prob, prob_prior, edges_mask, nodes_mask)
        
        # Decoder.
        # mask = (torch.rand_like(nodes_mask) > 0.2).to(torch.float32) 
        # mask = mask.to(nodes_mask.device)

        batch_size, nodes = nodes_mask.size()
        k = self.sib_k 
        indices = torch.randint(0, nodes, (batch_size, nodes), device=nodes_mask.device)
        indices = torch.argsort(indices, dim=1)[:, :k]
        mask = torch.zeros((batch_size, nodes), device=nodes_mask.device, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        mask_rev = nodes_mask * mask  # each sample has k 1
  
        # edges = torch.zeros_like(edges).to(edges.device)
        z_enc, z_mu, z_sigma = self.encoder(encoder_input, nodes_mask, mask_rev)
        pred_decoder_y_output = self.pred_decoder(z_enc, edges, nodes_mask, edges_mask)

        # loss_kl_x = gaussian_KL(z_mu, z_sigma, nodes_mask)
        loss_kl_x = gaussian_KL(z_mu, z_sigma, nodes_mask, mask_rev)

        if self.training:
            pred_decoder_y_output = torch.einsum('abcd,ab->abcd', pred_decoder_y_output, mask_rev)
            pred_decoder_y_label = torch.einsum('abcd,ab->abcd', pred_decoder_y_label, mask_rev)

        error_pred_y = self.compute_prediction_error(pred_decoder_y_output, pred_decoder_y_label)  # batch
        # error_g = self.compute_edge_error(edges, g_target) # batch
        mse_error = self.mse(pred_decoder_y_output, pred_decoder_y_label)
        mae_error = self.mae(pred_decoder_y_output, pred_decoder_y_label)

        auroc = calculate_auroc_batch(edges[:,:,1], g_target, edges_mask)
        acc = edge_accuracy(edges, g_target, edges_mask)

        # Loss.
        if self.gsl_use_prior_learner:
            loss = 1e-5*(error_pred_y + loss_kl_x + loss_kl_g_prior)
            # loss = 1e-5*(error_pred_y + loss_kl_g_prior)
        else:
            loss = 1e-5*(error_pred_y + loss_kl_x + loss_kl_g)
            # loss = error_pred_y + self.kl_weight*loss_kl_g

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return {'loss': loss, 'loss_kl':loss_kl_g.squeeze()* self.kl_weight, 
                'pred_error': error_pred_y.squeeze(), 'mse_error': mse_error, 
                'mae_error': mae_error, 'edge_acc': acc, 'edges':edges,
                'gt_x':batch_input[:,:,:self.pred_decoder_predict_dim,:],
                'pred_x':pred_decoder_y_output, 'auroc':auroc}

    def forward(self, x, h, y, g_target, nodes_mask, edges_mask, epoch):

        loss_dict = self.compute_loss(x, h, y, g_target, nodes_mask, edges_mask, epoch)

        return loss_dict
    
    
    


    