import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def kl_categorical(preds, prior, edges_mask, nodes_mask, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - torch.log(prior + eps))
    kl_div = (kl_div.permute(2,0,1) * edges_mask).permute(1,2,0)
    kl_div = kl_div.sum(-1).sum(-1)
    
    return kl_div/nodes_mask.sum(-1)


def kl_categorical_uniform(preds, edge_mask, nodes_mask, add_const=False, 
                           eps=1e-16):
    num_edge_types=preds.shape[-1]
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    kl_div = (kl_div.permute(2,0,1) * edge_mask).permute(1,2,0)
    kl_div = kl_div.reshape(kl_div.size(0), -1).sum(-1)/nodes_mask.sum(-1)
    return kl_div

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    neg_log_p = sum_except_batch(neg_log_p)/(target.size(1))
    return neg_log_p


def edge_accuracy(preds, target, edges_mask):
    _, preds = preds.max(-1)
    ones_pos = torch.where(target == 1)
    zeros_pos = torch.where(target == 0)
    false_zero = torch.where(edges_mask == 0)
    correct = preds.float().data.eq(target.float().data)
    correct = correct * edges_mask
    ones_precision = correct[ones_pos].sum().float()/(len(ones_pos[0]))
    zreo_precision = correct[zeros_pos].sum().float()/(len(zeros_pos[0])-len(false_zero[0]))
    total_precision = correct.sum().float()/(edges_mask.sum().float())
    precision = torch.cat([ones_precision.unsqueeze(0), zreo_precision.unsqueeze(0), total_precision.unsqueeze(0)])
    return precision

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(-1)

def gaussian_KL(p_mu, p_sigma, node_mask, mask_rev):
    cach = 1 + p_sigma - p_mu.pow(2) - torch.exp(p_sigma)
    cach = torch.einsum('abcd,ab->abcd', cach.permute(0,2,1,3), mask_rev).permute(0,2,1,3)
    cach = torch.einsum('abcd,ab->abcd', cach.permute(0,2,1,3), node_mask).permute(0,2,1,3)
    return sum_except_batch(
            (-0.5 * cach)/p_mu.size(1)
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (d * torch.log(p_sigma / (q_sigma + 1e-8) + 1e-8) 
            + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) 
            - 0.5 * d
            )


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def calc_auroc(pred_edges, GT_edges):
    pred_edges = pred_edges[:, :, 1]
    # return roc_auc_score(
    #     GT_edges.cpu().detach().flatten(),
    #     pred_edges.cpu().detach().flatten(),  # [:, :, 1]
    # )


def compute_auroc(pred, real, mask):
    """
    calc AUROC
    :param pred: [batch, edges]
    :param real: [batch, edges]
    :param mask:  [batch, edges]
    :return: AUROC
    """

    pred_flat = pred.reshape(-1)
    real_flat = real.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    pred_masked = pred_flat[mask_flat == 1]
    real_masked = real_flat[mask_flat == 1]
    
    pred_masked = pred_masked.detach().cpu().numpy()
    real_masked = real_masked.detach().cpu().numpy()
    
    auroc = roc_auc_score(real_masked, pred_masked)
    
    return auroc

def calculate_auroc(pred, real, mask):

    pred_masked = pred[mask == 1]
    real_masked = real[mask == 1]

    sorted_indices = torch.argsort(pred_masked, descending=True)
    sorted_real = real_masked[sorted_indices]

    tpr = torch.cumsum(sorted_real, dim=0) / torch.sum(sorted_real)
    fpr = torch.cumsum(1 - sorted_real, dim=0) / torch.sum(1 - sorted_real)

    auc = torch.trapz(tpr, fpr)

    return auc

def calculate_auroc_batch(pred, real, mask):
    batch_size = pred.shape[0]
    auroc_list = []

    for i in range(batch_size):

        pred_masked = pred[i][mask[i] == 1]
        real_masked = real[i][mask[i] == 1]

        if torch.unique(real_masked).numel() == 1:
            # auroc_list.append(torch.tensor(float('nan')))
            continue

        sorted_indices = torch.argsort(pred_masked, descending=True)
        sorted_real = real_masked[sorted_indices]

        tpr = torch.cumsum(sorted_real, dim=0) / torch.sum(sorted_real)
        fpr = torch.cumsum(1 - sorted_real, dim=0) / torch.sum(1 - sorted_real)

        auc = torch.trapz(tpr, fpr)
        auroc_list.append(auc)
    auroc = torch.stack(auroc_list)

    return auroc

def get_act_fn(act):
    if act == 'relu':
        act_fn = torch.nn.ReLU()
    elif act == 'leaky_relu':
        act_fn = torch.nn.LeakyReLU()
    elif act == 'elu':
        act_fn = torch.nn.ELU()
    elif act == 'selu':
        act_fn = torch.nn.SELU()
    elif act == 'gelu':
        act_fn = torch.nn.GELU()
    elif act == 'silu':
        act_fn = torch.nn.SiLU()
    else:
        raise ValueError(act)
    return act_fn


def repara_trick(x_mu, x_sigma, kl_mask):
    eps = torch.randn_like(x_mu).to(x_mu.device)
    noise =  x_sigma*eps 
    if kl_mask is not None:
        noise = torch.einsum('abcd,ab->abcd', noise.permute(0,2,1,3), kl_mask).permute(0,2,1,3) 
    x = x_mu + noise
    return x
