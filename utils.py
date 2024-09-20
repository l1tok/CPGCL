import torch
import numpy as np
import scipy.sparse as sp
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import opt
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_graph(k, graph_k_save_path, graph_save_path, data_path):
    if k:
        path = graph_k_save_path
    else:
        path = graph_save_path

    print("Loading path:", path)

    data = np.loadtxt(data_path, dtype=float)
    n, _ = data.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))
    
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def info_nce_loss(v1, v2, temperature=0.2):
    batch_size = v1.shape[0]
    v1 = F.normalize(v1, p=2, dim=1)
    v2 = F.normalize(v2, p=2, dim=1)
    sim_v1 = torch.mm(v1, v1.t())  
    sim_v2 = torch.mm(v2, v2.t())  
    sim_cross_view = torch.mm(v1, v2.t())  

    # print("sim_matrix range:", sim_matrix.min().item(), sim_matrix.max().item())
    sim_cross_view = torch.exp(sim_cross_view / temperature)
    sim_v1 = torch.exp(sim_v1 / temperature)
    sim_v2 = torch.exp(sim_v2 / temperature)

    pos_sim = sim_cross_view[range(batch_size), range(batch_size)]
    same_sample_v1 = sim_v1[range(batch_size), range(batch_size)]
    same_sample_v2 = sim_v2[range(batch_size), range(batch_size)]
  
    loss_0 = pos_sim / (sim_cross_view.sum(dim=1) + sim_v1.sum(dim=1) - same_sample_v1)
    # loss_0 = pos_sim / (sim_matrix.sum(dim=1))
    loss_0 = - torch.log(loss_0).mean()

    loss_1 = pos_sim / (sim_cross_view.sum(dim=0) + sim_v2.sum(dim=0) -same_sample_v2)
    # loss_1 = pos_sim / (sim_matrix.sum(dim=0))
    loss_1 = - torch.log(loss_1).mean()
    loss = (loss_0 + loss_1) / 2.0
    return loss

def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / (q + 1e-12) + 1e-12), dim=-1)

def calculate_jsd_similarity_matrix(c):
    batch_size = c.size(0)

    c_expanded_1 = c.unsqueeze(1)  # [batch_size, 1, num_clusters]
    c_expanded_2 = c.unsqueeze(0)  # [1, batch_size, num_clusters]
    
    m = 0.5 * (c_expanded_1 + c_expanded_2)  # [batch_size, batch_size, num_clusters]

    kl_p_m = kl_divergence(c_expanded_1, m)  # [batch_size, batch_size]
    kl_q_m = kl_divergence(c_expanded_2, m)  # [batch_size, batch_size]
    jsd_matrix = 0.5 * (kl_p_m + kl_q_m)
    
    jsd_similarity_matrix = 1 - jsd_matrix / np.log(2)

    return jsd_similarity_matrix 

def cluster_sim_weight(c1, c2, beta=1):
    sim_c = calculate_jsd_similarity_matrix(c1)
    
    sim_weight = torch.pow(sim_c, beta)
    sim_weight = 1 - sim_weight
    
    diag_mask = torch.eye(sim_weight.size(0), dtype=torch.bool, device=sim_weight.device)
    sim_weight[diag_mask] = 1 #sim_c[diag_mask]
    
    return sim_weight

def info_nce_loss_with_cluster_wight(v1, v2, c1, c2, temperature=0.2):
    batch_size = v1.shape[0]
    c = (c1 + c2) / 2
    # print(c1)
    # print(c2)
    v1 = F.normalize(v1, p=2, dim=1)
    v2 = F.normalize(v2, p=2, dim=1)

    sim_cross_view = torch.mm(v1, v2.t())
    sim_v1 = torch.mm(v1, v1.t())  
    sim_v2 = torch.mm(v2, v2.t())  
    # sim_weight = cluster_sim_weight(c1, c2)
    # sim_weight_v1 = cluster_sim_weight(c1, c1)
    # sim_weight_v2 = cluster_sim_weight(c2, c2)
    sim_weight= cluster_sim_weight(c, c)
    
    sim_cross_view = torch.exp(sim_weight * sim_cross_view / temperature)
    sim_v1 = torch.exp(sim_weight * sim_v1 / temperature)
    sim_v2 = torch.exp(sim_weight * sim_v2 / temperature)
    

    pos_sim = sim_cross_view[range(batch_size), range(batch_size)]
    same_sample_v1 = sim_v1[range(batch_size), range(batch_size)]
    same_sample_v2 = sim_v2[range(batch_size), range(batch_size)]
    # loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
    loss_0 = pos_sim / (sim_cross_view.sum(dim=1) + sim_v1.sum(dim=1) - same_sample_v1)
    loss_0 = - torch.log(loss_0).mean()


    # loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss_1 = pos_sim / (sim_cross_view.sum(dim=0) + sim_v2.sum(dim=0) -same_sample_v2)
    loss_1 = - torch.log(loss_1).mean()
    loss = (loss_0 + loss_1) / 2.0

    
    return loss

def construct_high_confidence_same_cluster_matrix(c):
    confidence_threshold = opt.args.confidence_threshold
    batch_size = c.shape[0]
    high_conf_mask = c.max(dim=1)[0] > confidence_threshold  # [batch_size]
    
    high_conf_cluster = torch.argmax(c, dim=1)  # [batch_size]

    same_cluster = (high_conf_cluster.unsqueeze(1) == high_conf_cluster.unsqueeze(0))  # [batch_size, batch_size]

    high_conf_mask = high_conf_mask.unsqueeze(1) & high_conf_mask.unsqueeze(0)  # [batch_size, batch_size]
    
    P = same_cluster & high_conf_mask
    same_sample_mask = torch.eye(batch_size, dtype=torch.bool, device=c.device)
    P = P | same_sample_mask

    return P.float() 