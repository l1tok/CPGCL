
import opt 
import torch
import numpy as np

from utils import *
from train import Train_model
from sklearn.decomposition import PCA
from module import GCN

import warnings
warnings.filterwarnings('ignore')
setup_seed(np.random.randint(1000))

device = torch.device("cuda" if opt.args.cuda else "cpu")

opt.args.data_path = 'data/{}_feature.txt'.format(opt.args.dataset)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.dataset)
opt.args.graph_k_save_path = 'data/{}{}_graph.txt'.format(opt.args.dataset, opt.args.k)
opt.args.graph_save_path = 'data/{}_graph.txt'.format(opt.args.dataset)
opt.args.device = device


print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

x = np.loadtxt(opt.args.data_path, dtype=float)
y = np.loadtxt(opt.args.label_path, dtype=int)

opt.args.k = None
adj = load_graph(opt.args.k, opt.args.graph_k_save_path, opt.args.graph_save_path, opt.args.data_path).to(device)
adj = adj.to_dense()


pca1 = PCA(n_components=opt.args.n_components)
x1 = pca1.fit_transform(x)
dataset = LoadDataset(x1)
data = torch.Tensor(dataset.x).to(device)

model = GCN(n_input=data.shape[1], hidden_dim1=opt.args.hidden_dim1, hidden_dim2=opt.args.hidden_dim2, z_dim=opt.args.z_dim,
            n_clusters=opt.args.n_clusters).to(device)

Train_model(model, data, adj.to(device), y)


