import argparse

parser = argparse.ArgumentParser(description='CPGCL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--dataset', type=str, default='amap')
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--n_clusters', type=int, default=8) # dblp: 4, cora: 7, cite: 6, amap: 8, acm:3
parser.add_argument('--n_input', type=int, default=100)

parser.add_argument('--data_path', type=str, default='.txt')
parser.add_argument('--label_path', type=str, default='.txt')
parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_components', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1600)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--hidden_dim1', type=int, default=1024)
parser.add_argument('--hidden_dim2', type=int, default=512)
parser.add_argument('--z_dim', type=int, default=512)

parser.add_argument('--beta', type=int, default=0.4)
parser.add_argument('--confidence_threshold', type=int, default=0.75)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()