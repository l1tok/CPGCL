import torch
import opt
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F

from utils import *
import warnings
warnings.filterwarnings('ignore')


def Train_model(model, data, adj, label):
    best_epoch = -1
    best_per = [0, 0, 0, 0]
    model_loss_result = []
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    device = torch.device("cuda" if opt.args.cuda else "cpu")
    
    aug_adj = torch.FloatTensor(np.eye(data.shape[0])).to(device)
    
    for epoch in range(opt.args.epoch):
        X1 = data
        X2 = data
       
        model.train()
        model.zero_grad()
        
        z1, c1 = model(X1, adj)
        
        z2, c2 = model(X2, aug_adj)

        z_mat=torch.matmul(z1, z2.T)



        z = (z1 + z2) / 2    
        c = (c1 + c2) / 2   

        loss_hmse = F.mse_loss(z_mat, construct_high_confidence_same_cluster_matrix(c))
        
        loss_community = info_nce_loss(c1.T, c2.T) 
        loss_node = info_nce_loss_with_cluster_wight(z1, z2, c1, c2)
        model_loss = loss_community +loss_node + opt.args.beta * loss_hmse

        print('loss_hmse:{}'.format(loss_hmse),'loss_community:{}'.format(loss_community)
              ,'loss_node:{}'.format(loss_node))
        print('{} loss: {}'.format(epoch, model_loss))

        
        model_loss.backward() 
        optimizer.step()
        model.eval()

        
        label_hat = c.argmax(dim=-1)

        acc, nmi, ari, f1 = eva(label, label_hat.data.cpu().numpy(), epoch)

        model_loss_result.append(model_loss.cpu().item())
        if acc > best_per[0]:
            best_epoch = epoch
            best_per[0] = acc
            best_per[1] = nmi
            best_per[2] = ari
            best_per[3] = f1


    print('Epoch_{}'.format(best_epoch), ':acc {:.4f}'.format(best_per[0]), ', nmi {:.4f}'.format(best_per[1]), 
          ', ari {:.4f}'.format(best_per[2]), ', f1 {:.4f}'.format(best_per[3]))

