import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.model_zoo as model_zoo
import math

# Augment adjacency matrix as follows:

#   A_text | A_ti
#  ---------------
#   A_ti.T | A_img

def gen_Aall(A_text, l_text, l_img):
    A_ti = torch.eq(l_text.unsqueeze(1),l_img.unsqueeze(1).T).float()
    A_img = torch.eq(l_img.unsqueeze(1),l_img.unsqueeze(1).T).float()
    #A_img = torch.eye(l_img.size(0)).cuda()
    A_t = torch.cat((A_text, A_ti), dim=1)
    A_i = torch.cat((A_ti.T,A_img), dim=1)
    A_all = torch.cat((A_t,A_i), dim=0)
    return A_all


def gen_A(features):
    #th = 0.95
    features_norm = features / features.norm(dim=-1,keepdim=True)
    _adj = features_norm @ features_norm.T
    _adj = (_adj - _adj.min())/(_adj.max()-_adj.min())
    for th in [0.8,0.9,0.95]:
        similar = (_adj > th).float()
        above = similar.sum()
        below = (1-similar).sum()
        print("ratio-{}:{:.1f}".format(th, above*100/(above+below)),end=' ')
    print('')
    #print("above {}, below {} ratio {:.1f}".format(above, below, above*100/(above+below)))
    #for i in range(_adj.size(0)):
    #    for j in range(_adj.size(1)):
    #        _adj[i][j] = 0 if _adj[i][j] < 0.5 else 1
    return similar

def gen_A2(features, ratio=0.002): #v2: generated by ratio
    #th = 0.95
    #ratio = 0.002
    ratio = 1 - ratio
    features_norm = features / features.norm(dim=-1,keepdim=True)
    _adj = features_norm @ features_norm.T
    #_adj = (_adj - _adj.min())/(_adj.max()-_adj.min())
    #for th in [0.8,0.9,0.95]:
    _adj_flat = _adj.view(-1)
    th, _ = torch.kthvalue(_adj_flat, int(ratio*_adj_flat.size(0))) #top k-th value of the full similarity matrix
    similar = (_adj >= th).float()
    above = similar.sum()
    below = (1-similar).sum()
    #print("ratio-{}:{:.1f}".format(th, above*100/(above+below)),end=' ')
    #print('')
    print("above {}, below {} ratio {:.1f}".format(above, below, above*100/(above+below)))
    #for i in range(_adj.size(0)):
    #    for j in range(_adj.size(1)):
    #        _adj[i][j] = 0 if _adj[i][j] < 0.5 else 1
    return similar*_adj

def gen_A2Plus(features, ratio=0.002): #v2: generated by ratio
    nprompts = 20
    ncls = int(features.size(0)/nprompts)
    mask = 1 - gen_A_gt(ncls, nprompts)
    #th = 0.95
    #ratio = 0.002
    ratio = 1 - ratio
    features_norm = features / features.norm(dim=-1,keepdim=True)
    _adj = features_norm @ features_norm.T
    _adj_m = _adj * mask
    #_adj = (_adj - _adj.min())/(_adj.max()-_adj.min())
    #for th in [0.8,0.9,0.95]:
    _adj_flat = _adj_m.view(-1)
    th, _ = torch.kthvalue(_adj_flat, int(ratio*_adj_flat.size(0))) #top k-th value of the full similarity matrix
    similar = (_adj >= th).float()
    above = similar.sum()
    below = (1-similar).sum()
    #print("ratio-{}:{:.1f}".format(th, above*100/(above+below)),end=' ')
    #print('')
    print("above {}, below {} ratio {:.1f}".format(above, below, above*100/(above+below)))
    similar = similar * mask + (1-mask)
    #for i in range(_adj.size(0)):
    #    for j in range(_adj.size(1)):
    #        _adj[i][j] = 0 if _adj[i][j] < 0.5 else 1
    return similar*_adj

def gen_A3(features, ratio=0.002): #v3: ratio + eculidean distance
    #th = 0.95
    #ratio = 0.001 #change the ratio for datasets
    beta = 1.0
    ratio = 1 - ratio
    N, C = features.size()
    #features_expand1 = features.cpu().repeat(N,1)
    #features_expand2 = features.cpu().repeat(1,N).view(-1,C)
    #l2_dis = (features_expand1-features_expand2).norm(dim=-1).cuda()
    l2_dis = [] 
    for i in range(N):
        features_expand = features[i].repeat(N,1)
        dis = (features_expand-features).norm(dim=-1)
        l2_dis.append(dis) 
    l2_dis = torch.cat(l2_dis, dim=0)
    _adj = torch.exp(-l2_dis*beta).view(N,N)
    #_adj = (_adj - _adj.min())/(_adj.max()-_adj.min())
    #for th in [0.8,0.9,0.95]:
    _adj_flat = _adj.view(-1)
    th, _ = torch.kthvalue(_adj_flat, int(ratio*_adj_flat.size(0))) #top k-th value of the full similarity matrix
    similar = (_adj >= th).float()
    above = similar.sum()
    below = (1-similar).sum()
    #print("ratio-{}:{:.1f}".format(th, above*100/(above+below)),end=' ')
    #print('')
    print("above {}, below {} ratio {:.1f}".format(above, below, above*100/(above+below)))
    #for i in range(_adj.size(0)):
    #    for j in range(_adj.size(1)):
    #        _adj[i][j] = 0 if _adj[i][j] < 0.5 else 1
    return similar*_adj

def gen_A_gt(ncls,nprompts):
    a = torch.ones(nprompts,nprompts)
    a_all = a.unsqueeze(0).repeat(ncls,1,1)
    return torch.block_diag(*a_all).cuda()


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        #print("bias",bias)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #print(self.weight.data)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN_aug(nn.Module):

    def __init__(self, text_features, text_labels, ratio):
        n_class, n_prompt, text_dim = text_features.size()
        #print("size2:", text_features.size())
        self.text_dim = text_dim
        #print(self.text_dim)
        self.text_features = text_features.reshape(-1, self.text_dim).cuda()
        self.n_texts = self.text_features.size(0)
        self.ratio = ratio
        super(GCN_aug, self).__init__()

        self.gnn_gc1 = GraphConvolution(self.text_dim, self.text_dim)
        self.gnn_gc2 = GraphConvolution(self.text_dim, n_class)
        self.gnn_relu1 = nn.LeakyReLU(0.2)
        #st = torch.load("gcn_state_dict.pt")
        #self.load_state_dict(st,strict=False)

        self.Atext = gen_A2Plus(self.text_features.detach(), ratio)
        self.text_labels = text_labels.cuda()
        
    def forward(self, f, label):
        fn = f / f.norm(dim=-1,keepdim=True)
        inp = torch.cat((self.text_features, fn),dim=0)
        A_all = gen_Aall(self.Atext, self.text_labels, label)
        adj = gen_adj(A_all).detach()

        g = self.gnn_gc1(inp, adj)
        g = self.gnn_relu1(g)
        y = self.gnn_gc2(g, adj)

        return y[self.n_texts:], g[self.n_texts:]
    
