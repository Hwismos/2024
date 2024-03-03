from torch.nn import Sequential, Linear, ReLU
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from dgl.nn import EdgeWeightNorm
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from utils import *

# choi
from collections import defaultdict
from torch.nn.parameter import Parameter

EOS = 1e-10
norm = EdgeWeightNorm(norm='both')


class GCL(nn.Module):
    def __init__(self, nlayers, nlayers_proj, in_dim, emb_dim, proj_dim, dropout, sparse, batch_size, nnodes):
        super(GCL, self).__init__()

        self.encoder1 = SGC(nlayers, in_dim, emb_dim, dropout, sparse, nnodes)
        self.encoder2 = SGC(nlayers, in_dim, emb_dim, dropout, sparse, nnodes)

        if nlayers_proj == 1:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim))
        elif nlayers_proj == 2:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))

        self.batch_size = batch_size


    def get_embedding(self, x, a1, a2, adj_unnorm, source='all'):
        emb1 = self.encoder1(x, a1, adj_unnorm)
        emb2 = self.encoder2(x, a2, adj_unnorm)
        return torch.cat((emb1, emb2), dim=1)


    def get_projection(self, x, a1, a2):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        proj1 = self.proj_head1(emb1)
        proj2 = self.proj_head2(emb2)
        return torch.cat((proj1, proj2), dim=1)


    def forward(self, x1, a1, x2, a2, adj_unnrom):
        emb1 = self.encoder1(x1, a1, adj_unnrom)
        emb2 = self.encoder2(x2, a2, adj_unnrom)
        proj1 = self.proj_head1(emb1)
        proj2 = self.proj_head2(emb2)
        loss = self.batch_nce_loss(proj1, proj2)
        return loss


    def set_mask_knn(self, X, k, dataset, metric='cosine'):
        if k != 0:
            path = '/home/hwiric/2024/projects/240225_UDT_extend/code/data/knn/{}'.format(dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = path + '/{}_{}.npz'.format(dataset, k)
            if os.path.exists(file_name):
                knn = sparse.load_npz(file_name)
                # print('Load exist knn graph.')
            else:
                print('Computing knn graph...')
                knn = kneighbors_graph(X, k, metric=metric)
                sparse.save_npz(file_name, knn)
                print('Done. The knn graph is saved as: {}.'.format(file_name))
            knn = torch.tensor(knn.toarray()) + torch.eye(X.shape[0])
        else:
            knn = torch.eye(X.shape[0])
        self.pos_mask = knn
        self.neg_mask = 1 - self.pos_mask


    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss


    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.cuda()
        neg_mask = neg_mask.cuda()
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()


class Edge_Discriminator(nn.Module):
    def __init__(self, nnodes, input_dim, alpha, sparse, hidden_dim=128, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()

        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)

        self.temperature = temperature
        self.bias = bias
        self.nnodes = nnodes
        self.sparse = sparse
        self.alpha = alpha


    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h


    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2


    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()


    def weight_forward(self, features, edges):
        embeddings = self.get_node_embedding(features)
        edges_weights_raw = self.get_edge_weight(embeddings, edges)
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        weights_hp = 1 - weights_lp
        return weights_lp, weights_hp


    def weight_to_adj(self, edges, weights_lp, weights_hp):
        if not self.sparse:
            adj_lp = get_adj_from_edges(edges, weights_lp, self.nnodes)
            adj_lp += torch.eye(self.nnodes).cuda()
            adj_lp = normalize_adj(adj_lp, 'sym', self.sparse)

            adj_hp = get_adj_from_edges(edges, weights_hp, self.nnodes)
            adj_hp += torch.eye(self.nnodes).cuda()
            adj_hp = normalize_adj(adj_hp, 'sym', self.sparse)

            mask = torch.zeros(adj_lp.shape).cuda()
            mask[edges[0], edges[1]] = 1.
            mask.requires_grad = False
            adj_hp = torch.eye(self.nnodes).cuda() - adj_hp * mask * self.alpha
        else:
            adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_lp = dgl.add_self_loop(adj_lp)
            weights_lp = torch.cat((weights_lp, torch.ones(self.nnodes).cuda())) + EOS
            weights_lp = norm(adj_lp, weights_lp)
            adj_lp.edata['w'] = weights_lp

            adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_hp = dgl.add_self_loop(adj_hp)
            weights_hp = torch.cat((weights_hp, torch.ones(self.nnodes).cuda())) + EOS
            weights_hp = norm(adj_hp, weights_hp)
            weights_hp *= - self.alpha
            weights_hp[edges.shape[1]:] = 1
            adj_hp.edata['w'] = weights_hp
        return adj_lp, adj_hp

    
    # choi
    def get_original_adj(self, edges):
        weights = torch.ones(len(edges[0])).cuda()
        adj_unnorm = get_adj_from_edges(edges, weights, self.nnodes)
        return adj_unnorm


    def forward(self, features, edges):
        weights_lp, weights_hp = self.weight_forward(features, edges)
        adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp)

        # choi
        adj_unnorm = self.get_original_adj(edges)

        # choi
        return adj_lp, adj_hp, weights_lp, weights_hp, adj_unnorm
        
        # return adj_lp, adj_hp, weights_lp, weights_hp


class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout, sparse, nnodes):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sparse = sparse

        self.linear = nn.Linear(in_dim, emb_dim)
        self.k = nlayers

        # choi: UDT
        self.h_mlp = nn.Linear(in_dim, emb_dim)
        self.h_str = nn.Linear(nnodes, emb_dim)
        self.h_acm = nn.Linear(emb_dim*3, emb_dim)  

        # choi: attention
        self.init_attention(emb_dim)

    
    # choi
    def init_attention(self, emb_dim):
        
        self.att_vec_11 = nn.Linear(2*emb_dim, 1)
        self.att_vec_12 = nn.Linear(2*emb_dim, 1)
        self.att_vec_13 = nn.Linear(2*emb_dim, 1)


    # choi
    def feat_and_structure_information(self, x, adj_unnorm):
        
        y1 = torch.relu(self.h_mlp(x))
        y2 = torch.relu(self.h_str(adj_unnorm))
        # y = torch.cat((y1, y2), dim=1)
        return y1, y2
    

    def attention(self, x, y1, y2):
        x_mean_vec = x.mean(dim=0)
        y1_mean_vec = y1.mean(dim=0)
        y2_mean_vec = y2.mean(dim=0)
        
        e_11 = torch.exp(
                F.leaky_relu(
                    self.att_vec_11(
                        torch.cat((x_mean_vec, x_mean_vec))
                        )
                    )
                )
        e_12 = torch.exp(
                F.leaky_relu(
                    self.att_vec_12(
                        torch.cat((x_mean_vec, y1_mean_vec))
                        )
                    )
                )
        e_13 = torch.exp(
                F.leaky_relu(
                    self.att_vec_13(
                        torch.cat((x_mean_vec, y2_mean_vec))
                        )
                    )
                )

        denominator = e_11 + e_12 + e_13
        self.att_11 = e_11 / denominator
        self.att_12 = e_12 / denominator
        self.att_13 = e_13 / denominator

        x = torch.sigmoid(self.att_11*x + self.att_12*y1 + self.att_13*y2)
        return x


    def forward(self, x, g, adj_unorm):
        # choi 
        y1, y2 = self.feat_and_structure_information(x, adj_unorm)
        y = torch.cat((y1, y2), dim=1)

        x = torch.relu(self.linear(x))
        if self.sparse:
            with g.local_scope():
                g.ndata['h'] = x
                for _ in range(self.k):
                    g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))

                # original
                # x = g.ndata['h']
                # return g.ndata['h']
                    
                # choi: UDT
                # x = torch.cat((g.ndata['h'], y), dim=1)
                # x = self.h_acm(x)
                    
                x = g.ndata['h']
                x = self.attention(x, y1, y2)
                return x  # if actor, it returns here
        else:
            for _ in range(self.k):
                x = torch.matmul(g, x)
            
            # choi: UDT
            # x = torch.cat((x, y), dim=1)
            # x = self.h_acm(x)
            x = self.attention(x, y1, y2)
            return x
