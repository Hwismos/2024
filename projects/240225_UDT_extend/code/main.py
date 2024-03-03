import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random

from data_loader import load_data
from model import *
from utils import *

import sys

EOS = 1e-10

# choi
from datetime import datetime
import time


# choi
class Logger():
    def __init__(self) -> None:
        pass

    def attention_dict(self):
        self.encoder1_att_dict = {'att_11': [], 'att_12': [], 'att_13': []}
        self.encoder2_att_dict = {'att_11': [], 'att_12': [], 'att_13': []}


    def attention_dict_update(self, encoder1_att, encoder2_att):
        self.encoder1_att_dict['att_11'].append(encoder1_att[0].item())
        self.encoder1_att_dict['att_12'].append(encoder1_att[1].item())
        self.encoder1_att_dict['att_13'].append(encoder1_att[2].item())

        self.encoder2_att_dict['att_11'].append(encoder2_att[0].item())
        self.encoder2_att_dict['att_12'].append(encoder2_att[1].item())
        self.encoder2_att_dict['att_13'].append(encoder2_att[2].item())
    

    def log_attention(self):
        print('#' * 100)
        
        print('[ENCODER 1]')
        print('1. [att_11 AVG]: {:.4f}, [att_11 STD]: {:.4f}'.format(np.mean(self.encoder1_att_dict['att_11']), np.std(self.encoder1_att_dict['att_11'])))
        print('2. [att_12 AVG]: {:.4f}, [att_12 STD]: {:.4f}'.format(np.mean(self.encoder1_att_dict['att_12']), np.std(self.encoder1_att_dict['att_12'])))
        print('3. [att_13 AVG]: {:.4f}, [att_13 STD]: {:.4f}'.format(np.mean(self.encoder1_att_dict['att_13']), np.std(self.encoder1_att_dict['att_13'])))
        print('=' * 10)
        print('4. [att_11 raw]: {}'.format(self.encoder1_att_dict['att_11']))
        print('5. [att_12 raw]: {}'.format(self.encoder1_att_dict['att_12']))
        print('6. [att_13 raw]: {}'.format(self.encoder1_att_dict['att_13']))
        
        print('=' * 60)
        
        print('[ENCODER 2]')
        print('1. [att_11 AVG]: {:.4f}, [att_11 STD]: {:.4f}'.format(np.mean(self.encoder2_att_dict['att_11']), np.std(self.encoder2_att_dict['att_11'])))
        print('2. [att_12 AVG]: {:.4f}, [att_12 STD]: {:.4f}'.format(np.mean(self.encoder2_att_dict['att_12']), np.std(self.encoder2_att_dict['att_12'])))
        print('3. [att_13 AVG]: {:.4f}, [att_13 STD]: {:.4f}'.format(np.mean(self.encoder2_att_dict['att_13']), np.std(self.encoder2_att_dict['att_13'])))
        print('=' * 10)
        print('4. [att_11 raw]: {}'.format(self.encoder2_att_dict['att_11']))
        print('5. [att_12 raw]: {}'.format(self.encoder2_att_dict['att_12']))
        print('6. [att_13 raw]: {}'.format(self.encoder2_att_dict['att_13']))
        print('#' * 100)



    def get_time_info(self):
        now = datetime.now() # current date and time

        year = now.strftime("%Y")[2:]
        month = now.strftime("%m")
        day = now.strftime("%d")
        time = now.strftime("%H:%M:%S")

        time_info_str = year+month+day+'_'+time
        return time_info_str


    # 시간 정보를 이용해서 디렉토리를 만들고, 디렉토리 path를 반환함
    def get_path_info(self, time_info_str):
        idx = time_info_str.find('_')
        path = '/home/hwiric/2024/projects/240225_UDT_extend/out/' + time_info_str[:idx]
        # print(f'\033[0;30;46m{path}\033[0m')

        if not os.path.exists(path): 
            os.makedirs(path)
        
        return path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def train_cl(cl_model, discriminator, optimizer_cl, features, str_encodings, edges):

    cl_model.train()
    discriminator.eval()

    # choi
    # adj_1, adj_2, weights_lp, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
    adj_1, adj_2, weights_lp, _, adj_unnorm = discriminator(torch.cat((features, str_encodings), 1), edges)

    features_1, adj_1, features_2, adj_2 = augmentation(features, adj_1, features, adj_2, args, cl_model.training)

    # choi
    # cl_loss = cl_model(features_1, adj_1, features_2, adj_2)
    cl_loss = cl_model(features_1, adj_1, features_2, adj_2, adj_unnorm)

    optimizer_cl.zero_grad()
    cl_loss.backward()
    optimizer_cl.step()

    return cl_loss.item()


def train_discriminator(cl_model, discriminator, optimizer_disc, features, str_encodings, edges, args):

    cl_model.eval()
    discriminator.train()

    # choi
    # adj_1, adj_2, weights_lp, weights_hp = discriminator(torch.cat((features, str_encodings), 1), edges)
    adj_1, adj_2, weights_lp, weights_hp, adj_unnorm = discriminator(torch.cat((features, str_encodings), 1), edges)

    rand_np = generate_random_node_pairs(features.shape[0], edges.shape[1])
    psu_label = torch.ones(edges.shape[1]).cuda()

    # choi
    # embedding = cl_model.get_embedding(features, adj_1, adj_2)
    embedding = cl_model.get_embedding(features, adj_1, adj_2, adj_unnorm)

    edge_emb_sim = F.cosine_similarity(embedding[edges[0]], embedding[edges[1]])

    rnp_emb_sim_lp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_lp = F.margin_ranking_loss(edge_emb_sim, rnp_emb_sim_lp, psu_label, margin=args.margin_hom, reduction='none')
    loss_lp *= torch.relu(weights_lp - 0.5)

    rnp_emb_sim_hp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim, psu_label, margin=args.margin_het, reduction='none')
    loss_hp *= torch.relu(weights_hp - 0.5)

    rank_loss = (loss_lp.mean() + loss_hp.mean()) / 2

    optimizer_disc.zero_grad()
    rank_loss.backward()
    optimizer_disc.step()

    return rank_loss.item()


# choi
class DisriminatorEval():
    def __init__(self) -> None:
        self.disc_acc_results = []
        self.disc_hm_acc_results = []
        self.disc_ht_acc_results = []

        self.disc_acc_cnt_results = []
        self.disc_hm_acc_cnt_results = []
        self.disc_ht_acc_cnt_results = [] 


    def final_log(self, dataset, ntrials, start_time):

        print('#' * 100)
        print('1. [FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.4f}+-({:.4f})'.format(dataset, ntrials, np.mean(self.results), np.std(self.results)))
        print('2. [FINAL DISC ACC RESULT] ACC:{:.4f}+-({:.4f})'.format(np.mean(self.disc_acc_results), np.std(self.disc_acc_results)))
        print('3. [FINAL HOMO ACC: {:.4f} | HETERO ACC: {:.4f}]'.format(np.mean(self.disc_hm_acc_results), np.mean(self.disc_ht_acc_results)))
        print('4. [FINAL DISC CNT ACC RESULT] ACC:{:.4f}+-({:.4f})'.format(np.mean(self.disc_acc_cnt_results), np.std(self.disc_acc_cnt_results)))
        print('5. [FINAL HOMO CNT ACC: {:.4f}/({}) | HETERO CNT ACC: {:.4f}/({})]'.format(np.mean(self.disc_hm_acc_cnt_results), int(self.hm_edges * np.mean(self.disc_hm_acc_cnt_results)), np.mean(self.disc_ht_acc_cnt_results), int(self.ht_edges * np.mean(self.disc_ht_acc_cnt_results))))
        
        end_time = time.time()
        print('6. TOTAL_TIME: {:.4f}s'.format(end_time - start_time))        
        print('#' * 100)



    def trial_log(self, results, trial):

        self.results = results
        trial_res = self.results[trial]
        hm_score = self.disc_hm_acc_results[trial]
        ht_score = self.disc_ht_acc_results[trial]
        total_score = self.disc_acc_results[trial]
        hm_cnt = self.disc_hm_acc_cnt_results[trial] 
        ht_cnt = self.disc_ht_acc_cnt_results[trial]
        total_cnt = self.disc_acc_cnt_results[trial] 

        print('#' * 100)
        print(str(trial+1) + '-1. [TRIAL RESULT] ACC: {:.4f}'.format(trial_res))
        print(str(trial+1) + '-2. [TRIAL DISC ACC RESULT] TOTAL ACC: {:.4f}'.format(total_score))
        print(str(trial+1) + '-3. [TRIAL HOMO ACC: {:.4f} | HETERO ACC: {:.4f}]'.format(hm_score, ht_score))
        print(str(trial+1) + '-4. [TRIAL DISC CNT ACC RESULT] TOTAL ACC: {:.4f}'.format(total_cnt))
        print(str(trial+1) + '-5. [TRIAL HOMO CNT ACC: {:.4f}/({}) | HETERO CNT ACC: {:.4f}/({})]'.format(hm_cnt, int(self.hm_edges * hm_cnt), ht_cnt, int(self.ht_edges * ht_cnt)))
        print('#' * 100)


    def update(self, hm_score, ht_score, total_score, 
                     hm_cnt, ht_cnt, total_cnt,
                     hm_edges, ht_edges):
        
        self.hm_score = hm_score
        self.ht_score = ht_score
        self.total_score = total_score
        self.hm_cnt = hm_cnt
        self.ht_cnt = ht_cnt
        self.total_cnt = total_cnt
        self.hm_edges = hm_edges
        self.ht_edges = ht_edges

        self.disc_hm_acc_results.append(hm_score)
        self.disc_ht_acc_results.append(ht_score)
        self.disc_acc_results.append(total_score)

        self.disc_hm_acc_cnt_results.append(hm_cnt)
        self.disc_ht_acc_cnt_results.append(ht_cnt)
        self.disc_acc_cnt_results.append(total_cnt)    


    def get_edges_idx_with_hm_or_ht_info(self, edges_idx, labels):
        src_nodes = edges_idx[0].detach().cpu().numpy()
        dest_nodes = edges_idx[1].detach().cpu().numpy()
        edges_idx_with_hm_or_ht_info = []

        edges_num = len(src_nodes)
        for edge_idx in range(0, edges_num):
            class_of_src_node = labels[src_nodes[edge_idx]]
            class_of_dest_node = labels[dest_nodes[edge_idx]]

            if class_of_src_node == class_of_dest_node:
                wheter_homo = True
            else:
                wheter_homo = False
            
            edge_idx_with_hm_or_ht_info = (src_nodes[edge_idx], dest_nodes[edge_idx], wheter_homo)
            edges_idx_with_hm_or_ht_info.append(edge_idx_with_hm_or_ht_info)

        return edges_idx_with_hm_or_ht_info


    def get_weight_score(self, wlp_dense, edges_idx_with_hm_or_ht_info):
        wlp_dense = wlp_dense.detach().cpu().numpy()
        weight_score = 0.0
        hm_edge_indices = []
        ht_edge_indices = []

        for unit in edges_idx_with_hm_or_ht_info:
            i, j, whether_hm = unit[0], unit[1], unit[2]
            
            if whether_hm == True:
                hm_edge_indices.append((i, j))
            else:
                ht_edge_indices.append((i, j))

        wlp_cnt = 0
        for pair in hm_edge_indices:
            i, j = pair[0], pair[1]
            weight_score += wlp_dense[i][j]
            if wlp_dense[i][j] > 0.5:
                wlp_cnt += 1

        wlp_score = weight_score
        weight_score = 0.0

        whp_cnt = 0
        for pair in ht_edge_indices:
            i, j = pair[0], pair[1]
            weight_score += (1- wlp_dense[i][j])
            if (1- wlp_dense[i][j]) > 0.5:
                whp_cnt += 1
        whp_score = weight_score
        
        return (wlp_score, whp_score, wlp_cnt, whp_cnt)


    def get_hm_or_ht_edges_num(self, edges_idx_with_hm_or_ht_info):
        hm_edge_indices = []
        ht_edge_indices = []

        for unit in edges_idx_with_hm_or_ht_info:
            i, j, whether_hm = unit[0], unit[1], unit[2]
            
            if whether_hm == True:
                hm_edge_indices.append((i, j))
            else:
                ht_edge_indices.append((i, j))
        
        hm_edges_num = len(hm_edge_indices)
        ht_edges_num = len(ht_edge_indices)

        return (hm_edges_num, ht_edges_num)


    def get_discriminator_performance_accuracy(self, edges_idx, wlp, whp, labels):
        edges_idx_with_hm_or_ht_info = self.get_edges_idx_with_hm_or_ht_info(edges_idx, labels)

        nnodes = len(labels)
        # from utils.py
        wlp_dense = get_adj_from_edges(edges_idx, wlp, nnodes)

        wlp_score_raw, whp_score_raw, wlp_cnt_raw, whp_cnt_raw = self.get_weight_score(wlp_dense, edges_idx_with_hm_or_ht_info)
        hm_edges_num, ht_edges_num = self.get_hm_or_ht_edges_num(edges_idx_with_hm_or_ht_info)

        # scroe
        wlp_score = wlp_score_raw / hm_edges_num
        whp_score = whp_score_raw / ht_edges_num

        # cnt
        wlp_cnt = wlp_cnt_raw / hm_edges_num
        whp_cnt = whp_cnt_raw / ht_edges_num    
        
        total_score = (wlp_score_raw + whp_score_raw) / (hm_edges_num + ht_edges_num)
        total_cnt = (wlp_cnt_raw + whp_cnt_raw) / (hm_edges_num + ht_edges_num)
        
        discriminator_performance_accuracy = (wlp_score, whp_score, total_score, 
                                            wlp_cnt, whp_cnt, total_cnt, 
                                            hm_edges_num, ht_edges_num
                                            )

        return discriminator_performance_accuracy


def main(args):

    setup_seed(0)
    features, edges, str_encodings, train_mask, val_mask, test_mask, labels, nnodes, nfeats = load_data(args.dataset)
    results = []

    # choi
    logger = Logger()
    disc_evaluator = DisriminatorEval()
    logger.attention_dict()

    # choi
    time_info = logger.get_time_info()
    dir_path = logger.get_path_info(time_info)
    file_name = time_info + '_' + str(args.dataset)
    origin_sys_stdout = sys.stdout

    # choi
    start_time = time.time()

    # choi
    print('\n...')
    with open(dir_path + '/' + file_name + '.txt', 'w') as sys.stdout:
        print(args)
        print('=============================================================\n=============================================================\n')        
        for trial in range(args.ntrials):

            setup_seed(trial)

            cl_model = GCL(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim,
                        proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse, batch_size=args.cl_batch_size, 
                        nnodes=nnodes).cuda()
            cl_model.set_mask_knn(features.cpu(), k=args.k, dataset=args.dataset)
            discriminator = Edge_Discriminator(nnodes, nfeats + str_encodings.shape[1], args.alpha, args.sparse).cuda()

            optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)
            optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)

            features = features.cuda()
            str_encodings = str_encodings.cuda()
            edges = edges.cuda()

            best_acc_val = 0
            best_acc_test = 0

            # choi
            best_wlp_view = None

            for epoch in range(1, args.epochs + 1):

                for _ in range(args.cl_rounds):
                    cl_loss = train_cl(cl_model, discriminator, optimizer_cl, features, str_encodings, edges)
                rank_loss = train_discriminator(cl_model, discriminator, optimizer_discriminator, features, str_encodings, edges, args)

                print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} | RANK loss:{:.4f} ".format(epoch, cl_loss, rank_loss))

                if epoch % args.eval_freq == 0:
                    cl_model.eval()
                    discriminator.eval()

                    # adj_1, adj_2, _, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
                    
                    # choi 
                    # adj_1, adj_2, wlp, whp = discriminator(torch.cat((features, str_encodings), 1), edges)
                    adj_1, adj_2, wlp, whp, adj_unnorm = discriminator(torch.cat((features, str_encodings), 1), edges)

                    # choi
                    # embedding = cl_model.get_embedding(features, adj_1, adj_2)
                    embedding = cl_model.get_embedding(features, adj_1, adj_2, adj_unnorm)

                    cur_split = 0 if (train_mask.shape[1]==1) else (trial % train_mask.shape[1])
                    acc_test, acc_val = eval_test_mode(embedding, labels, train_mask[:, cur_split],
                                                    val_mask[:, cur_split], test_mask[:, cur_split])
                    print(
                        '[TEST] Epoch:{:04d} | CL loss:{:.4f} | RANK loss:{:.4f} | VAL ACC:{:.2f} | TEST ACC:{:.2f}'.format(
                            epoch, cl_loss, rank_loss, acc_val, acc_test))

                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        best_acc_test = acc_test

                        # choi
                        best_wlp = wlp
                        best_whp = whp

            # choi
            disc_acc_score = disc_evaluator.get_discriminator_performance_accuracy(edges, best_wlp, best_whp, labels)
            hm_score, ht_score, total_score, hm_cnt, ht_cnt, total_cnt, hm_edges, ht_edges = disc_acc_score
            disc_evaluator.update(hm_score, ht_score, total_score, hm_cnt, ht_cnt, total_cnt, hm_edges, ht_edges)

            # choi
            encoder1_att = [cl_model.encoder1.att_11, cl_model.encoder1.att_12, cl_model.encoder1.att_13]
            encoder2_att = [cl_model.encoder2.att_11, cl_model.encoder2.att_12, cl_model.encoder2.att_13]

            logger.attention_dict_update(encoder1_att, encoder2_att)
            results.append(best_acc_test)
            disc_evaluator.trial_log(results, trial)
            
        # choi
        disc_evaluator.final_log(args.dataset, args.ntrials, start_time)
        logger.log_attention()

    # choi    
    sys.stdout = origin_sys_stdout
    print('DONE!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ESSENTIAL
    parser.add_argument('-dataset', type=str, default='squirrel',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell',
                                'texas', 'wisconsin', 'computers', 'photo', 'cs', 'physics', 'wikics'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-eval_freq', type=int, default=20)
    parser.add_argument('-epochs', type=int, default=400)
    parser.add_argument('-lr_gcl', type=float, default=0.001)
    parser.add_argument('-lr_disc', type=float, default=0.001)
    parser.add_argument('-cl_rounds', type=int, default=2)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.5)

    # DISC Module - Hyper-param
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-margin_hom', type=float, default=0.5)
    parser.add_argument('-margin_het', type=float, default=0.5)

    # GRL Module - Hyper-param
    parser.add_argument('-nlayers_enc', type=int, default=2)
    parser.add_argument('-nlayers_proj', type=int, default=1, choices=[1, 2])
    parser.add_argument('-emb_dim', type=int, default=128)
    parser.add_argument('-proj_dim', type=int, default=128)
    parser.add_argument('-cl_batch_size', type=int, default=0)
    
    parser.add_argument('-k', type=int, default=0)
    # parser.add_argument('-k', type=int, default=10)
    # parser.add_argument('-k', type=int, default=20)
    # parser.add_argument('-k', type=int, default=30)
    
    parser.add_argument('-maskfeat_rate_1', type=float, default=0.1)
    parser.add_argument('-maskfeat_rate_2', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_1', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_2', type=float, default=0.1)

    args = parser.parse_args()

    # choi
    # wandb.init(project='GREET')
    # wandb.config.update(args)

    print(args)
    main(args)