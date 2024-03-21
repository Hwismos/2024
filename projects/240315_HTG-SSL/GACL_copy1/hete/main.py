import sys
sys.path.append("..")
from hete import utils
import seaborn as sb
import torch
from hete.model import Model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from layers import LogReg
from torch.utils.data import Dataset
import torch.nn.functional as F
import statistics
import argparse
import random

import choi
import dgl


class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    

# Training
# choi: 인자 변경
def train(adj1, g, feats, x2, lr, epoch, device, lambda_loss, hidden_dim, sample_size=10,moving_average_decay=0.0):
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())
    in_dim = feats.shape[1]
    GNNModel = Model(in_dim, hidden_dim, 2, sample_size, lambda_loss=lambda_loss, 
                     tau=args.tau,  moving_average_decay=moving_average_decay, 
                     num_MLP=args.num_MLP)
    GNNModel.to(device)
    opt = torch.optim.Adam([{'params': GNNModel.parameters()}], lr=lr, weight_decay=0.0003)
    for i in range(epoch):
        feats = feats.to(device)

        # choi
        total_loss, gacl_loss, semantic_const_loss, node_embeddings = GNNModel(adj1, g, feats, x2, neighbor_dict, device=device)
        opt.zero_grad()
        total_loss.backward()
        # print(i, loss.item())
        print('[EPOCH]: {:03d}, [TOTAL]: {:.4f}, [GACL]: {:.4f}, [SEMANTIC]: {:.4f}'.format(i, total_loss, gacl_loss, semantic_const_loss))
        opt.step()
        GNNModel.update_moving_average()

    # choi
    return node_embeddings.cpu().detach(), total_loss.item(), gacl_loss.item(), semantic_const_loss.item()


def evaluate(model, loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    return (correct / total).item()


def write_results(acc, best_epoch, s_time):
    logger = choi.Logger()
    logger.time_str()
    logger.path_str()
    file_path = logger.path_without_dataset + args.dataset + '.txt'
    print('\033[42m{}\033[0m'.format(file_path))

    best_epoch = [str(tmp) for tmp in best_epoch]
    # f = open("results/" + args.dataset + '_heterSSL', 'a+')
    f = open(file_path, 'w+')
    f.write('[ARGUMENTS]: ' + args.dataset + ' --epochs ' + str(args.epoch_num) + ' --seed ' + str(args.seed) + ' --lr ' + str(args.lr) + ' --lambda_loss ' + str(args.lambda_loss) + ' --moving_average_decay ' + str(args.moving_average_decay) + ' --dimension ' + str(args.dimension) + ' --sample_size ' + str(args.sample_size) + ' --wd2 ' + str(args.wd2) + ' --num_MLP ' + str(args.num_MLP) + ' --tau ' + str(args.tau) + ' --best_epochs ' + " ".join(best_epoch) + '\n')
    f.write('[FINAL TEST]: {:.4f} ± {:.4f}\n'.format(np.mean(acc), np.std(acc)))
    e_time = choi.end_time()
    f.write('[TOTAL TIME]: {}\n'.format(choi.elaspsed_time(s_time, e_time)))
    f.close()

# choi
import copy
def feat_drop(x, p=0.2):
    # copy x
    x = copy.deepcopy(x)
    mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x[:, mask] = 0

    return x


def train_new_datasets(dataset_str, epoch_num = 10, lr = 5e-6, lambda_loss=1, sample_size=10, hidden_dim=None,moving_average_decay=0.0):
    s_time = choi.start_time()
    g, labels, split_lists, adj1 = utils.read_real_datasets(dataset_str)  # choi: adj1 추가
    g = g.to(device)  # choi: adj2
    node_features = g.ndata['attr']  # choi: x1

    # choi
    adj1 = adj1.to(device)
    x2 = feat_drop(node_features)

    node_labels = labels
    # attr, feat
    if hidden_dim == None:
        hidden_dim = node_features.shape[1]
    else:
        hidden_dim = hidden_dim
    acc = []
    epochs = []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    #pre-training

    # choi: 이쪽도 수정
    node_embeddings, total_loss, gacl_loss, semantic_const_loss = train(adj1, g, node_features, x2, lr=lr, epoch=epoch_num, device=device,
                                  lambda_loss=lambda_loss, sample_size=sample_size, hidden_dim=hidden_dim,
                                  moving_average_decay=moving_average_decay)  # choi: adj1, x2 추가
    #evaluation
    for index in range(10):
        input_dims = node_embeddings.shape
        # print(input_dims[1])
        class_number = int(max(node_labels)) + 1
        FNN = LogReg(input_dims[1], class_number).to(device)
        FNN = FNN.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(FNN.parameters(), lr =1e-2, weight_decay=args.wd2)
        dataset = NodeClassificationDataset(node_embeddings, node_labels)
        split = utils.DataSplit(dataset, split_lists[index]['train_idx'], split_lists[index]['valid_idx'], split_lists[index]['test_idx'], shuffle=True)
        train_loader, val_loader, test_loader = split.get_split(batch_size=100000, num_workers=0)
        # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        best = -float('inf')
        best_epoch = 0
        test_acc = 0
        for epoch in range(3000):  # 3000
            for i, data in enumerate(train_loader, 0):
                # data = data.to(device)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_pred = FNN(inputs)
                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = FNN(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        total += labels.size(0)
                        correct += torch.sum(predicted == labels)
                if correct / total > best:
                    best = correct / total
                    test_acc = evaluate(FNN, test_loader)
                    best_epoch = epoch
       
        # print(test_acc)
        acc.append(test_acc)
        epochs.append(best_epoch)
    
    # choi
    print('\033[42m{}{}\033[0m'.format('[MEAN]: ', statistics.mean(acc)))
    print('\033[42m{}{}\033[0m'.format('[STD]: ', statistics.stdev(acc)))
    
    # print("mean:")
    # print(statistics.mean(acc))
    # print("std:")
    # print(statistics.stdev(acc))
    
    print('\033[42m{}\033[0m'.format('Writing on file...'))
    write_results(acc, epochs, s_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default="texas")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--lambda_loss', type=float, default=1)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--dimension', type=int, default=512)
    parser.add_argument('--moving_average_decay', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--wd2', type=float, default=1e-2)
    parser.add_argument('--num_MLP', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2014)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    dataset_str = args.dataset
    print('\033[42m{}\033[0m'.format(args))
    train_new_datasets(dataset_str=dataset_str, lr=args.lr, epoch_num=args.epoch_num, lambda_loss=args.lambda_loss, sample_size=args.sample_size, hidden_dim=args.dimension,moving_average_decay=args.moving_average_decay)
