import warnings
import torch
import scipy.sparse as sp
import numpy as np
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.utils import remove_self_loops

warnings.simplefilter("ignore")


# =======================================================================
#! csh
import dgl
import networkx as nx
from torch.nn import functional as F

class MiniTest():
    def __init__(self) -> None:
        pass

    def start(self):
        print(f'\033[0;30;43mTEST STARTS\033[0m')

    def end(self):
        print(f'\033[0;30;43mTEST ENDS\033[0m')
        exit()


class YandexDataLoader():
    def __init__(self) -> None:
        dataset_names = ['chameleon-directed', 'chameleon-filtered-directed', 
                        'squirrel-directed', 'squirrel-filtered-directed',
                        'roman-empire', 'minesweeper',
                        ]
        cur_dataset = dataset_names[5]
        print(f'\033[0;30;44m[{cur_dataset}]\033[0m')

        data = np.load(os.path.join('/home/hwiric/2024/2024_filtered_heterophily_graph_data_test/yandex/data', 
                                    f'{cur_dataset.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features))

        if 'directed' not in cur_dataset:
            graph = dgl.to_bidirected(graph)

        graph = dgl.add_self_loop(graph)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        self.name = cur_dataset
        self.graph = graph
        self.node_features = node_features
        self.labels = labels

        self.train_idx_list = [train_idx for train_idx in train_idx_list]
        self.val_idx_list = [val_idx for val_idx in val_idx_list]
        self.test_idx_list = [test_idx for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'


    def convert_index_mask_to_boolean_mask(self):
        nodes_num = len(self.labels)
        train_bool = torch.zeros((10, nodes_num), dtype=torch.bool)
        val_bool = torch.zeros((10, nodes_num), dtype=torch.bool)
        test_bool = torch.zeros((10, nodes_num), dtype=torch.bool)

        for row in range(10):
            train_bool[row][self.train_idx_list[row]] = True
            val_bool[row][self.val_idx_list[row]] = True
            test_bool[row][self.test_idx_list[row]] = True

        train_bool = train_bool.transpose(0, 1)
        val_bool = val_bool.transpose(0, 1)
        test_bool = test_bool.transpose(0, 1)

        return train_bool, val_bool, test_bool

# =======================================================================


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, num_splits: int = 10):

    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True

        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True

        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True

        trains.append(train_mask.unsqueeze(1))
        vals.append(val_mask.unsqueeze(1))
        tests.append(test_mask.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    val_mask_all = torch.cat(vals, 1)
    test_mask_all = torch.cat(tests, 1)

    return train_mask_all, val_mask_all, test_mask_all


def get_structural_encoding(edges, nnodes, str_enc_dim=16):

    row = edges[0, :].numpy()
    col = edges[1, :].numpy()
    data = np.ones_like(row)

    A = sp.csr_matrix((data, (row, col)), shape=(nnodes, nnodes))
    D = (np.array(A.sum(1)).squeeze()) ** -1.0

    Dinv = sp.diags(D)
    RW = A * Dinv
    M = RW

    SE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(str_enc_dim - 1):
        M_power = M_power * M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
    SE = torch.stack(SE, dim=-1)
    return SE


def load_data(dataset_name):

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name)
    elif dataset_name in ['chameleon']:
        dataset = WikipediaNetwork(path, dataset_name)
    elif dataset_name in ['squirrel']:
        dataset = WikipediaNetwork(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['actor']:
        dataset = Actor(path)
    elif dataset_name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset_name)
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['cs', 'physics']:
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['wikics']:
        dataset = WikiCS(path)

    data = dataset[0]

    #! csh: Yandex와 그래프 데이터 통일
    dataloader_yandex = YandexDataLoader()
    graph = dataloader_yandex.graph

    #? edges = remove_self_loops(data.edge_index)[0]
    src = graph.edges()[0].numpy()
    dst = graph.edges()[1].numpy()
    edges = torch.tensor(np.array([src, dst]))

    #? features = data.x
    features = dataloader_yandex.node_features
    
    [nnodes, nfeats] = features.shape
    nclasses = torch.max(data.y).item() + 1

    #? if dataset_name in ['computers', 'photo', 'cs', 'physics', 'wikics']:
    #?     train_mask, val_mask, test_mask = get_split(nnodes)
    #? else:
    #?     train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    
    train_mask, val_mask, test_mask = dataloader_yandex.convert_index_mask_to_boolean_mask()

    if len(train_mask.shape) < 2:
        train_mask = train_mask.unsqueeze(1)
        val_mask = val_mask.unsqueeze(1)
        test_mask = test_mask.unsqueeze(1)

    #? labels = data.y
    labels = dataloader_yandex.labels

    path = 'GREET/data/se/{}'.format(dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + '/{}_{}.pt'.format(dataset_name, 16)
    
    #? if os.path.exists(file_name):
    #?     se = torch.load(file_name)
    #?     # print('Load exist structural encoding.')
    #? else:
    #?     print('Computing structural encoding...')
    #?     se = get_structural_encoding(edges, nnodes)
    #?     torch.save(se, file_name)
    #?     print('Done. The structural encoding is saved as: {}.'.format(file_name))

    print('Computing structural encoding Starts!!!')
    se = get_structural_encoding(edges, nnodes)
    print('Computing structural encoding Done!!!')
    # torch.save(se, file_name)
    # print('Done. The structural encoding is saved as: {}.'.format(file_name))

    #? return features, edges, se, train_mask, val_mask, test_mask, labels, nnodes, nfeats
    return features, edges, se, train_mask, val_mask, test_mask, labels, nnodes, nfeats, dataloader_yandex.name



