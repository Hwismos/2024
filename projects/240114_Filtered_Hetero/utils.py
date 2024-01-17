from datetime import datetime
import os
import dgl
from dgl import add_self_loops
import numpy as np
import torch 
import networkx as nx
from torch.nn import functional as F

# conda activate 240112_NHGT; cd /home/hwiric/2024/2024_filtered_heterophily_graph_data_test

class Logger():
    def __init__(self, args) -> None:
        self.args = args
    
    
    def render(self):
        time_data = datetime.now().strftime('%Y%m%d_%H:%M:%S:%f')[2:-3]
        idx = time_data.find('_')
        path = './res/' + self.args['dir'] + '/' + time_data[:idx]
        
        if not os.path.exists(path): 
            os.makedirs(path)
        
        return path + '/' + time_data + '_' + str(self.args['dataset']) + '.' + self.args['format']


class ColorPrint:
    def __init__(self) -> None:
        pass

    def print_bk_blue(self, msg):
        print(f'\033[0;30;46m{msg}\033[0m')

    def print_bk_pink(self, msg):
    	print(f"\033[0;30;41m{msg}\033[0m")


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
        cur_dataset = dataset_names[0]

        data = np.load(os.path.join('/home/hwiric/2024/2024_filtered_heterophily_graph_data_test/yandex/data', 
                                    f'{cur_dataset.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)

        if 'directed' not in cur_dataset:
            graph = dgl.to_bidirected(graph)

        if add_self_loops:
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
    
    def save_dgl_to_nx(self):
        nodes_id = self.graph.nodes().numpy()
        edges_src = self.graph.edges()[0].numpy()
        edges_dst = self.graph.edges()[1].numpy()
        nodes_features = self.node_features.numpy()
        nodes_labels = self.labels.numpy()

        edges = np.array(list(zip(edges_src, edges_dst)))
        src, dst = edges[0]

        G = nx.DiGraph().to_undirected()
        for node_id in nodes_id:
            node_id = int(node_id)
            G.add_node(node_id, features=nodes_features[node_id], label=nodes_labels[node_id])
        for edge in edges:
            src, dst = edge
            G.add_edge(int(src), int(dst))
        
        return G


def main():
    pass


if __name__ == '__main__':
    main()