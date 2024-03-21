from datetime import datetime
import os

class Logger():
    def __init__(self) -> None:
        pass

    def file_name(self):
        return self.path + '/' + self.cur_time

    def time_str(self):
        now = datetime.now()

        year = now.strftime("%Y")[2:]
        month = now.strftime("%m")
        day = now.strftime("%d")
        time = now.strftime("%H:%M:%S")

        self.cur_time = year+month+day+'_'+time

    def path_str(self):
        idx = self.cur_time.find('_')
        self.path = os.path.abspath(os.getcwd()) + '/out/' + self.cur_time[:idx]
        # print(f'\033[0;30;46m{path}\033[0m')

        if not os.path.exists(self.path): 
            os.makedirs(self.path)
    
    # def init_loss_dict(self):
    #     self.train_loss_main = {'GNN Loss': [], 'MLP Loss': []}

    def train_print(self, run, epoch, gnnloss, mlploss):
        # self.train_loss_sub = {'Hetero Loss': [], 'Reg Loss': [], 'L2 Loss': []}
        # self.train_loss_main['GNN Loss'].append(gnnloss)
        # self.train_loss_main['MLP Loss'].append(mlploss)
        print("[TRAIN] Run: {} | Epoch:{:04d} | GNN Loss {:.4f} | MLP loss:{:.4f} ".format(run, epoch, gnnloss, mlploss))

    def test_print(self, run, best_micro):
        print("[TEST] Run:{} | Best Micro: {:.4f}".format(run, best_micro))
    
    def final_print(self, mic_mean, mic_std, mac_mean, mac_std, args):
        print('#' * 100)
        print('[Best Test Micro F1]: {:.4f}±{:.4f}'.format(mic_mean, mic_std))
        print('[Best Test Macro F1]: {:.4f}±{:.4f}'.format(mac_mean, mac_std))
        print('[Arguments Representation]: ', args.__repr__())
        print('#' * 100)
        print()


