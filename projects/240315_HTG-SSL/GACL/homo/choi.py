from datetime import datetime, timedelta
import time
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
        # print('\033[41m{}\033[0m'.format('DONE'))

        if not os.path.exists(self.path): 
            os.makedirs(self.path)
        
        self.path_without_dataset = self.path + '/' + self.cur_time + '_'
    

def start_time():
    return time.time()


def end_time():
    return time.time() 

def elaspsed_time(s, e):
    elaspsed = e - s
    return str(timedelta(seconds=elaspsed))