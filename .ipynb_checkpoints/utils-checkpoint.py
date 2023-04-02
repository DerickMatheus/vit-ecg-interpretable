import random
import numpy as np
import pandas as pd
import torch
import os
import datetime
import json
from torch.utils.tensorboard import SummaryWriter

eps = 1e-6

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def set_output_folder(log_folder, log_subfolder):
    run_folder_name = 'output_' + str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_")
    folder = os.path.join(os.getcwd(), log_folder, log_subfolder, run_folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def save_config(args, folder):
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')


def save_dset(dset, folder):
    train_csv = os.path.join(folder, 'train.csv')
    valid_csv = os.path.join(folder, 'valid.csv')
    dset.outcomes[dset.train].reset_index()[['id_exame', 'reg']].to_csv(train_csv, index=False)
    dset.outcomes[dset.val].reset_index()[['id_exame', 'reg']].to_csv(valid_csv, index=False)



class Logger(object):
    def __init__(self, log_folder, metric_name):
        self.folder = log_folder
        self.logger = SummaryWriter(log_dir=self.folder)
        self.history = pd.DataFrame()
        self.metric_names = metric_name

    def log(self, name, val, iteration):
        self.logger.add_scalar(name, val, iteration)

    def log_all(self, log_data, iteration):
        for name in log_data:
            val = log_data[name]
            # change name for metrics and loss to log in specific sections
            name = 'metrics/' + name if name in self.metric_names else name
            name = 'loss/' + name if name.endswith('loss') else name
            # log individually
            self.log(name, val, iteration)

    def log_valid_pred(self, epoch, valid_pred):
        # save
        with open(os.path.join(self.folder, 'valid_pred_ep{}.npy'.format(epoch)), 'wb') as f:
            np.save(f, valid_pred)

    def init_history(self, columns: list):
        self.history = pd.DataFrame(columns=columns)

    def save_history(self, hist_log, file_name='history.csv'):
        self.history = self.history.append(hist_log, ignore_index=True)  # can only append a dict if ignore_index=True
        self.history.to_csv(os.path.join(self.folder, file_name), index=False)

def net_param_map(index):
    net_filter_size_list = [[64, 196, 320],
                            [64, 128, 196, 256, 320],
                            [64, 128, 128, 196, 256, 256, 320, 512, 512],
                            [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]]
    net_seq_length_list = [[4096, 256, 16],
                            [4096, 1024, 256, 64, 16],
                            [4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
                            [4096, 2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16]]
    sizes = [2,4,8,16]
    if(index in sizes):
        return net_filter_size_list[sizes.index(index)], net_seq_length_list[sizes.index(index)]
