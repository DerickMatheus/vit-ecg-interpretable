import h5py
from collections import OrderedDict
import torch
import pandas as pd
import numpy as np
import neurokit2 as nk
from torch.utils.data import Dataset

class CODE2Dataloader(Dataset):
    def __init__(self, h5_file, path_to_csv, traces_dset='signal', exam_id_dset='id_exam',
                 reg_dset='register_num', train=True):
#         f = h5py.File(path_to_h5, 'r')
        f = h5_file
        df = pd.read_csv(path_to_csv)
        csv_index = df['id_exame']

        # Save missing ids

        # Set index
        df = df.set_index('id_exame')
        # save h5 values
        self.f = f
        self.traces = f[traces_dset]
        self.exam_ids = np.array(f[exam_id_dset])
        self.csv_exam_ids = csv_index
        self.reg = np.array(f[reg_dset])
        self.missing_ids_in_hdf5 = ~np.isin(self.exam_ids, df.index)
        # save csv values
        df = df.reindex(self.exam_ids, fill_value=False, copy=True)
        if train:
            self.val = np.array(df['validation']) & (~self.missing_ids_in_hdf5)
            self.train = (~self.val) & ~self.missing_ids_in_hdf5
        else:
            self.train = np.array(df['validation']) & (~self.missing_ids_in_hdf5)
            
        self.train=np.where(self.train)[0]
        del df['validation']
        df = df[['FA', 'BAV1o', 'BRD', 'BRE', 'Bradi', 'Taqui']]
        df = df.set_index(self.reg, append=True)  # add register as a second index
        df.index.names = ['id_exame', 'reg']
        self.outcomes = df
        self.error = 0
        self.features = 0
            
    def __getitem__(self, idx):
        new_idx = self.train[idx]
        X=self.traces[new_idx]
        traces = np.empty(shape = (self.traces.shape[0], 1024))
        try:
            segment = nk.ecg_peaks(X[0], sampling_rate=400)
            Rpeak = segment[1]['ECG_R_Peaks'][1]
            if Rpeak < 4096 - 1024:
                traces = X[:,Rpeak:Rpeak+1024]
            else:
                traces = X[:,0:1024]
        except:
            traces = X[:,0:1024]
        X=torch.tensor(traces)
        Y=torch.tensor(self.outcomes.iloc[new_idx].values, dtype=torch.float32)
#         Y=torch.from_numpy(self.y.loc[self.y['id_exame'] == self.exam_ids[ind]].values[:,1:-1].astype(bool)).int().squeeze()
        return X,Y
        
    def __len__(self):
        return len(self.train)

class CODE2Dataset:
    def __init__(self, path_to_h5, path_to_csv, traces_dset='signal', exam_id_dset='id_exam',
                 reg_dset='register_num'):
        f = h5py.File(path_to_h5, 'r')
        df = pd.read_csv(path_to_csv)
        csv_index = df['id_exame']

        # Save missing ids

        # Set index
        df = df.set_index('id_exame')
        # save h5 values
        self.f = f
        self.traces = f[traces_dset]
        self.exam_ids = np.array(f[exam_id_dset])
        self.csv_exam_ids = csv_index
        self.reg = np.array(f[reg_dset])
        self.missing_ids_in_hdf5 = ~np.isin(self.exam_ids, df.index)
        # save csv values
        df = df.reindex(self.exam_ids, fill_value=False, copy=True)
        self.val = np.array(df['validation']) & (~self.missing_ids_in_hdf5)
        self.train = (~self.val) & ~self.missing_ids_in_hdf5
        del df['validation']
        df = df[['FA', 'BAV1o', 'BRD', 'BRE', 'Bradi', 'Taqui']]
        df = df.set_index(self.reg, append=True)  # add register as a second index
        df.index.names = ['id_exame', 'reg']
        self.outcomes = df
        self.error = 0
        self.features = 0
        
    def addfeatures(self, features):
        self.features = features

    def getbatch(self, start=0, end=None, seq_size=1024):
        if end is None:
            end = len(self)
        
        #align signal start
        traces = np.empty(shape = (end-start, self.traces.shape[1], seq_size))
        for i,x in enumerate(self.traces[start:end]):
            try:
                segment = nk.ecg_peaks(x[0], sampling_rate=400)
                Rpeak = segment[1]['ECG_R_Peaks'][1]
                traces[i] = x[:,Rpeak:Rpeak+seq_size]
            except:
                traces[i] = x[:,0:seq_size]
                self.error += 1

        return traces, self.outcomes.iloc[start:end].values
#         return self.traces[start:end], self.outcomes.iloc[start:end].values

    def get_features_batch(self, start=0, end=None):
        return self.features[start:end], self.oucomes.iloc[start:end]
#    def __del__(self):
#        self.f.close()

    def __len__(self):
        return len(self.outcomes.index[~self.missing_ids_in_hdf5])


class BatchDataloader:
    def __init__(self, dset, batch_size, features=False, mask=None):
        nonzero_idx, = np.nonzero(mask)
        self.dset = dset
        self.batch_size = batch_size
        self.mask = mask
        self.features = features
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx)
            self.end_idx = max(nonzero_idx)+1
        else:
            self.start_idx = 0
            self.end_idx = 0

    def __next__(self):
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        batch_mask = self.mask[self.start:end]
        while sum(batch_mask) == 0:
            self.start = end
            end = min(self.start + self.batch_size, self.end_idx)
            batch_mask = self.mask[self.start:end]
        if self.features:
            batch = self.dset.get_features_batch(self.start, end)
        else:
            batch = self.dset.getbatch(self.start, end)
        self.start = end
        self.sum += sum(batch_mask)
        return [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch]

    def __iter__(self):
        self.start = self.start_idx
        self.sum = 0
        return self

    def __len__(self):
        count = 0
        start = self.start_idx
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)
            batch_mask = self.mask[start:end]
            if sum(batch_mask) != 0:
                count += 1
            start = end
        return count


if __name__ == "__main__":
    dset = CODE2Dataset(
        path_to_h5='../data/examples.h5',
        path_to_csv='../data/classification_data.csv'
    )

    mask = np.zeros(len(dset), dtype=bool)
    mask[:50] = True
    mask[[88, 90, 92]] = True
    loader = BatchDataloader(dset, 10, mask)
    for x, y in loader:
        print(x.shape)
        print(y.shape)


