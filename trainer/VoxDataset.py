import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pickle
import random
from torch.utils.data import Dataset, DataLoader
from utils import pickle2array
from trainer.train_utils import clipped_feature
import pandas as pd
import glob
from tqdm import tqdm
import librosa

# class VoxDataset(Dataset):
    
#     def __init__(self, filepaths, labels_to_id, transform=None):
#         """
#         Args:
#             filepaths: list of Paths to the pickle files with annotations.
#             root_dir (string): Directory with all the pickles.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.filepaths = filepaths
#         self.transform = transform
#         self.labels_to_id = labels_to_id
        
#     def __len__(self):
#         return len(self.filepaths)
    
#     def __getitem__(self, idx):
        
#         try:
#             x_ = pickle2array(self.filepaths[idx])
#         except (OSError, pickle.UnpicklingError):
#             print('file error!')
        
#         x = clipped_feature(np.expand_dims(x_, axis=0))
#         last = self.filepaths[idx].split("/")[-1]
#         y = self.labels_to_id[last.split("-")[0]]

#         sample = {'audio': x, 'identity': y}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample

class VoxDataset(Dataset):

    def __init__(self, data_path, labels_to_id, seg_len=3.0, sr=16000, transform=None):
        self.data_path = data_path
        if isinstance(data_path, list):
            self.csv_path = []
            for p in data_path:
                self.csv_path.append(os.path.join(p, 'metadata.csv'))
        else:
            self.csv_path = os.path.join(data_path, 'metadata.csv')

        self.labels_to_id = labels_to_id
        self.seg_len = seg_len
        self.frame_num = int(seg_len*sr)
        self.transform = transform
        # Open csv file
        if isinstance(self.csv_path, list):
            self.df = pd.read_csv(self.csv_path[0])
            self.df['data_path'] = data_path[0]
            for i in range(1, len(self.csv_path)):
                df_temp = pd.read_csv(self.csv_path[i])
                df_temp['data_path'] = data_path[i]
                self.df = pd.concat([self.df, df_temp], axis=0, ignore_index=True)
        else:
            self.df = pd.read_csv(self.csv_path)
            self.df['data_path'] = data_path
        max_len = len(self.df)
        if self.frame_num is not None:
            self.df = self.df[self.df['frame_num'] >= self.frame_num]
            print(f"Drop {max_len - len(self.df)} utterances from {max_len} "
                  f"(shorter than {self.frame_num} frames)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get data path
        file_path = os.path.join(row['data_path'], row['file_name'])
        try:
            x_ = pickle2array(file_path)
            x_ = librosa.util.normalize(x_)
        except (OSError, pickle.UnpicklingError):
            print('file error!')
            
        while True:
            x = clipped_feature(np.expand_dims(x_, axis=0), self.seg_len)
            
            if np.abs(x).sum() > 0.1:
                break

        y = self.labels_to_id[row['spk_id']]

        sample = {'audio': x.astype(np.double), 'identity': y}
        if self.transform:
            sample = self.transform(sample)

        return sample

def data2csv(data_folder, subsets=['val', 'train'], data_suffix='/*.pkl'):
    for oneset in subsets:
        subset_folder = os.path.join(data_folder, oneset)
        spk_ids = []
        frame_num = []
        file_names = []
        files = glob.glob(subset_folder + data_suffix)
        for f in tqdm(files, desc='Parsing {} set'.format(oneset)):
            tf_data = pickle2array(f)
            f_name = f.split('/')[-1]
            file_names.append(f_name)
            spk_ids.append(f_name.split('-')[0])
            frame_num.append(tf_data.shape[-1])
        dataframe = pd.DataFrame({'file_name': file_names, 'spk_id': spk_ids, 'frame_num': frame_num})
        dataframe.to_csv(os.path.join(subset_folder, "metadata.csv".format(oneset)), index=False)

if __name__ == '__main__':
    # data2csv(data_folder='/storageNVME/fei/data/speech/VoxCeleb1')
    data2csv(data_folder='/storageNVME/ge/voxceleb2/')