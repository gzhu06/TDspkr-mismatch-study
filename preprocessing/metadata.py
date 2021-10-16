import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pickle
import random
from torch.utils.data import Dataset, DataLoader
from utils import pickle2array
import pandas as pd
import glob
from tqdm import tqdm

def data2csv(data_folder, subsets=['train', 'val'], data_suffix='/*.pkl'):
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
    data2csv(data_folder='/storageNVME/ge/voxceleb2noisy100k/', subsets=['train'])