import sys
import os, glob
sys.path.append(os.getcwd())

import math
import numpy as np
import logging
import argparse
import csv

import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from preprocessing.wavform_extract import data_catalog

import utils

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-config_file", type=str, default='', help="config file")
args = parser.parse_args()

config = utils.Params(args.config_file)

def latest_checkpoint_path(dir_path, regex="model_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    epoch_str = int(x.split('model_')[1].split('_ckpt.pt')[0])
    print(x)
    return x, epoch_str

def clipped_feature(x, seg_len=config.seg_len):
    frame_length = int(seg_len*config.sr)
    if x.shape[-1] > frame_length:
        bias = np.random.randint(0, x.shape[1] - frame_length)
        clipped_x = x[:, bias: frame_length + bias]
    else:
        clipped_x = x

    return clipped_x

def split_data(files, labels, batch_size):
    # test_size = max(batch_size/len(labels), 0.05)
    test_size = 0.01
    train_paths, test_paths, y_train, y_test = train_test_split(files, labels, test_size=test_size, random_state=42)
    return train_paths, test_paths

def create_dict(files, labels, spk_uniq):
    train_dict = {}

    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []

    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])

    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    unique_speakers=list(train_dict.keys())
    return train_dict, unique_speakers

def npy2file(ipt_dir):
    
    # data importing
    logging.info('== Looking for fbank features [.npy] files in {}.'.format(ipt_dir))
    libri = data_catalog(ipt_dir)
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    
    files = list(libri['filename'])
    spk_ids = list(libri['speaker_id'])
    
    return files, spk_ids

def npy2dict(ipt_dir):
    # data importing
    logging.info('== Looking for fbank features [.npy] files in {}.'.format(ipt_dir))
    libri = data_catalog(ipt_dir)
    num_speakers = len(libri['speaker_id'].unique())
    print('== Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))

    unique_speakers = libri['speaker_id'].unique()
    spk_utt_dict, unique_speakers = create_dict(libri['filename'].values,libri['speaker_id'].values,unique_speakers)

    return spk_utt_dict, unique_speakers

def accuracy(true, pred):
    # true and pred are both a torch tensor
    pred = pred.data.max(1, keepdim=True)[1]
    correct = pred.eq(true.data.view_as(pred)).cpu().sum().numpy()
    return correct * 1.0 / config.batch_size

def val_loss(classifier_model, wav_embed, spk_embed, dataloader_val, train_steps):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    losses = []
    accs = []
    
    with torch.no_grad():
        
        for i_batch, sample_batched in enumerate(dataloader_val):

            # one batch of training data
            data, target = sample_batched['audio'].to(device), sample_batched['identity'].to(device)
            
            # get loss
            utt_embeddings = spk_embed(wav_embed(data.float()))
                
            loss, softmax_output = classifier_model(utt_embeddings, target) 

            acc = accuracy(target, softmax_output)

            losses.append(loss.mean().item())
            accs.append(acc)

        loss_val = np.mean(np.array(losses))
        acc_val = np.mean(np.array(accs))
        
        with open(os.path.join(config.train_path, 'val_loss.txt'), "a") as f:
            f.write("{0},{1},{2}\n".format(train_steps, loss_val, acc_val))

        return loss_val, acc_val

# def val_step(model, embed_model, testloader, trial_train, trial_val, best_eer, train_steps):
def val_step(model, wav_embed, spk_embed, testloader, trial_val, best_acc, train_steps):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    model.eval()
    wav_embed.eval()
    spk_embed.eval()
    
    # evaluate loss on dev set
    print('Validation Loss...')
    loss_val, acc_val = val_loss(model, wav_embed, spk_embed, testloader, train_steps)

    logging.info('Test the Data ---------, Loss={0:.3f}, Accuracy={1:.3f}'.format(loss_val, acc_val))

    if acc_val > best_acc:
        best_acc = acc_val

        torch.save({'wav_embed_state_dict':wav_embed.state_dict(),
                    'spk_embed_state_dict':spk_embed.state_dict()}, 
                   os.path.join(config.train_path, 'best_embedding.pt'))
        print('Best model saved!')
    
    model.to(device).train()
    wav_embed.to(device).train()
    spk_embed.to(device).train()

    return best_acc