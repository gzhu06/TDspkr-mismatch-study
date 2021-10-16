import sys
import os
sys.path.append(os.getcwd())
import utils

from glob import glob
import logging
import numpy as np
import random
from time import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from pytorch_model_summary import summary

import importlib

from evaluate.trials import trial_from_file
from VoxDataset import VoxDataset

import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

import train_utils

def print_model(train_path, s):
    with open(os.path.join(train_path, 'modelsummary.txt'),'a') as f:
        print(s, file=f)

def main():
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-config_file", type=str, default='', help="config file")
    args = parser.parse_args()
    config = utils.Params(args.config_file)
    
    print('assign GPU...')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_idx
    
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(np.random.randint(low=70000, high=80000))
    
    mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, config,))

def train_and_eval(rank, n_gpus, config):

    train_steps = 0
    epoch_point = 120000
    
    best_acc = 0

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)

    # generate trial list
    if rank == 0:
        print('initialized device...')
        print('generating trials...')
        trial_file = os.path.join(config.list, 'veri_test2.txt')
        trial_full = trial_from_file(trial_file, os.path.join(config.test_o, 'test'))
        random.shuffle(trial_full)
        trial_val = trial_full[:6400]
        print('loading data...')
    
    # spk dictionary
    files, spk_ids = [], []
    for datafolder in config.train_source:
        files_unit, ids_unit = train_utils.npy2file(datafolder)
        files += files_unit
        spk_ids += ids_unit

    # data generator
    labels_to_id = {}
    i = 0

    for label in np.unique(spk_ids):
        labels_to_id[label] = i
        i += 1

    no_of_speakers = len(np.unique(spk_ids))
    print(len(files), no_of_speakers)
    
    # distribute dataloader
    train_vox = VoxDataset(data_path=config.train_source, labels_to_id=labels_to_id, 
                           seg_len=config.seg_len)
    val_num = int(0.01*len(train_vox))
    torch.manual_seed(2021)
    train_vox, val_vox = torch.utils.data.random_split(train_vox, [len(train_vox)-val_num, val_num])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_vox,
                                                                    num_replicas=n_gpus,
                                                                    rank=rank,
                                                                    shuffle=True)
    
    train_dataloader = DataLoader(train_vox, batch_size=config.batch_size, 
                                  shuffle=False, num_workers=8, pin_memory=True, 
                                  drop_last=True, sampler=train_sampler)
    if rank == 0:
        val_dataloader = DataLoader(val_vox, batch_size=config.batch_size, shuffle=True, 
                                    pin_memory=True, num_workers=4, drop_last=True)
    
    # model definition
    wav_frontend = importlib.import_module('backbones.wavencoder.' + config.wavencoder)
    frame_backend = importlib.import_module('backbones.aggregator.' + config.backbone)
    loss_module = importlib.import_module('backbones.loss.' + config.loss_type)
    
    wav_embedding = wav_frontend.architecture().cuda(rank)
    spk_embedding = frame_backend.architecture().cuda(rank)
    spk_classifier = loss_module.loss(config.embed_dim, no_of_speakers).cuda(rank)
    
    if config.optim == 'SGD':
        if l2_reg:
            optimizer_g = optim.SGD([{'params': wav_embedding.parameters()},
                                     {'params': spk_embedding.parameters()}, 
                                     {'params': spk_classifier.parameters()}],
                                    lr=config.lr, momentum=0.9, nesterov=False)
            lr_ratio = 0.1
        else:
            optimizer_g = optim.SGD([{'params': wav_embedding.parameters()},
                                     {'params': spk_embedding.parameters(), 'weight_decay':5e-4}, 
                                     {'params': spk_classifier.parameters(), 'weight_decay':5e-4}],
                                    lr=config.lr, momentum=0.95, nesterov=False)
            lr_ratio = 0.1
        
    elif config.optim == 'ADAM':
        optimizer_g = optim.Adam([{'params': wav_embedding.parameters()}, 
                                  {'params': spk_embedding.parameters()}, 
                                  {'params': spk_classifier.parameters()}],
                                 lr=config.lr)
        lr_ratio = 0.6

    [wav_embedding, spk_embedding, spk_classifier], optimizer_g = amp.initialize([wav_embedding,
                                                                                  spk_embedding,
                                                                                  spk_classifier],
                                                                                 optimizer_g,
                                                                                 opt_level="O1")
    wav_embedding = DDP(wav_embedding)
    spk_embedding = DDP(spk_embedding)
    spk_classifier = DDP(spk_classifier)
    
    # training
    try:
        ckpt, epoch_str = train_utils.latest_checkpoint_path(config.train_path, "model_*_ckpt.pt")
        checkpoint = torch.load(ckpt, map_location='cpu')
        wav_embedding.load_state_dict(checkpoint['wav_embed_state_dict'])
        spk_embedding.load_state_dict(checkpoint['spk_embed_state_dict'])
        spk_classifier.load_state_dict(checkpoint['classifier_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
        
        iteration = int(epoch_point /(config.batch_size)) * (epoch_str) + 1
        epoch_str += 1
        
        lr_p = int(epoch_str / 120)
        optimizer_g.param_groups[0]['lr'] = config.lr * np.power(lr_ratio, lr_p)
        optimizer_g.param_groups[1]['lr'] = config.lr * np.power(lr_ratio, lr_p)
        optimizer_g.param_groups[2]['lr'] = config.lr * np.power(lr_ratio, lr_p)
    except:
        if rank == 0:
            print('no checkpoint found!')
            
        if rank == 0:
            sl = int(config.seg_len*config.sr)
            print_model(config.train_path,summary(wav_embedding.cuda(),
                                                  torch.zeros((1,1,sl)).cuda(), 
                                                  max_depth=None, show_input=False))
            print_model(config.train_path,summary(spk_embedding.cuda(),
                                                  torch.zeros((1, 512, 400)).cuda(),
                                                  max_depth=None, show_input=False))
            
        epoch_str = 0
        iteration = 0
        
    wav_embedding.train()
    spk_classifier.train()
    spk_embedding.train()
    
    torch.save({'wav_embed_state_dict':wav_embedding.state_dict()}, 
               '{0}/model_init'.format(config.train_path))
    
    for epoch in range(epoch_str, config.epoch):
        
        train_dataloader.sampler.set_epoch(epoch)
        
        print('Epoch:', epoch,'LR:', 
              optimizer_g.param_groups[0]['lr'], 
              optimizer_g.param_groups[1]['lr'], 
              optimizer_g.param_groups[2]['lr'])
        
        for i_batch, sample_batched in enumerate(train_dataloader):
            
            orig_time = time()
            
            # one batch of training data
            data, target = sample_batched['audio'], sample_batched['identity']
            data, target = data.cuda(rank, non_blocking=True), target.cuda(rank, non_blocking=True)
            
            # gradient accumulates
            optimizer_g.zero_grad()
            
            utt_embeddings = spk_embedding(wav_embedding(data))
            loss, softmax_output = spk_classifier(utt_embeddings, target)
            loss = loss.mean()
            kld_loss = wav_embedding.module.tdfbanks.complex_conv.kld()
            loss += kld_loss * 0.02
            
            # back propagation
            with amp.scale_loss(loss, optimizer_g) as scaled_loss:
                scaled_loss.backward()
                
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_g), 2.0)
            optimizer_g.step()
            
            if rank==0:
                train_acc = train_utils.accuracy(target, softmax_output)
            
                mesg = "Time:{0:.2f}, Epoch:{1}, Iter:{2}, Loss:{3:.3f}, kld Loss:{4:.3f}, Accuracy:{5:.3f}, LR:{6:.3f}, {7:.3f}".format(time()-orig_time, epoch, iteration, loss.item(), kld_loss.item(), train_acc, optimizer_g.param_groups[0]['lr'], optimizer_g.param_groups[1]['lr'])
                print(mesg)
            
                with open(os.path.join(config.train_path, 'train_loss.txt'), "a") as f:
                    f.write("{0},{1},{2},{3}\n".format(iteration, loss.item(), 
                                                       kld_loss.item(), train_acc))
            
            iteration += 1
        
            if (i_batch+1)*config.batch_size*n_gpus >= epoch_point:
                break
        
        if rank==0:
            utils.create_dir_and_delete_content(config.train_path)
            torch.save({'wav_embed_state_dict':wav_embedding.state_dict(),
                        'spk_embed_state_dict':spk_embedding.state_dict(), 
                        'classifier_state_dict':spk_classifier.state_dict(),
                        'optimizer_state_dict':optimizer_g.state_dict()}, 
                       '{0}/model_{1}_ckpt.pt'.format(config.train_path, epoch))
            best_acc = train_utils.val_step(spk_classifier, wav_embedding, 
                                            spk_embedding, val_dataloader, 
                                            trial_val, best_acc, iteration)
        
        
        if epoch in [128, 192]:
            optimizer_g.param_groups[0]['lr'] *= lr_ratio
            optimizer_g.param_groups[1]['lr'] *= lr_ratio
            optimizer_g.param_groups[2]['lr'] *= lr_ratio
    
if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format=' | %(message)s')
    main()