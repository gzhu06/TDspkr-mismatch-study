import sys
import os, fnmatch, shutil
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib
import glob
import librosa
import kaldi_python_io

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from torch.multiprocessing import Pool, Process, set_start_method
torch.multiprocessing.set_start_method('spawn', force=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-config_file", type=str, default='', help="config file")
args = parser.parse_args()
config = utils.Params(args.config_file)

def load_model(model_file):
    # model definition
    wav_frontend = importlib.import_module('backbones.wavencoder.' + config.wavencoder)
    frame_backend = importlib.import_module('backbones.aggregator.' + config.backbone)
    wav_embedding = wav_frontend.architecture().cuda()
    spk_embedding = frame_backend.architecture().cuda()
    
    from apex import amp
    [wav_embedding, spk_embedding], _ = amp.initialize([wav_embedding, spk_embedding], 
                                                       [], opt_level="O1")

    wav_embedding = nn.DataParallel(wav_embedding)
    spk_embedding = nn.DataParallel(spk_embedding)
    checkpoint = torch.load(model_file, map_location='cpu')

    wav_embedding.load_state_dict(checkpoint['wav_embed_state_dict'])
    spk_embedding.load_state_dict(checkpoint['spk_embed_state_dict'])
    return wav_embedding, spk_embedding

def compute_embeddings(wav_model, spk_model, layer_names, device, batch_idx, files, vectorpath):
    
    layer_a, layer_b = layer_names
    
    wav_model.eval()
    spk_model.eval()
    outXvecArk_a = os.path.join(vectorpath, 'xvector_a.%d.ark'%(batch_idx))
    outXvecScp_a = os.path.join(vectorpath, 'xvector_a.%d.scp'%(batch_idx))
    
    outXvecArk_b = os.path.join(vectorpath, 'xvector_b.%d.ark'%(batch_idx))
    outXvecScp_b = os.path.join(vectorpath, 'xvector_b.%d.scp'%(batch_idx))
    
    activation = {}
    def get_activation(name):
        def hook(model, inputs, output):
            activation[name] = output.detach()
        return hook
    eval('spk_model.%s.register_forward_hook(get_activation(layer_a))' %layer_a)
    eval('spk_model.%s.register_forward_hook(get_activation(layer_b))' %layer_b)
    
    with torch.no_grad():
        with kaldi_python_io.ArchiveWriter(outXvecArk_a, outXvecScp_a, matrix=False) as writer_a:
            with kaldi_python_io.ArchiveWriter(outXvecArk_b, outXvecScp_b, matrix=False) as writer_b:
                for file in tqdm(files, desc='Computing embeddings'):
                    # prepare data
                    utt = utils.pickle2array(file)
                    utt = librosa.util.normalize(utt) # pay attention to this line
                    key = file.split('/')[-1]
                    utt = np.expand_dims(utt, axis=0)
                
                    # compute embeddings
                    utt_ipts = torch.from_numpy(utt).float().unsqueeze(0)
                    utt_embedding = spk_model(wav_model(utt_ipts.to(device)))
                    
                    writer_a.write(key, np.squeeze(activation[layer_a].cpu().numpy()))
                    writer_b.write(key, np.squeeze(activation[layer_b].cpu().numpy()))
                
if __name__ == '__main__':
    
    model_path = utils.get_last_checkpoint_if_any(config.train_path)
    print(model_path)
    division = config.eval_njob

    layerNames = ['module.tdnn_aggregator.linear', 'module.tdnn_aggregator.bn2']
    
    # prepare train dataset
    files = []
    for datafolder in config.train_source:
        files += glob.glob(datafolder + "/*.pkl")
    
    plda_path = os.path.join(config.train_path, 'plda')
    vector_path = os.path.join(plda_path, config.extract_dataset)
    if not os.path.exists(plda_path):
        os.mkdir(plda_path)
    if not os.path.exists(vector_path):
        os.mkdir(vector_path)
        
    # get train file embeddings
    if not os.path.exists(os.path.join(vector_path, 'xvector_a.scp')):

        patch = int(len(files)/division)
        
        # prepare device and load models
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_idx
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wav_model, spk_model = load_model(model_path)
        wav_model.to(device)
        spk_model.to(device)
        
        # prepare multi processing command
        L = []
        for i in range(division):
            if i < division-1:
                subfiles = files[i * patch: (i+1) * patch]
            else:
                subfiles = files[i * patch:]
            print("task %s sub filelist length: %d" %(i, len(subfiles)))
            L.append((wav_model, spk_model, layerNames, device, i, subfiles, vector_path))
        
        print('Extracting xvectors by distributing jobs to pool workers... ')
        pool2 = Pool(processes=division)
        pool2.starmap(compute_embeddings, L)
        pool2.terminate()
        
        print('Multithread job has been finished.')
        print('Writing xvectors to {}'.format(vector_path))
        os.system('cat %s/xvector_a.*.scp > %s/xvector_a.scp' %(vector_path, vector_path))
        os.system('cat %s/xvector_b.*.scp > %s/xvector_b.scp' %(vector_path, vector_path))
        
    if not os.path.exists(os.path.join(config.train_path, 'plda/utt2spk')):
        with open(os.path.join(config.train_path, 'plda/utt2spk'), 'w') as writer:
            for file in tqdm(files):
                utt_name = file.split('/')[-1]
                spk_id = utt_name.split('-')[0]
                writer.write(utt_name + ' ' + spk_id + '\n')
                
    if not os.path.exists(os.path.join(config.train_path, 'plda/spk2utt')):
        spkDict = {}
        for file in tqdm(files):
            utt_name = file.split('/')[-1]
            spk_id = utt_name.split('-')[0]
            
            if spk_id not in spkDict:
                spkDict[spk_id] = [utt_name]
            else:
                spkDict[spk_id].append(utt_name)
        
        with open(os.path.join(config.train_path, 'plda/spk2utt'), 'w') as writer:
            for spk in spkDict:
                writer.write(spk + " " + " ".join(spkDict[spk]) + '\n')
                    