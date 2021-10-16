import sys
import os
sys.path.append(os.getcwd())
import utils

from glob import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd 
from multiprocessing import Pool
from tqdm import tqdm
from time import time
import argparse

import preprocessing.silence_detector 

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-config_file", type=str, default=os.path.join(os.getcwd(), 'parameters.json'), help="config file")
args = parser.parse_args()

config = utils.Params(args.config_file)

def find_files(directory, pattern='**/*.wav'):
    print(os.path.join(directory, pattern))
    return glob(os.path.join(directory, pattern), recursive=True)

def VAD(audio):
    chunk_size = int(config.sr * 0.05) # 50ms
    index = 0
    sil_detector = preprocessing.silence_detector.SilenceDetector(9)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)

def read_audio(filename, sample_rate=config.sr):
    import soundfile as sf
#     audio, sr = sf.read(filename, samplerate=config.sr)
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
#     audio = librosa.util.normalize(audio)
    return audio

def features(libri, out_dir=config.train_o, name='0'):

    for i in tqdm(range(len(libri))):
        filename = libri[i:i+1]['filename'].values[0]        
        
        target_filename_partial = os.path.join(out_dir, filename.split("/")[-3] + '-' + filename.split("/")[-2] + '-' + filename.split("/")[-1].split('.')[0])  #clean
#         target_filename_partial = os.path.join(out_dir, filename.split("/")[-1].split(".")[0])  # librispeech clean
        try:
            raw_audio = read_audio(filename)
        except:
            print(filename, 'file error!')
            continue
        
        sample_num = int(config.seg_len * config.sr)
        
        if raw_audio.shape[0] < sample_num:
            print(raw_audio.shape[0])
            print('there is an error in file:',filename)
            continue
        else:
            target_filename = target_filename_partial + '.pkl'
            utils.array2pickle(raw_audio, target_filename)

def preprocess_and_save(wav_dir, out_dir):

    libri = data_catalog(wav_dir, pattern='**/*.m4a') 

    print("extract fbank from audio and save as pickle, using multiprocessing pool........ ")
    p = Pool(5)
    patch = int(len(libri)/5)
    for i in range(5):
        if i < 4:
            slibri = libri[i * patch: (i+1) * patch]
        else:
            slibri = libri[i * patch:]
        print("task %s slibri length: %d" %(i, len(slibri)))
        p.apply_async(features, args=(slibri, out_dir, i))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()

def data_catalog(dataset_dir, pattern='*.pkl'):
    libri = pd.DataFrame()
    libri['filename'] = find_files(dataset_dir, pattern=pattern)
    if pattern == '**/*.wav':
        libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-3])
        # libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-2]) #(test)
    else:
        libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[0])

    return libri

def test(out_dir=config.train_o):
    # libri = data_catalog()
    filename = "/storage/ge/voxceleb2/test/aac/id07426/GlWomRXtbt8/00042.m4a"   
    raw_audio = read_audio(filename)
    print(raw_audio.shape)
    exit()
    frame_num = int(config.seg_len * config.sr)
    hop_num = int(config.hop_size / 1000 * config.sr)
    num_segments = int(np.floor((raw_audio.shape[0] - frame_num * 2) / hop_num))
    print(num_segments)
    exit()
    sample_num = int(config.seg_len * config.sr)
    print(sample_num)
    num_segments = int(np.floor(raw_audio.shape[0] / sample_num))
    print(num_segments)
    
    target_filename_partial = os.path.join(out_dir, filename.split("/")[-2] + '-' + filename.split("/")[-1].split('.')[0])  #clean
    print(target_filename_partial)
    feature = raw_audio[:sample_num]
    utils.array2pickle(feature, target_filename_partial)
    exit()
    pieces = int(feature.shape[1]/config.seg_len)
    for i in range(pieces):
        temp_feature = feature[:, i*config.seg_len: (i+1)*config.seg_len]
        target_filename = target_filename_partial + '-' + str(i) +'.npy'
        print(target_filename)

if __name__ == '__main__':
    # test
#     test()
    preprocess_and_save(config.val_i, config.val_o)
    preprocess_and_save(config.train_i, config.train_o)
    
