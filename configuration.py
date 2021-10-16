import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
def get_config():
    config, unparsed = parser.parse_known_args()
    return config, parser

# Feature extraction procedure
rawdatapath = '/storage/ge/voxceleb2'
data_path = '/storageNVME/ge/voxceleb2'
data_arg = parser.add_argument_group('Data')
data_arg.add_argument("-train_i", type=str, default=os.path.join(rawdatapath, 'dev', 'aac'), help="input folder path with raw files")
data_arg.add_argument("-train_o", type=str, default=os.path.join(data_path, 'train'), help="output folder with data")
data_arg.add_argument("-val_i", type=str, default=os.path.join(rawdatapath, 'test', 'aac'), help="input folder path with raw files")
data_arg.add_argument("-val_o", type=str, default=os.path.join(data_path, 'val'), help="output folder with data")
data_arg.add_argument("-sr", type=int, default=16000, help="sampling rate")
data_arg.add_argument("-seg_len", type=float, default=3.9, help="duration of segments(s)")
data_arg.add_argument("-eer_length", type=float, default=3.9, help="duration of eer computing segments(s)")
# data_arg.add_argument("-filter_size", type=int, default=2, help="duration of segments(ms)")

# Training Parameters
train_arg = parser.add_argument_group('Training')
train_path = '/home/ge/SV-wav-pytorch/experiments/dummy/'
train_arg.add_argument("-train_source", type=str, action='append', help="data source for training")
train_arg.add_argument("-wavencoder", type=str, default='cnn', help="wavform encoder")
train_arg.add_argument("-backbone", type=str, default='rawcnn', help="architecture for training")
train_arg.add_argument("-train_path", type=str, default=train_path, help="train path")
train_arg.add_argument('-epoch', type=int, default=30, help="number of epoch for training")
train_arg.add_argument('-batch_size', type=int, default=128, help="batch size")
train_arg.add_argument('-embed_dim', type=int, default=256, help="embedding vector dimension")
train_arg.add_argument('-loss_type', type=str, default='softmaxloss', help="loss type for training")
train_arg.add_argument('-final_layer', type=str, default='None', help="loss type for training")
train_arg.add_argument('-optim', type=str, default='SGD', help="optimizer")
train_arg.add_argument('-lr', type=float, default=0.01, help="learning rate")
train_arg.add_argument('-gpu_idx', type=str, default='0', help="gpu index")

# Evaluation Parameters
eva_arg = parser.add_argument_group('Evaluation')
testpath = '/storage/ge/voxceleb/test'
# testpath = '/data/ge/voxceleb/test'
eva_arg.add_argument("-enroll", type=str, default=os.path.join(testpath, 'enroll'), help="enroll folder path with raw files")
eva_arg.add_argument("-layer", type=str, default='fc2', help="layer name")
eva_arg.add_argument("-eval", type=str, default=os.path.join(testpath, 'eval'), help="eval folder path with raw files")
eva_arg.add_argument("-extract_dataset", type=str, default='vox2', help="dataset name for vector extraction")
eva_arg.add_argument("-eval_njob", type=int, default=3, help="number of eval multipreocessing jobs")
eva_arg.add_argument("-list", type=str, default=os.path.join(testpath, 'wav'), help="eval list, it should also contain eval files")
eva_arg.add_argument("-test_i", type=str, default=os.path.join(testpath, 'test'), help="eval folder path with pickle files")
eva_arg.add_argument("-test_o", type=str, default='/storageNVME/ge/voxceleb', help="eval folder path with pickle files")
config = get_config()
