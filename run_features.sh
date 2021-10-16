# stage 0.1: voices /storage/ge/voices/Speaker_Recognition/sid_eval /storageNVME/ge/voices/eval
# stage 0.2: sitw /storage/ge/SITW/eval/audio /storageNVME/ge/SITW/eval
# stage 0.3: vox2noisy /storage/ge/SITW/eval/audio /storageNVME/ge/SITW/eval
stage=0.3
dataset_path=/storage/ge/voxceleb2_aug
feature_path=/storageNVME/ge/voxceleb2noisy100k/train
python main.py -stage $stage -train_path ./experiments/voices_extract -val_i $dataset_path -val_o $feature_path -seg_len 1.0