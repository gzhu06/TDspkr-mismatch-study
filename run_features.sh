stage=0
dataset_path=/storage/ge/voices/Speaker_Recognition
feature_path=/storageNVME/ge/voxceleb2noisy100k/train_demo
python main.py -stage $stage -train_path ./experiments/voices_extract -data_i $dataset_path -data_o $feature_path -seg_len 3.9