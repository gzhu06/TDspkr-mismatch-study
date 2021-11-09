# Robustness-of-raw-speaker-embedding (Under Development)
Code base for "A study of the robustness of raw waveform based speaker embeddings under mismatched conditions" [https://arxiv.org/abs/2110.04265]
Pretrained models will be uploaded soon.

## Requirements
apex \
kaldi \
pytorch \
kaldi_python_io \
matplotlib \
tqdm

## Usage

### Step 0: data preparation

Training: VoxCeleb2 dev + 1M MUSAN aug data. \
In-domian evaluation: VoxCeleb 1 \
Out-of-domian evaluation: VOiCEs

### Step 1: feature extraction
In our experiment, we applied a filterbank of 30 to extract waveform embeddings.

```
bash run_features.sh
```

### Step 2: training
First edit ```run.sh``` file to change training configurations, then:
```
bash run.sh
```

### Step 3: evaluation on VoxCeleb test and VOiCEs 


```
bash run_plda.sh 0
```

## Results

(Update to the paper: we replace the original TDNN with [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) ([implementation](https://github.com/lawlict/ECAPA-TDNN). Notice that, we use log mel with 30 mel bins instead of 80 MFCCs in the original paper, which may cause this worse performance. Also, we did not apply AAM-softmax, adaptive score norm, data aug or cyclic lr with Adam. Results are based on cosine score, values in the table: EER (0.01 minDCF), interestingly, PLDA score is a little bit worse.) 

| Front-end  |VoxCeleb1-O  | VoxCeleb1-E  |VoxCeleb1-H | VOiCEs     |
|------------|-------------|--------------|------------|------------|
| log mel    | 1.91        |	1.95      |   3.31     |6.68 (0.469)|
| TDF-H-VD   | 1.6	(0.154)| 1.66 (0.173) | 2.86 (0.26)|7.95 (0.582)|

As can bee seen, although in-domain results are better, the mismatch still exists in out-of-domain data.

## prior works

[1] Zhu, G., Jiang, F., Duan, Z. (2021) Y-Vector: Multiscale Waveform Encoder for Speaker Embedding. Proc. Interspeech 2021, 96-100, doi: 10.21437/Interspeech.2021-1707 (Code:[Y-vector](https://github.com/gzhu06/Y-vector))