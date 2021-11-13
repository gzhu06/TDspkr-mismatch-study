# Robustness-of-raw-speaker-embedding (Under Development)
Code base for "A study of the robustness of raw waveform based speaker embeddings under mismatched conditions" [https://arxiv.org/abs/2110.04265]
Pretrained models will be uploaded soon.

## Introduction

Waveform based models are notorious for channel mismatch in speaker recognition due to densely sampled waveform. It's overwhelming for the neural networks to learn meaningful representations given guidence from scratch. 
{:refdef: style="text-align: center;"}
![mismatch](/doc/filter_scale.pdf){:width="500"}
\
Typical waveform based models:

Common backbones (due to page limits, we didn't have space for this image):
{:refdef: style="text-align: center;"}
![commonbackbone](/doc/dsblock){:width="500"}
\
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

### Step 4: Adaptive score normalization

Please see ```snorm.ipynb```.

## Results

 Update to the paper: we replace the original TDNN with [ECAPA-TDNN (denoted as ECAPA)](https://arxiv.org/abs/2005.07143) ([implementation](https://github.com/lawlict/ECAPA-TDNN), here we also follow A.2 model.) Here, we use log mel with 30 mel bins instead of 80 MFCCs in the original paper, which may cause this worse result. Also, we did not apply adaptive score norm (will be updated later), various data aug or cyclic lr with Adam. Results are based on cosine score, values in the table: EER (0.01 minDCF), interestingly, PLDA score is a little bit worse. Other details: we use 512 as embedding dimension, and for AAM-softmax, we use a=30,m=0.3 temporarily.

| Frontend   |Backbone   |Softmax|Vox1-O    | Vox1-E   |Vox1-H    | VOiCEs   |
|------------|-----------|-------|----------|----------|----------|----------|
| log mel*   | Vanilla   |AM     |2.26/0.256|2.37/0.279|4.14/0.408|6.79/0.553|
| log mel    |ECAPA-C512 |AM     |1.91      |1.95	   |3.31 	  |6.68/0.469|
| log mel    |ECAPA-C1024|AM     |1.38/0.146|1.51/0.152|2.7/0.254 |6.60/0.427|
| log mel    |ECAPA-C512 |AAM    |1.74/0.194|1.82/0.194|3.07/0.292|6.39/0.452|
| log mel    |ECAPA-C1024|AAM    |1.44/0.137|1.57/0.169|2.81/0.269|6.65/0.457|
| TDF-H-VD*  | Vanilla   |AM     |1.99/0.266|2.26/0.253|3.93/0.385|7.40/0.633|
| TDF-H-VD   |ECAPA-C512 |AM     |1.6 /0.154|1.66/0.173|2.86 /0.26|7.95/0.582|

(*: Results shown in paper.) \
As can bee seen, although in-domain results are better, the mismatch still exists in out-of-domain data.

## Prior works

[1] Zhu, G., Jiang, F., Duan, Z. (2021) Y-Vector: Multiscale Waveform Encoder for Speaker Embedding. Proc. Interspeech 2021, 96-100, doi: 10.21437/Interspeech.2021-1707 (Code:[Y-vector](https://github.com/gzhu06/Y-vector))