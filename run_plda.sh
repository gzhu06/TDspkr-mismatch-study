. ./cmd.sh
. ./path.sh

exp_folder=./experiments/
architecture=

gpu=3
multi_proc=1
stage=$1
voxceleb1_trials=./trials/vox1o

if [ $stage -le 0 ]; then
    
    python main.py -stage 4 -train_path $exp_folder -gpu_idx $gpu -wavencoder $architecture -backbone TDNN -seg_len 3.9 -optim SGD -eer_length 3.9 -train_source /storageNVME/ge/voxceleb2/train -train_source /storageNVME/ge/voxceleb2noisy100k/train -eval_njob $multi_proc -extract_dataset vox2
    
    python main.py -stage 4 -train_path $exp_folder -gpu_idx $gpu -wavencoder $architecture -backbone TDNN -seg_len 3.9 -optim SGD -eer_length 3.9 -train_source /storageNVME/ge/voxceleb/train -train_source /storageNVME/ge/voxceleb/test -train_source /storageNVME/ge/voxceleb/val -eval_njob $multi_proc -extract_dataset vox1
    
fi

if [ $stage -le 1 ]; then
    for embd_layer in a b; do
        # Compute the mean vector for centering the evaluation xvectors.
        $train_cmd $exp_folder/log/compute_mean.log \
          ivector-mean scp:$exp_folder/plda/vox2/xvector_${embd_layer}.scp \
          $exp_folder/plda/mean_${embd_layer}.vec

        # This script uses LDA to decrease the dimensionality prior to PLDA.
        lda_dim=200
        $train_cmd $exp_folder/log/lda.log \
          ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
          "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/vox2/xvector_${embd_layer}.scp ark:- |" \
          ark:$exp_folder/plda/utt2spk $exp_folder/plda/transform_${embd_layer}.mat
    done
fi

if [ $stage -le 2 ]; then
    for embd_layer in a; do
        # Train the PLDA model.
        $train_cmd $exp_folder/log/plda.log \
          ivector-compute-plda ark:$exp_folder/plda/spk2utt \
          "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/vox2/xvector_${embd_layer}.scp ark:- | ivector-normalize-length ark:-  ark:- |" $exp_folder/plda/plda_${embd_layer}

        # Train the LDA-PLDA model.
        $train_cmd $exp_folder/log/ldaplda.log \
          ivector-compute-plda ark:$exp_folder/plda/spk2utt \
          "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/vox2/xvector_${embd_layer}.scp ark:- | transform-vec $exp_folder/plda/transform_${embd_layer}.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" $exp_folder/plda/ldaplda_${embd_layer}
    done
fi

if [ $stage -le 3 ]; then
    for embd_layer in b; do
      $train_cmd $exp_folder/log/voxceleb1_cosine.log \
        cat $voxceleb1_trials \| awk '{print $1" "$2}' \| \
        ivector-compute-dot-products - \
        "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/vox1/xvector_${embd_layer}.scp ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/vox1/xvector_${embd_layer}.scp ark:- | ivector-normalize-length ark:- ark:- |" \
        $exp_folder/log/cosine_voxceleb1 || exit 1;

      eer=`compute-eer <($KALDI_ROOT/egs/voxceleb/pytorch/local/prepare_for_eer.py $voxceleb1_trials $exp_folder/log/cosine_voxceleb1) 2> /dev/null`
      mindcf1=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.01 $exp_folder/log/cosine_voxceleb1 $voxceleb1_trials 2> /dev/null`
      mindcf2=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.001 $exp_folder/log/cosine_voxceleb1 $voxceleb1_trials 2> /dev/null`
      echo "layer: ${embd_layer}"
      echo "cosine EER: $eer%"
      echo "minDCF(p-target=0.01): $mindcf1"
      echo "minDCF(p-target=0.001): $mindcf2"
    done
fi

if [ $stage -le 4 ]; then
    for embd_layer in a; do
      $train_cmd $exp_folder/log/voxceleb1_ldaplda.log \
        ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 $exp_folder/plda/ldaplda_${embd_layer} - |" \
        "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/vox1/xvector_${embd_layer}.scp ark:- | transform-vec $exp_folder/plda/transform_${embd_layer}.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/vox1/xvector_${embd_layer}.scp ark:- | transform-vec $exp_folder/plda/transform_${embd_layer}.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $exp_folder/log/ldaplda_voxceleb1

      eer=`compute-eer <($KALDI_ROOT/egs/voxceleb/pytorch/local/prepare_for_eer.py $voxceleb1_trials $exp_folder/log/ldaplda_voxceleb1) 2> /dev/null`
      mindcf1=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.01 $exp_folder/log/ldaplda_voxceleb1 $voxceleb1_trials 2> /dev/null`
      mindcf2=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.001 $exp_folder/log/ldaplda_voxceleb1 $voxceleb1_trials 2> /dev/null`
      echo "layer: ${embd_layer}"
      echo "lda-plda EER: $eer%"
      echo "minDCF(p-target=0.01): $mindcf1"
      echo "minDCF(p-target=0.001): $mindcf2"
    done
fi