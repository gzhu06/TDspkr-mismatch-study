. ./cmd.sh
. ./path.sh

exp_folder=./experiments/tdfbank-ana-yvector-vd-ecapa
architecture=tdfbank-ana-vd-yvector

gpu=3
multi_proc=1
stage=$1
voices_trials=./trials/voices

if [ $stage -le 0 ]; then

    python main.py -stage 4 -train_path $exp_folder -gpu_idx $gpu -wavencoder $architecture -backbone ECAPA-TDNN -batch_size 64 -lr 0.005 -loss_type AMsoftmax -embed_dim 512 -epoch 360 -seg_len 3.9 -optim SGD -eer_length 3.9 -train_source /storageNVME/ge/voices/eval -eval_njob $multi_proc -extract_dataset voices
    
fi

if [ $stage -le 1 ]; then
    for embd_layer in b; do
      $train_cmd $exp_folder/log/voices_cosine.log \
        cat $voices_trials \| awk '{print $1" "$2}' \| \
        ivector-compute-dot-products - \
        "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/voices/xvector_${embd_layer}.scp ark:- | ivector-normalize-length ark:- ark:- |" "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/voices/xvector_${embd_layer}.scp ark:- | ivector-normalize-length ark:- ark:- |" $exp_folder/log/cosine_voices || exit 1;

      eer=`compute-eer <($KALDI_ROOT/egs/voxceleb/pytorch/local/prepare_for_eer.py $voices_trials $exp_folder/log/cosine_voices) 2> /dev/null`
      mindcf1=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.01 $exp_folder/log/cosine_voices $voices_trials 2> /dev/null`
      mindcf2=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.001 $exp_folder/log/cosine_voices $voices_trials 2> /dev/null`
      echo "layer: ${embd_layer}"
      echo "cosine EER: $eer%"
      echo "minDCF(p-target=0.01): $mindcf1"
      echo "minDCF(p-target=0.001): $mindcf2"
    done
fi

if [ $stage -le 2 ]; then
    for embd_layer in a; do
      $train_cmd $exp_folder/log/voices_ldaplda.log \
        ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 $exp_folder/plda/ldaplda_${embd_layer} - |" \
        "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/voices/xvector_${embd_layer}.scp ark:- | transform-vec $exp_folder/plda/transform_${embd_layer}.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $exp_folder/plda/mean_${embd_layer}.vec scp:$exp_folder/plda/voices/xvector_${embd_layer}.scp ark:- | transform-vec $exp_folder/plda/transform_${embd_layer}.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$voices_trials' | cut -d\  --fields=1,2 |" $exp_folder/log/ldaplda_voices

      eer=`compute-eer <($KALDI_ROOT/egs/voxceleb/pytorch/local/prepare_for_eer.py $voices_trials $exp_folder/log/ldaplda_voices) 2> /dev/null`
      mindcf1=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.01 $exp_folder/log/ldaplda_voices $voices_trials 2> /dev/null`
      mindcf2=`$KALDI_ROOT/egs/voxceleb/pytorch/sid/compute_min_dcf.py --p-target 0.001 $exp_folder/log/ldaplda_voices $voices_trials 2> /dev/null`
      echo "layer: ${embd_layer}"
      echo "lda-plda EER: $eer%"
      echo "minDCF(p-target=0.01): $mindcf1"
      echo "minDCF(p-target=0.001): $mindcf2"
    done
fi