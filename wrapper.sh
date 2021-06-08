#!/bin/bash
# Be very wary of this explicit setting of CUDA_VISIBLE_DEVICES. Say you are
# running one task and asked for --gpus-per-node=1 then setting this variable will mean
# all your processes will want to run GPU 0 - disaster!! Setting this variable
# only makes sense in specific cases that I have described above where you are
# using --gpus-per-node=8 and I have spawned 8 tasks. So I need to divvy up the GPUs
# between the tasks. Think THRICE before you set this!!
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Debug output
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES

# lr = 2e-5

# Your CUDA enabled program here
ROOT="data/clevrdialog/"
# python train_language_only_baseline.py \
python train.py \
    -batch_size 120  -batch_multiply 1 -lr 1e-4 -image_lr 2e-5 \
    -mask_prob 0.1 -sequences_per_image 4 -n_gpus=8 \
    -visdial_processed_train="${ROOT}clevr_dialog_vd_train.json"\
    -visdial_processed_val="${ROOT}clevr_dialog_vd_val.json"\
    -visdial_processed_test="${ROOT}clevr_dialog_vd_test.json"\
    -visdial_image_feats="${ROOT}clevr_butd/random/clevr_dialog_butd.lmdb" \
    -img_loss_coeff=0.0 -save_path="checkpoints/cd_qi/" \
    -ignore_history
