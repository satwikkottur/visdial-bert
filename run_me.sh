# ROOT="data/deep_clevrdialog/"
# python preprocessing/pre_process_clevrdialog.py \
#     -visdial_train="${ROOT}deep_clevr_dialog_vd_train.json"\
#     -visdial_val="${ROOT}deep_clevr_dialog_vd_val.json"\
#     -visdial_test="${ROOT}deep_clevr_dialog_vd_test.json"

# python preprocessing/examine_data.py

# ROOT="data/clevrdialog/"
# python preprocessing/sample_clevrdialog.py \
#     --train_clevr_dialog_json="${ROOT}clevr_dialog_vd_train.json"

# python preprocessing/convert_clevrdialog_visdial.py \
#     --train_clevr_dialog_json="${ROOT}clevr_train_raw_70k.json" \
#     --val_clevr_dialog_json="${ROOT}clevr_val_raw_70k.json" \
#     --save_root="${ROOT}"

# ROOT="data/deep_clevrdialog/"
# python preprocessing/convert_deep_clevrdialog_visdial.py \
#     --train_clevr_dialog_json="${ROOT}deep_clevr_dialog_train.json" \
#     --val_clevr_dialog_json="${ROOT}deep_clevr_dialog_val.json" \
#     --test_clevr_dialog_json="${ROOT}deep_clevr_dialog_test.json" \
#     --save_root="${ROOT}"

# Training with images.
ROOT="data/clevrdialog/"
# # # CUDA_VISIBLE_DEVICES=0 python train.py \
# CHECKPOINT="checkpoints/22-Nov-20-17:54:26-Sun_601546/visdial_dialog_encoder_4900.ckpt"
# python train.py \
#     -batch_size 120 -batch_multiply 1 -lr 1e-4 -image_lr 2e-5 \
#     -mask_prob 0.1 -sequences_per_image 4 -n_gpus=8 \
#     -visdial_processed_train="${ROOT}clevr_dialog_vd_train_20k.json"\
#     -visdial_processed_val="${ROOT}clevr_dialog_vd_val.json"\
#     -visdial_processed_test="${ROOT}clevr_dialog_vd_test.json"\
#     -visdial_image_feats="${ROOT}clevr_butd/random/clevr_dialog_butd.lmdb" \
#     -img_loss_coeff=0.0
    # -img_loss_coeff=0.0 -start_path=$CHECKPOINT


# Training with images (Deep CLEVR-Dialog).
# ROOT="data/deep_clevrdialog/"
# # # python train.py \
# CUDA_VISIBLE_DEVICES=1 python train.py \
#     -batch_size 5 -batch_multiply 1 -lr 1e-4 -image_lr 2e-5 \
#     -mask_prob 0.1 -sequences_per_image 4 -n_gpus=2 \
#     -visdial_processed_train="${ROOT}deep_clevr_dialog_vd_test_light.json"\
#     -visdial_processed_val="${ROOT}deep_clevr_dialog_vd_val_light.json"\
#     -visdial_processed_test="${ROOT}deep_clevr_dialog_vd_test_light.json" \
#     -visdial_image_feats="${ROOT}clevr_butd/random/clevr_dialog_butd.lmdb" \
#     -img_loss_coeff=0.0 -max_seq_len=512 -deep_dialogs -visdial_tot_rounds=31


# Training without images.
# CUDA_VISIBLE_DEVICES=1 python train_language_only_baseline.py \
# python train_language_only_baseline.py \
#     -batch_size 120  -batch_multiply 1 -lr 1e-4 -image_lr 2e-5 \
#     -mask_prob 0.1 -sequences_per_image 4 -n_gpus=8 \
#     -visdial_processed_train="${ROOT}clevr_dialog_vd_train.json"\
#     -visdial_processed_val="${ROOT}clevr_dialog_vd_val.json"\
#     -visdial_processed_test="${ROOT}clevr_dialog_vd_test.json"\
#     -visdial_image_feats="${ROOT}clevr_butd/random/clevr_dialog_butd.lmdb"

    # -visdial_processed_train="${ROOT}clevr_dialog_vd_train.json"\
    # -visdial_processed_val="${ROOT}clevr_dialog_vd_val.json"\
    # -visdial_processed_test="${ROOT}clevr_dialog_vd_test.json"\

    # -visdial_processed_train="${ROOT}clevr_dialog_train_dev.json"\
    # -visdial_processed_val="${ROOT}clevr_dialog_val_dev.json"\
    # -visdial_processed_test="${ROOT}clevr_dialog_test_dev.json"\

    # -visdial_processed_train="${ROOT}clevr_dialog_train_light.json"\
    # -visdial_processed_val="${ROOT}clevr_dialog_val_light.json"\
    # -visdial_processed_test="${ROOT}clevr_dialog_test_light.json" \


# python train_visdial.py -batch_size 10  -batch_multiply 1 -lr 2e-5 -image_lr 2e-5 \
#     -mask_prob 0.1 -sequences_per_image 2

# Without images.
# CHECKPOINT="checkpoints/22-Nov-20-00:33:04-Sun_6775499/visdial_dialog_encoder_700.ckpt"
# # CHECKPOINT="checkpoints/21-Nov-20-16:56:49-Sat_6852119/visdial_dialog_encoder_350.ckpt"
CHECKPOINT="checkpoints/cd_q/22-Nov-20-02:55:26-Sun_6142272/visdial_dialog_encoder_92664.ckpt"
# CUDA_VISIBLE_DEVICES=0,1 python evaluate_language_only.py \
#     -n_gpus 2 -start_path $CHECKPOINT -save_name "history_dev_clevr_dialog"\
#     -visdial_processed_train="${ROOT}clevr_dialog_vd_train.json" \
#     -visdial_processed_val="${ROOT}clevr_dialog_vd_val.json" \
#     -visdial_processed_test="${ROOT}clevr_dialog_vd_test.json" \
#     -visdial_image_feats="${ROOT}clevr_butd/random/clevr_dialog_butd.lmdb" \
#     -batch_size=300

# Evaluate complete model.
# 8 gpu, cd_all
CHECKPOINT="checkpoints/cd_all/22-Nov-20-22:59:07-Sun_938646/visdial_dialog_encoder_11583.ckpt"
# 4 gpu, cd_q
CHECKPOINT="checkpoints/cd_q/22-Nov-20-02:55:26-Sun_6142272/visdial_dialog_encoder_92664.ckpt"

# 8 gpu, cd_all 
CHECKPOINT="checkpoints/cd_all/22-Nov-20-22:59:07-Sun_938646/visdial_dialog_encoder_57915.ckpt"
# 0.6820

# 8 gpu, cd_qi
CHECKPOINT="checkpoints/cd_qi/23-Nov-20-06:27:50-Mon_1019552/visdial_dialog_encoder_46332.ckpt"

# 8 gpu, dcd_all
CHECKPOINT="checkpoints/dcd_all/23-Nov-20-06:15:07-Mon_9166682/visdial_dialog_encoder_46332.ckpt"

# 
# checkpoints/dcd_all/23-Nov-20-06:15:07-Mon_9166682

# python evaluate_language_only.py \
# python evaluate.py \
#     -n_gpus 2 -start_path $CHECKPOINT -save_name "history_dev_clevr_dialog"\
#     -visdial_processed_train="${ROOT}clevr_dialog_vd_train.json" \
#     -visdial_processed_val="${ROOT}clevr_dialog_vd_val.json" \
#     -visdial_processed_test="${ROOT}clevr_dialog_vd_test.json" \
#     -visdial_image_feats="${ROOT}clevr_butd/random/clevr_dialog_butd.lmdb" \
#     -batch_size=100 -ignore_history
    # -batch_size=1000 -ignore_history
# 8 gpus, 500.

ROOT="data/deep_clevrdialog/"
python evaluate.py \
    -n_gpus 8 -start_path $CHECKPOINT -save_name "history_dev_clevr_dialog"\
    -visdial_processed_train="${ROOT}deep_clevr_dialog_vd_test.json"\
    -visdial_processed_val="${ROOT}deep_clevr_dialog_vd_val.json"\
    -visdial_processed_test="${ROOT}deep_clevr_dialog_vd_test.json" \
    -visdial_image_feats="${ROOT}clevr_butd/random/clevr_dialog_butd.lmdb" \
    -max_seq_len=512 -deep_dialogs -visdial_tot_rounds=31 -batch_size=500

# DATA_ROOT="data/clevrdialog/clevr_butd/"
# python preprocessing/convert_clevrdialog_features.py \
#     --train_clevr_features="${DATA_ROOT}train.hdf5" \
#     --train_clevr_map="${DATA_ROOT}train_ids_map.json" \
#     --val_clevr_features="${DATA_ROOT}val.hdf5" \
#     --val_clevr_map="${DATA_ROOT}val_ids_map.json" \
#     --save_root="${DATA_ROOT}/random/"
