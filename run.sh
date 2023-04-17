#!/usr/bin/env bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2,7,0 python -m torch.distributed.launch --master_port 12138  --nproc_per_node=3 tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor MODEL.PRETRAIN_ON False MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 \
SOLVER.MAX_ITER 1200001 SOLVER.VAL_PERIOD 1 SOLVER.CHECKPOINT_PERIOD 4000 GLOVE_DIR /data/hetao/ne/neural-motifs/data \
MODEL.PRETRAINED_DETECTOR_CKPT  /data/hetao/datasets/vg/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR /data/hetao/checkpoints/motif_close4
