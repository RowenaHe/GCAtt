#!/bin/bash
device="0"


echo run VGSE-SMO GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_group_base_real_emb_100  \
--nepoch 200 --classifier_lr 0.0002 --dataset AWA2  --batch_size 64 --gzsl --calibrated_stacking 0.95 --manualSeed 5214
echo run VGSE-SMO ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_group_base_real_emb_100 \
--nepoch 200 --classifier_lr 0.00001 --dataset AWA2  --batch_size 64 --manualSeed 5214

echo run VGSE_SMO on GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_base_self_emb_322 \
--nepoch 300 --classifier_lr 0.0001 --dataset CUB  --batch_size 64 --gzsl --calibrated_stacking 0.6 --manualSeed 5626
echo run VGSE_SMO on ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_base_self_emb_322 \
--nepoch 200 --classifier_lr 0.001 --dataset CUB  --batch_size 64 --manualSeed 5626


echo run VGSE-SMO GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_base_emb_331 \
--nepoch 300 --classifier_lr 0.002 --dataset FLO  --batch_size 64 --gzsl --calibrated_stacking 0.8 --manualSeed 5214
echo run VGSE-SMO ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_base_emb_331  \
--nepoch 300 --classifier_lr 0.0001 --dataset FLO  --batch_size 64 --manualSeed 5214


echo run VGSE_SMO on GZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_base_emb_339 \
--nepoch 200 --classifier_lr 0.0005 --dataset SUN  --batch_size 64  --gzsl --calibrated_stacking 0.4 --manualSeed 1143
echo run VGSE_SMO on ZSL
CUDA_VISIBLE_DEVICES=${device} python SJE.py --image_embedding res101 --class_embedding CLIP_attribute_base_emb_339 \
--nepoch 100 --classifier_lr 0.001 --dataset SUN  --batch_size 64 --manualSeed 1143
