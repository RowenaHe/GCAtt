# GZSL  ZSL

python train.py --dataset AWA2 --ga 0.5 --beta 1 --dis 0.3 --nSample 5000 --gpu 7 --S_dim 1024 --NS_dim 1024 --lr 0.0001 --classifier_lr 0.003 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 --matdataset --ae_drop 0.2 --gen_nepoch 220 --evl_start 40000 --evl_interval 400 --manualSeed 6152 --class_embedding CLIP_attribute_group_base_real_emb_100
python train.py --dataset AWA2 --ga 0.5 --beta 1 --dis 0.3 --nSample 5000 --gpu 7 --S_dim 1024 --NS_dim 1024 --lr 0.0001 --classifier_lr 0.003 --kl_warmup 0.01 --tc_warmup 0.001 --vae_dec_drop 0.5 --vae_enc_drop 0.4 --dis_step 2 --matdataset --ae_drop 0.2 --gen_nepoch 220 --evl_start 40000 --evl_interval 400 --manualSeed 6152 --zsl true --class_embedding CLIP_attribute_group_base_real_emb_100


python train.py --matdataset --dataset CUB --ga 5 --beta 0.003 --dis 0.3 --nSample 1000 --gpu 7 --S_dim 2048 --NS_dim 2048 --lr 0.0001 --classifier_lr 0.002 --gen_nepoch 600 --kl_warmup 0.001 --tc_warmup 0.0001 --weight_decay 1e-8 --vae_enc_drop 0.1 --vae_dec_drop 0.1 --dis_step 3 --ae_drop 0.0 --class_embedding CLIP_attribute_base_self_emb_322
python train.py --matdataset --dataset CUB --ga 5 --beta 0.003 --dis 0.3 --nSample 1200 --gpu 7 --S_dim 2048 --NS_dim 2048 --lr 0.0001 --classifier_lr 0.002 --gen_nepoch 600 --kl_warmup 0.001 --tc_warmup 0.0001 --weight_decay 1e-8 --vae_dec_drop 0.1 --dis_step 3 --ae_drop 0.0 --zsl true --class_embedding CLIP_attribute_base_self_emb_322

python train.py --matdataset --dataset FLO --ga 3 --beta 0.1 --dis 0.1 --nSample 1200 --gpu 7 --S_dim 2048 --NS_dim 2048 --lr 0.0001 --classifier_lr 0.003 --dis_step 3 --kl_warmup 0.001 --vae_dec_drop 0.4 --vae_enc_drop 0.4 --class_embedding CLIP_attribute_base_emb_331 --ae_drop 0.2
python train.py --matdataset --dataset FLO --ga 1 --beta 0.1 --dis 0.1 --nSample 1200 --gpu 7 --S_dim 2048 --NS_dim 2048 --lr 0.0001 --classifier_lr 0.003 --dis_step 3 --zsl true --kl_warmup 0.001 --vae_dec_drop 0.4 --vae_enc_drop 0.4 --class_embedding CLIP_attribute_base_emb_331 --ae_drop 0.4 

python train.py --matdataset --dataset SUN --ga 30 --beta 0.3 --dis 0.5 --nSample 400 --gpu 7 --S_dim 2048 --NS_dim 2048 --class_embedding CLIP_attribute_base_emb_339 --lr 0.0003 --kl_warmup 0.001 --tc_warmup 0.0003 --vae_dec_drop 0.2 --dis_step 3 --ae_drop 0.4
python train.py --matdataset --dataset SUN --ga 15 --beta 0.1 --dis 3.0 --nSample 400 --gpu 7 --S_dim 2048 --NS_dim 2048 --class_embedding CLIP_attribute_base_emb_339 --lr 0.0003 --zsl True --classifier_lr 0.005