# cls-lr *2, *5


# extra
python single_experiment_extra.py --dataset AWA2 --lr_gen_model 1.5e-4 --extra_class 0 --extra_num_per_class 100 --filter_lower_boundary 1 --generalized True --device 4 --class_embedding 'CLIP_attribute_group_base_real_emb_100'
python single_experiment_extra.py --dataset AWA2 --lr_gen_model 1.5e-4 --extra_class 0 --extra_num_per_class 100 --filter_lower_boundary 1 --generalized False --device 4 --class_embedding 'CLIP_attribute_group_base_real_emb_100'


python single_experiment_extra.py --dataset CUB --lr_gen_model 1.5e-4 --extra_class 0 --extra_num_per_class 100 --filter_lower_boundary 1 --generalized True  --device 4 --class_embedding 'CLIP_attribute_base_self_emb_322'
python single_experiment_extra.py --dataset CUB --lr_gen_model 1.5e-4 --extra_class 0 --extra_num_per_class 100 --filter_lower_boundary 1 --generalized False  --device 4 --class_embedding 'CLIP_attribute_base_self_emb_322'


python single_experiment_extra.py --dataset SUN --lr_gen_model 1.5e-4 --extra_class 0 --extra_num_per_class 20 --filter_lower_boundary 1 --generalized True  --device 4 --class_embedding 'CLIP_attribute_base_emb_339' --lr_cls 0.002
python single_experiment_extra.py --dataset SUN --lr_gen_model 1.5e-4 --extra_class 0 --extra_num_per_class 20 --filter_lower_boundary 1 --generalized False  --device 4 --class_embedding 'CLIP_attribute_base_emb_339' --lr_cls 0.002