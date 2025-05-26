
### execute this function to train and test the vae-model
#break
from vaemodel_extra import Model
import numpy as np
import pickle
import torch
import os
import argparse
print('CUDA:',torch.cuda.is_available())
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset')
parser.add_argument('--class_embedding')
parser.add_argument('--num_shots',type=int,default=0)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--generalized', type = str2bool)
parser.add_argument('--extra_class',type=int,default=0)
parser.add_argument('--extra_num_per_class',type=int,default=0)
parser.add_argument('--filter_lower_boundary',type=int,default=0)
parser.add_argument('--lr_gen_model',type=float,default=0.00015)
parser.add_argument('--lr_cls',type=float,default=0.001)
parser.add_argument('--dataroot', default='/cver/qzhe/GCAtt-opensource/data', help='path to dataset')


# mode-aug classification 也需要 sample-level 数据的读取
# mode-level, sample-level , mode-/cls- rectified sample-level都在这里
parser.add_argument('--sample_level', action='store_true', default=False, help='load sample-level data?') 

# 关于 sample-level, mode-level, cls-/mode- rectified sample-level 训练数据的超参数
parser.add_argument('--sample_level_train', action='store_true', default=False, help='use sample level semantics in ZSL model training?')

parser.add_argument('--semantics_for_classifier', action='store_true', default=False, help='use semantics in addition to vision for classification?') 
parser.add_argument('--cls_sample_att1', action='store_true', default=False, help='default: only seen -> GZSL| True: seen+unseen -> GZSL+TZSL')
#parser.add_argument('--mode_level_lab', default='mode_level_CLIP_att_label', help='name of the utilized mode level labels (key name in the .hdf5 file)')
args = parser.parse_args()
# 读取的 sample-level  semantics 一定和 class-level semantics 对应，命名也一致
args.sample_level_emb = 'sample_level_'+args.dataset+'_'+args.class_embedding



########################################
# the basic hyperparameters
########################################
hyperparameters = {
    'num_shots': 0,
    'device': 'cuda:{}'.format(args.device),
    'model_specifics': {'cross_reconstruction': True,
                       'name': 'CADA',
                       'distance': 'wasserstein',
                       'warmup': {'beta': {'factor': 0.25,
                                           'end_epoch': 93,
                                           'start_epoch': 0},
                                  'cross_reconstruction': {'factor': 2.37,
                                                           'end_epoch': 75,
                                                           'start_epoch': 21},
                                  'distance': {'factor': 8.13,
                                               'end_epoch': 22,
                                               'start_epoch': 6}}},

    'lr_gen_model': args.lr_gen_model,
    'generalized': True,
    'batch_size': 50,
    'xyu_samples_per_class': {'SUN': (200, 0, 400, 0),
                              'APY': (200, 0, 400, 0),
                              'CUB': (200, 0, 400, 0),
                              'AWA2': (200, 0, 400, 0),
                              'FLO': (200, 0, 400, 0),
                              'AWA1': (200, 0, 400, 0)},
    'epochs': 100,
    'loss': 'l1',
    'auxiliary_data_source': 'att',
    'lr_cls': args.lr_cls,
    'dataset': 'CUB',
    'hidden_size_rule': {'resnet_features': (1560, 1660),
                        'att': (1450, 665),
                        'sentences': (1450, 665) },
    'latent_size': 64
}

# The training epochs for the final classifier, for early stopping,
# as determined on the validation split

# 这个调参很烦人啊，大概率是像 f-CLSWGAN 差不多，基本取到了最高点。
# 我干脆把它替换掉吧
cls_train_steps = [
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 21},
      {'dataset': 'SUN',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 30},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 33},
      {'dataset': 'AWA1', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 25},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': True, 'cls_train_steps': 23},
      {'dataset': 'CUB',  'num_shots': 0, 'generalized': False, 'cls_train_steps': 22},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': True, 'cls_train_steps': 29},
      {'dataset': 'AWA2', 'num_shots': 0, 'generalized': False, 'cls_train_steps': 39},
      ]

##################################
# change some hyperparameters here
##################################
hyperparameters['dataset'] = args.dataset
hyperparameters['class_embedding'] = args.class_embedding
hyperparameters['num_shots']= args.num_shots
hyperparameters['generalized']= args.generalized
hyperparameters['extra_class'] = args.extra_class
hyperparameters['extra_num_per_class'] = args.extra_num_per_class
hyperparameters['filter_lower_boundary'] = args.filter_lower_boundary

hyperparameters['cls_train_steps'] = [x['cls_train_steps']  for x in cls_train_steps
                                        if all([hyperparameters['dataset']==x['dataset'],
                                        hyperparameters['num_shots']==x['num_shots'],
                                        hyperparameters['generalized']==x['generalized'] ])][0]

print('***')
print(hyperparameters['device'])
print(hyperparameters['cls_train_steps'] )
if hyperparameters['generalized']:
    hyperparameters['samples_per_class'] = {'CUB': (200, 0, 400, 0), 'SUN': (200, 0, 400, 0),
                            'APY': (200, 0,  400, 0), 'AWA1': (200, 0, 400, 0),
                            'AWA2': (200, 0, 400, 0), 'FLO': (200, 0, 400, 0)}
else:
    hyperparameters['samples_per_class'] = {'CUB': (0, 0, 200, 0), 'SUN': (0, 0, 200, 0),
                                                'APY': (0, 0, 200, 0), 'AWA1': (0, 0, 200, 0),
                                                'AWA2': (0, 0, 200, 0), 'FLO': (0, 0, 200, 0)}
#print('HA?')
#print('Model construction.')
model = Model(hyperparameters, args)
#print('model to device')
model.to(hyperparameters['device'])

print('Initiating model training.')
losses = model.train_vae(args)

# 关注下面 model.train_classifier()
# u,s,h = model.train_classifier()