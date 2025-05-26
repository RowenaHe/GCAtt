import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy
import random

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

def choose_extra_samples_sample_level(feature, label, sample_attribute, num_use_class, num_each_class, filter_lower_boundary, mode_label=None):
    available_label = torch.from_numpy(label).unique()
    assert num_use_class<len(available_label), "not enough extra classes available"
    
    # 由于抽取 extra feature 及对应 label 是一个类一个类进行的，它们是一块儿一块儿组成的
    # 两个相邻 label 若不一样，则表明前一个类已结束
    # 这里将 label 作为词典 key 值，value 保存其 label-idx
    _dict = {}
    start = 0
    for idx, l in enumerate(label):
        # 通过与后一个标签比较方式划分数据，最后一个需要特殊处理
        if idx < len(label)-1:
            if l != label[idx+1]:
                end = idx+1
                _dict[l] = range(start, end)
                start=end
        # 最后一个
        else:
            _dict[l] = range(start,len(label))
    print('全部可用 extra data 类别数量为：',len(_dict.keys()))

    # 引入 filter-lower-bound 进行筛选
    _new_dict = copy.deepcopy(_dict)
    for _k in _dict.keys():
        if len(_dict[_k])<=filter_lower_boundary:
            del _new_dict[_k]
    _dict = _new_dict
    print('过滤后 extra data 类别数量为：{}. （数量下限为{}）'.format(len(_dict.keys()),filter_lower_boundary))
    
    assert num_use_class<len(_dict.keys()), "not enough extra classes available"

    # 从 满足条件的全部类别中 随机选择 若干类
    if num_use_class >=0:
        use_class = random.sample(list(_dict.keys()), num_use_class)
        print('CLASS chosen:', use_class)
    else:
        use_class = list(_dict.keys())
        print('使用全部满足 filtering 条件的类别')

    # 猜测下面这个 if-else 没有意义，通过前面 filter-lower-bound 筛选之后的 _dict 里可选的类别一定都满足
    use_idx = []
    for _class in use_class:
        #print('USED CLASS:', _class.item())
        if len(_dict[_class.item()])>num_each_class:
            #print('Enough NUM:',num_each_class)
            use_idx.extend(random.sample(_dict[_class.item()],num_each_class))
        else:
            #print('NOT Enough NUM:', len(_dict[_class.item()]))
            use_idx.extend(_dict[_class.item()])
    if mode_label is not None:
        return feature[use_idx], label[use_idx], sample_attribute[use_idx], mode_label[use_idx]
    else:
        return feature[use_idx], label[use_idx], sample_attribute[use_idx]
    
def choose_extra_samples(feature, label, num_use_class, num_each_class, filter_lower_boundary):
    available_label = torch.from_numpy(label).unique()
    assert num_use_class<len(available_label), "not enough extra classes available"
    _dict = {}
    start = 0
    for idx, l in enumerate(label):
        # 通过与后一个标签比较方式划分数据，最后一个需要特殊处理
        if idx < len(label)-1:
            if l != label[idx+1]:
                end = idx+1
                _dict[l] = range(start, end)
                start=end
        # 最后一个
        else:
            _dict[l] = range(start,len(label))
    print('全部可用 extra data 类别数量为：',len(_dict.keys()))

    # 引入 filter-lower-bound 进行筛选
    _new_dict = copy.deepcopy(_dict)
    for _k in _dict.keys():
        if len(_dict[_k])<=filter_lower_boundary:
            del _new_dict[_k]
    _dict = _new_dict
    print('过滤后 extra data 类别数量为：{}. （数量下限为{}）'.format(len(_dict.keys()),filter_lower_boundary))
    
    assert num_use_class<len(_dict.keys()), "not enough extra classes available"

    # 从 满足条件的全部类别中 随机选择 若干类
    if num_use_class >=0:
        use_class = random.sample(list(_dict.keys()), num_use_class)
        print('CLASS chosen:', use_class)
    else:
        use_class = list(_dict.keys())
        print('使用全部满足 filtering 条件的类别')

    # 猜测下面这个 if-else 没有意义，通过前面 filter-lower-bound 筛选之后的 _dict 里可选的类别一定都满足
    use_idx = []
    for _class in use_class:
        #print('USED CLASS:', _class.item())
        if len(_dict[_class.item()])>num_each_class:
            #print('Enough NUM:',num_each_class)
            use_idx.extend(random.sample(_dict[_class.item()],num_each_class))
        else:
            #print('NOT Enough NUM:', len(_dict[_class.item()]))
            use_idx.extend(_dict[_class.item()])
    return feature[use_idx], label[use_idx]

class DATA_LOADER(object):
    def __init__(self, opt, dataset, aux_datasource, extra_class, extra_num_per_class, filter_lower_boundary, device='cuda', class_embedding='att'):

        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder

        #print('Project Directory:')
        #print(project_directory)
        #data_path = '/cver/qzhe/PyCharmOnSSH/WNids/data'
        data_path = opt.dataroot
        #print('Data Path')
        print(data_path)
        sys.path.append(data_path)
        self.opt = opt
        self.data_path = data_path
        self.class_embedding = class_embedding
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource
        
        self.extra_class = extra_class
        self.extra_num_per_class = extra_num_per_class
        self.filter_lower_boundary = filter_lower_boundary

        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

#         if self.dataset == 'CUB':
#             self.datadir = self.data_path + '/CUB/'
#         elif self.dataset == 'SUN':
#             self.datadir = self.data_path + '/SUN/'
#         elif self.dataset == 'AWA1':
#             self.datadir = self.data_path + '/AWA1/'
#         elif self.dataset == 'AWA2':
#             self.datadir = self.data_path + '/AWA2/'
        self.read_h5dataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        print('===================loda finished')
    '''
    def next_batch(self, batch_size):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        idx = torch.randperm(self.ntrain_extra)[0:batch_size]
        batch_feature = self.train_feature_extra[idx]
        batch_label =  self.train_label_extra[idx]
        if self.opt.sample_level_train or self.opt.mode_level_train:
            batch_att = self.train_sample_attribute_extra[idx]
        else:
            batch_att = self.aux_data_extra[batch_label]
        return batch_label, [ batch_feature, batch_att]'''

    # QZHe version
        # QZHe version
    def read_h5dataset(self, opt):
        # read image feature
        
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "_res101.hdf5", 'r')
        feature = fid['feature_map'][()].squeeze()#.T
        label = fid['labels'][()]#.reshape(-1,1) 
        trainval_loc = fid['trainval_loc'][()] #+ 1 
        #train_loc = fid['train_loc'][()] 
        #val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] #+ 1 
        test_unseen_loc = fid['test_unseen_loc'][()] #+ 1 
        fid.close()
        # read attributes
        #f = open('/cver/qzhe/data/semantic-embedding/pkl-files/{}_{}.pkl'.format(self.dataset, self.class_embedding),'rb')
        f = open(opt.dataroot + "/" +'{}_{}.pkl'.format(opt.dataset, opt.class_embedding),'rb')
        a = pickle.load(f)
        f.close()
        if opt.extra_class != 0:
            f = open('/cver/qzhe/data/semantic-embedding/notebooks-getting-extra-embedding/{}_{}.pkl'.format(self.dataset, self.class_embedding),'rb')
            b = pickle.load(f)
            f.close()
        self.attribute = torch.from_numpy(a).float()
        self.aux_data = torch.from_numpy(a).float()
        if opt.extra_class != 0:
            self.attribute_extra = torch.cat((self.attribute, torch.from_numpy(b).float()),0).to(self.device)
            self.aux_data_extra = torch.cat((self.attribute, torch.from_numpy(b).float()),0).to(self.device)  
        else:
            self.attribute_extra = self.attribute.to(self.device)
            self.aux_data_extra = self.attribute.to(self.device)  
            
        self.attribute = self.attribute.to(self.device)
        self.aux_data = self.aux_data.to(self.device)
        if opt.sample_level:
            with open('/cver/qzhe/data/semantic-embedding/pkl-files/{}.pkl'.format(opt.sample_level_emb), 'rb') as f:
                sample_attribute = pickle.load(f)
            self.sample_attribute = torch.from_numpy(sample_attribute).float()
        
        self.feature = feature
        self.label = label


        self.train_feature = feature[trainval_loc] 
        self.train_label = label[trainval_loc] 
        if opt.sample_level:
            self.train_sample_attribute = torch.from_numpy(sample_attribute[trainval_loc]).to(torch.float32)#.float()
        self.test_unseen_feature = feature[test_unseen_loc] 
        self.test_unseen_label = label[test_unseen_loc] 
        if opt.sample_level:
            self.test_unseen_sample_attribute = torch.from_numpy(sample_attribute[test_unseen_loc]).to(torch.float32)#.float()
        self.test_seen_feature = feature[test_seen_loc] 
        self.test_seen_label = label[test_seen_loc] 
        if opt.sample_level:
            self.test_seen_sample_attribute = torch.from_numpy(sample_attribute[test_seen_loc]).to(torch.float32)#.float()
        
        self.num = len(label)
        train_num = len(self.train_label)
        print('总AWA2 数据集数量',self.num)
        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.allclasses = torch.from_numpy(np.unique(self.label)) 
        
        if opt.extra_class != 0:
            print('Loading extra data...')
            fid = h5py.File(self.data_path + "/" + "extra/"+self.dataset + "/" + "res101.hdf5", 'r')
            print('Extra data loaded.')
            feature = fid['feature_map'][()].squeeze()
            label = (np.array(fid['labels'][()])-1)
            fid.close()
        
        if opt.extra_class != 0 and opt.sample_level:
            with open('/cver/qzhe/data/semantic-embedding/notebooks-getting-extra-embedding/{}.pkl'.format(opt.sample_level_emb), 'rb') as f:
                sample_attribute = pickle.load(f)
            extra_feature,extra_label,extra_sample_attribute = choose_extra_samples_sample_level(feature, label, sample_attribute, opt.extra_class, opt.extra_num_per_class, opt.filter_lower_boundary)
        elif opt.extra_class != 0:
            extra_feature,extra_label = choose_extra_samples(feature, label, opt.extra_class, opt.extra_num_per_class, opt.filter_lower_boundary)
        print('总可用 Extra 数据样本量',len(label))
        
        if opt.extra_class != 0:
            self.train_feature_extra = np.concatenate((self.train_feature,extra_feature),0)
            self.train_label_extra = np.concatenate((self.train_label,extra_label),0)
            if opt.sample_level:
                self.train_sample_attribute_extra = torch.from_numpy(np.concatenate((self.train_sample_attribute.numpy(), extra_sample_attribute),0)).to(torch.float32)
            self.label_extra = np.concatenate((self.label,extra_label),0)
        else:
            self.train_feature_extra = self.train_feature
            self.train_label_extra = self.train_label#np.concatenate((,extra_label),0)
            if opt.sample_level:
                self.train_sample_attribute_extra = self.train_sample_attribute.to(torch.float32)
            self.label_extra = self.label #np.concatenate((self.label,extra_label),0)
            
        self.train_feature = torch.from_numpy(self.train_feature)
            
        self.nclasses = len(self.seenclasses)
        
        
        scaler = preprocessing.MinMaxScaler()
        _train_feature_extra = scaler.fit_transform(self.train_feature_extra)
        _train_feature = scaler.transform(self.train_feature)
        _test_seen_feature = scaler.transform(self.test_seen_feature)
        _test_unseen_feature = scaler.transform(self.test_unseen_feature)
        self.train_feature_extra = torch.from_numpy(_train_feature_extra).float().to(self.device)
        mx = self.train_feature_extra.max()
        self.train_feature_extra.mul_(1 / mx) 
        self.train_feature = torch.from_numpy(_train_feature).float().to(self.device)  
        self.train_feature.mul_(1 / mx)                                                     # train seen feature
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float().to(self.device)  
        self.test_unseen_feature.mul_(1 / mx)                                             # test unseen feature
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float().to(self.device)  
        self.test_seen_feature.mul_(1 / mx)                                              # test seen feature

        self.train_label_extra = torch.from_numpy(self.train_label_extra).long().to(self.device)             # train seen label
        self.train_label = torch.from_numpy(self.train_label).long().to(self.device)             # train seen label
        self.test_seen_label = torch.from_numpy(self.test_seen_label).long().to(self.device)        # test seen label
        self.test_unseen_label = torch.from_numpy(self.test_unseen_label).long().to(self.device)    # test unseen label
        
        self.seenclasses = torch.from_numpy(self.seenclasses)            # seen classes
        self.unseenclasses = torch.from_numpy(self.unseenclasses)    # unseen classes

        self.seenclasses_extra = torch.from_numpy(np.unique(self.train_label_extra.cpu().numpy()))        # seen classes
        self.allclasses_extra = torch.from_numpy(np.unique(self.label_extra))      # all classes

        # 类别数量（40，10，50）
        self.seenclass_num = self.seenclasses.size(0)       # number of seen classes
        self.unseenclass_num = self.unseenclasses.size(0)   # number of unseen classes
        self.allclasses_num = self.allclasses.size(0)       # number of all classes
        
        self.seenclass_num_extra = self.seenclasses_extra.size(0)       # number of seen classes
        self.allclasses_num_extra = self.allclasses_extra.size(0)       # number of all classes
        # semantic embedding 对应的 seen/unseen
        self.seen_att = self.attribute[self.seenclasses]        # attribute of seen classes
        if opt.extra_class != 0:
            self.seen_att_extra = torch.cat((self.seen_att,torch.from_numpy(b).to(self.device).float()),0)    # attribute of seen classes
        else:
            self.seen_att_extra = self.seen_att
        self.unseen_att = self.attribute[self.unseenclasses]    # attribute of unseen classes
        
        self.feature_dim = self.train_feature.shape[1]                  # dim of feature
        self.att_dim = self.attribute.shape[1]   
        self.ntrain_extra = int(self.train_label_extra.numel())
        
        self.data = {}
        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

    def read_matdataset(self):

        print('CRAP!!! Why here?')
        matcontent = sio.loadmat(self.datadir + 'res101.mat')
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(self.datadir + self.auxiliary_data_source + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    
        self.attribute = torch.from_numpy(matcontent['att'].T).float().to(self.device)
        self.aux_data = torch.from_numpy(matcontent['att'].T).float().to(self.device)
        self.feature = feature
        self.label = label
       
    
        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(self.feature[trainval_loc])
        _test_seen_feature = scaler.transform(self.feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(self.feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float().to(self.device)
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)                                                     # train seen feature
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float().to(self.device)
        self.test_unseen_feature.mul_(1 / mx)                                               # test unseen feature
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float().to(self.device)
        self.test_seen_feature.mul_(1 / mx)                                                 # test seen feature

        self.train_label = torch.from_numpy(self.label[trainval_loc]).long().to(self.device)             # train seen label
        self.test_seen_label = torch.from_numpy(self.label[test_seen_loc]).long().to(self.device)        # test seen label
        self.test_unseen_label = torch.from_numpy(self.label[test_unseen_loc]).long().to(self.device)    # test unseen label

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.cpu().numpy()))            # seen classes
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.cpu().numpy()))    # unseen classes
        self.allclasses = torch.from_numpy(np.unique(self.label))      # all classes


        self.seenclass_num = self.seenclasses.size(0)       # number of seen classes
        self.unseenclass_num = self.unseenclasses.size(0)   # number of unseen classes
        self.allclasses_num = self.allclasses.size(0)       # number of all classes

        self.seen_att = self.attribute[self.seenclasses]        # attribute of seen classes
        self.unseen_att = self.attribute[self.unseenclasses]    # attribute of unseen classes

        self.feature_dim = self.train_feature.shape[1]                  # dim of feature
        self.att_dim = self.attribute.shape[1]                          # dim of attribute
        print(self.train_label.numel())
        self.ntrain = int(self.train_label.numel())

        
        self.data = {}
        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None
        # 只是数据而已，self.data[][] 对应哪些数据可用罢了
        # 此处向下需要观察，在干什么用。前面和 f-CLSWGAN 基本一样，可替换（extra/hdf5）
        
        # 这里只是每类对应的 embedding
        #self.unseenclass_aux_data = self.aux_data[self.unseenclasses]
        #self.seenclass_aux_data = self.aux_data[self.seenclasses]


    # hia?? 好像是支持各种不同的 semantic embedding
    # 看到了 att, stc_emb, w2v, glove, wordnet_hierarchy
    def transfer_features(self, n, num_queries='num_features'):
        print('size before')
        print(self.data['test_unseen']['resnet_features'].size())
        print(self.data['train_seen']['resnet_features'].size())


        print('o'*100)
        print(self.data['test_unseen'].keys())
        for i,s in enumerate(self.unseenclasses):

            features_of_that_class   = self.data['test_unseen']['resnet_features'][self.data['test_unseen']['labels']==s ,:]

            if 'att' == self.auxiliary_data_source:
                attributes_of_that_class = self.data['test_unseen']['att'][self.data['test_unseen']['labels']==s ,:]
                use_att = True
            else:
                use_att = False
            if 'sentences' == self.auxiliary_data_source:
                sentences_of_that_class = self.data['test_unseen']['sentences'][self.data['test_unseen']['labels']==s ,:]
                use_stc = True
            else:
                use_stc = False
            if 'word2vec' == self.auxiliary_data_source:
                word2vec_of_that_class = self.data['test_unseen']['word2vec'][self.data['test_unseen']['labels']==s ,:]
                use_w2v = True
            else:
                use_w2v = False
            if 'glove' == self.auxiliary_data_source:
                glove_of_that_class = self.data['test_unseen']['glove'][self.data['test_unseen']['labels']==s ,:]
                use_glo = True
            else:
                use_glo = False
            if 'wordnet' == self.auxiliary_data_source:
                wordnet_of_that_class = self.data['test_unseen']['wordnet'][self.data['test_unseen']['labels']==s ,:]
                use_hie = True
            else:
                use_hie = False


            num_features = features_of_that_class.size(0)

            indices = torch.randperm(num_features)

            if num_queries!='num_features':

                indices = indices[:n+num_queries]


            print(features_of_that_class.size())


            if i==0:

                new_train_unseen      = features_of_that_class[   indices[:n] ,:]

                if use_att:
                    new_train_unseen_att  = attributes_of_that_class[ indices[:n] ,:]
                if use_stc:
                    new_train_unseen_stc  = sentences_of_that_class[ indices[:n] ,:]
                if use_w2v:
                    new_train_unseen_w2v  = word2vec_of_that_class[ indices[:n] ,:]
                if use_glo:
                    new_train_unseen_glo  = glove_of_that_class[ indices[:n] ,:]
                if use_hie:
                    new_train_unseen_hie  = wordnet_of_that_class[ indices[:n] ,:]


                new_train_unseen_label  = s.repeat(n)

                new_test_unseen = features_of_that_class[  indices[n:] ,:]

                new_test_unseen_label = s.repeat( len(indices[n:] ))

            else:
                new_train_unseen  = torch.cat(( new_train_unseen             , features_of_that_class[  indices[:n] ,:]),dim=0)
                new_train_unseen_label  = torch.cat(( new_train_unseen_label , s.repeat(n)),dim=0)

                new_test_unseen =  torch.cat(( new_test_unseen,    features_of_that_class[  indices[n:] ,:]),dim=0)
                new_test_unseen_label = torch.cat(( new_test_unseen_label  ,s.repeat( len(indices[n:]) )) ,dim=0)

                if use_att:
                    new_train_unseen_att    = torch.cat(( new_train_unseen_att   , attributes_of_that_class[indices[:n] ,:]),dim=0)
                if use_stc:
                    new_train_unseen_stc    = torch.cat(( new_train_unseen_stc   , sentences_of_that_class[indices[:n] ,:]),dim=0)
                if use_w2v:
                    new_train_unseen_w2v    = torch.cat(( new_train_unseen_w2v   , word2vec_of_that_class[indices[:n] ,:]),dim=0)
                if use_glo:
                    new_train_unseen_glo    = torch.cat(( new_train_unseen_glo   , glove_of_that_class[indices[:n] ,:]),dim=0)
                if use_hie:
                    new_train_unseen_hie    = torch.cat(( new_train_unseen_hie   , wordnet_of_that_class[indices[:n] ,:]),dim=0)



        print('new_test_unseen.size(): ', new_test_unseen.size())
        print('new_test_unseen_label.size(): ', new_test_unseen_label.size())
        print('new_train_unseen.size(): ', new_train_unseen.size())
        #print('new_train_unseen_att.size(): ', new_train_unseen_att.size())
        print('new_train_unseen_label.size(): ', new_train_unseen_label.size())
        print('>> num unseen classes: ' + str(len(self.unseenclasses)))

        #######
        ##
        #######

        self.data['test_unseen']['resnet_features'] = copy.deepcopy(new_test_unseen)
        #self.data['train_seen']['resnet_features']  = copy.deepcopy(new_train_seen)

        self.data['test_unseen']['labels'] = copy.deepcopy(new_test_unseen_label)
        #self.data['train_seen']['labels']  = copy.deepcopy(new_train_seen_label)

        self.data['train_unseen']['resnet_features'] = copy.deepcopy(new_train_unseen)
        self.data['train_unseen']['labels'] = copy.deepcopy(new_train_unseen_label)
        self.ntrain_unseen = self.data['train_unseen']['resnet_features'].size(0)

        if use_att:
            self.data['train_unseen']['att'] = copy.deepcopy(new_train_unseen_att)
        if use_w2v:
            self.data['train_unseen']['word2vec']   = copy.deepcopy(new_train_unseen_w2v)
        if use_stc:
            self.data['train_unseen']['sentences']  = copy.deepcopy(new_train_unseen_stc)
        if use_glo:
            self.data['train_unseen']['glove']      = copy.deepcopy(new_train_unseen_glo)
        if use_hie:
            self.data['train_unseen']['wordnet']   = copy.deepcopy(new_train_unseen_hie)

        ####
        self.data['train_seen_unseen_mixed'] = {}
        self.data['train_seen_unseen_mixed']['resnet_features'] = torch.cat((self.data['train_seen']['resnet_features'],self.data['train_unseen']['resnet_features']),dim=0)
        self.data['train_seen_unseen_mixed']['labels'] = torch.cat((self.data['train_seen']['labels'],self.data['train_unseen']['labels']),dim=0)

        self.ntrain_mixed = self.data['train_seen_unseen_mixed']['resnet_features'].size(0)

        if use_att:
            self.data['train_seen_unseen_mixed']['att'] = torch.cat((self.data['train_seen']['att'],self.data['train_unseen']['att']),dim=0)
        if use_w2v:
            self.data['train_seen_unseen_mixed']['word2vec'] = torch.cat((self.data['train_seen']['word2vec'],self.data['train_unseen']['word2vec']),dim=0)
        if use_stc:
            self.data['train_seen_unseen_mixed']['sentences'] = torch.cat((self.data['train_seen']['sentences'],self.data['train_unseen']['sentences']),dim=0)
        if use_glo:
            self.data['train_seen_unseen_mixed']['glove'] = torch.cat((self.data['train_seen']['glove'],self.data['train_unseen']['glove']),dim=0)
        if use_hie:
            self.data['train_seen_unseen_mixed']['wordnet'] = torch.cat((self.data['train_seen']['wordnet'],self.data['train_unseen']['wordnet']),dim=0)

class SampleFeatDataLayer(object):
    def __init__(self, feat_data, label, att, batch_size):
        """Set the roidb to be used by this layer during training."""
        #print(len(label), feat_data.shape[0], att.shape[0])
        assert len(label) == feat_data.shape[0] == att.shape[0]
        self.batch_size = batch_size
        self._feat_data = feat_data
        self._label = label
        self._att = att#[label]
        # self.attribute_extra[batch_label]
        self._shuffle_roidb_inds()
        self._epoch = 0
        # print('----data-type-----', type(self._feat_data))

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        minibatch_att = np.array([self._att[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'att': minibatch_att, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label, 'att': self._att}
        return blobs
class ClsFeatDataLayer(object):
    def __init__(self, feat_data, label, att, batch_size):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self.batch_size = batch_size
        self._feat_data = feat_data
        self._label = label
        self._att = att[label]
        # self.attribute_extra[batch_label]
        self._shuffle_roidb_inds()
        self._epoch = 0
        # print('----data-type-----', type(self._feat_data))

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self.batch_size]
        self._cur += self.batch_size

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        minibatch_att = np.array([self._att[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'att': minibatch_att, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label, 'att': self._att}
        return blobs
