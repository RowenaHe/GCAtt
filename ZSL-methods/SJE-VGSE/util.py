#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import h5py
import pickle

def calibrated_stacking(opt, output, lam=1e-3):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    output = output.cpu().numpy()
    seen_L = list(set(opt.test_seen_label.numpy()))
    output[:, seen_L] = output[:, seen_L] - lam
    return torch.from_numpy(output)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    
    # print("label", label)
    # print("classes", classes.shape)
    # print("mapped_label", mapped_label)
    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            self.read_matdataset(opt)
        else:
            self.read_h5dataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
         
    # QZHe version
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File('/cver/qzhe/GCAtt-opensource/data/' + opt.dataset + "_" + opt.image_embedding + ".hdf5", 'r')
        print(fid.keys())
        feature = fid['feature_map'][()].squeeze()#.T
        label = fid['labels'][()]#.reshape(-1,1) 
        trainval_loc = fid['trainval_loc'][()] #+ 1 
        #train_loc = fid['train_loc'][()] 
        #val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] #+ 1 
        test_unseen_loc = fid['test_unseen_loc'][()] #+ 1 
        fid.close()
        
        self.trainval_loc = trainval_loc
        self.test_seen_loc = test_seen_loc
        self.test_unseen_loc = test_unseen_loc
        
        
        if opt.sample_level:
            #sample_attribute = fid[opt.sample_level_emb][()]
            with open('/cver/qzhe/data/semantic-embedding/pkl-files/{}.pkl'.format(opt.sample_level_emb), 'rb') as f:
                sample_attribute = pickle.load(f)

        # 要特别注意下一致性。我们只准备写一个 res101.hdf5，在里面保存所有的 emb，其 keys() 命名要注意别错
        # read attributes
        f = open('/cver/qzhe/GCAtt-opensource/data/' + '{}_{}.pkl'.format(opt.dataset, opt.class_embedding),'rb')
        a = pickle.load(f)
        f.close()
        self.attribute = torch.from_numpy(a).float() # att / w2v / glove
        
        self.feature = feature
        self.label = label
        # 发现 feature, label, attribute 都变成了 torch.tensor，同转。
        if opt.sample_level:
            self.sample_attribute = torch.from_numpy(sample_attribute).float()

        self.train_feature = feature[trainval_loc] 
        self.train_label = label[trainval_loc] 
        if opt.sample_level:
            self.train_sample_attribute = torch.from_numpy(sample_attribute[trainval_loc])
        self.test_unseen_feature = feature[test_unseen_loc] 
        self.test_unseen_label = label[test_unseen_loc] 
        if opt.sample_level:
            self.test_unseen_sample_attribute = torch.from_numpy(sample_attribute[test_unseen_loc])
        self.test_seen_feature = feature[test_seen_loc] 
        self.test_seen_label = label[test_seen_loc] 
        if opt.sample_level:
            self.test_seen_sample_attribute = torch.from_numpy(sample_attribute[test_seen_loc])

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        #print()
        self.nclasses = len(self.seenclasses)
        
        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(self.feature[trainval_loc])
        _test_seen_feature = scaler.transform(self.feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(self.feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)                                                     # train seen feature
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)                                               # test unseen feature
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)                                                 # test seen feature


        self.train_label = torch.from_numpy(self.label[trainval_loc]).long()             # train seen label
        self.test_seen_label = torch.from_numpy(self.label[test_seen_loc]).long()        # test seen label
        self.test_unseen_label = torch.from_numpy(self.label[test_unseen_loc]).long()    # test unseen label

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))            # seen classes
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))    # unseen classes
        self.allclasses = torch.from_numpy(np.unique(self.label))      # all classes

        self.train_local_label = map_label(self.train_label, self.seenclasses)                 # train local label
        self.test_seen_local_label = map_label(self.test_seen_label, self.seenclasses)         # test seen local label
        self.test_unseen_local_label = map_label(self.test_unseen_label, self.unseenclasses)   # test unseen local label

        self.seenclass_num = self.seenclasses.size(0)       # number of seen classes
        self.unseenclass_num = self.unseenclasses.size(0)   # number of unseen classes
        self.allclasses_num = self.allclasses.size(0)       # number of all classes

        self.seen_att = self.attribute[self.seenclasses]        # attribute of seen classes
        self.unseen_att = self.attribute[self.unseenclasses]    # attribute of unseen classes

        self.feature_dim = self.train_feature.shape[1]                  # dim of feature
        self.att_dim = self.attribute.shape[1]                          # dim of attribute
        print(self.train_label.numel())
        self.ntrain = int(self.train_label.numel())
    def read_matdataset(self, opt):

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        print("using the matcontent:", opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")

        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        print("using the matcontent:", opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        self.trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        self.train_loc = matcontent['train_loc'].squeeze() - 1
        self.val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        self.test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        # print("_______________unseen image numbers:", len(self.test_unseen_loc))
        # self.allclasses_name = matcontent['allclasses_names']
        # print("________________using the attribute: att")
        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[self.trainval_loc])
        _test_seen_feature = scaler.transform(feature[self.test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[self.test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[self.trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(label[self.test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(label[self.test_seen_loc]).long()
        
        self.seenclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        # print("self.test_unseen_label:", list(set(self.test_unseen_label.numpy())))
        # print("self.unseenclasses:", list(set(self.unseenclasses.numpy())))
        # print("self.test_seen_label:", list(set(self.test_seen_label.numpy())))
        # print("self.seenclasses:", list(set(self.seenclasses.numpy())))

        self.ntrain = self.train_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att
