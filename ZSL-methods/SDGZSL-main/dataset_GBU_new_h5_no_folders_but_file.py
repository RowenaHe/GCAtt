import numpy as np
import h5py
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import pickle
from sklearn.cluster import KMeans
import sklearn.metrics as sm

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


class DATA_LOADER(object):
    def __init__(self, opt):
        self.finetune = opt.finetune
        '''
        if opt.dataset in ['FLO_EPGN','CUB_STC']:
            if self.finetune:
                self.read_fine_tune(opt)
            else:
                self.read(opt)
        elif opt.dataset in ['CUB', 'AWA2', 'APY', 'FLO', 'SUN']:
            if opt.matdataset == True:
                # 直接读取 mat-dataset
                print('mat file reading')
                self.read_matdataset(opt)
            else:
                print('hdf5 file reading')
                self.read_h5dataset(opt)
                '''
        self.read_h5dataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)

    
    def read_fine_tune(self,opt):
        # QZHe deleted
        pass
        
    def read(self, opt):
        # QZHe deleted
        pass
                
    # QZHe version
    def read_h5dataset(self, opt):
        # read image feature
        #fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "_" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature_map'][()].squeeze()#.T
        label = fid['labels'][()]#.reshape(-1,1) 
        trainval_loc = fid['trainval_loc'][()] #+ 1 
        #train_loc = fid['train_loc'][()] 
        #val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] #+ 1 
        test_unseen_loc = fid['test_unseen_loc'][()] #+ 1 
        
        if opt.sample_level:
            #sample_attribute = fid[opt.sample_level_emb][()]
            with open('/cver/qzhe/data/semantic-embedding/pkl-files/{}.pkl'.format(opt.sample_level_emb), 'rb') as f:
                sample_attribute = pickle.load(f)
            
            
        # read attributes
        if opt.class_embedding == 'att':
            a = fid['att'][()]
        else:
            #f = open('/cver/qzhe/data/semantic-embedding/pkl-files/{}_{}.pkl'.format(opt.dataset, opt.class_embedding),'rb')
            f = open(opt.dataroot + "/" +'{}_{}.pkl'.format(opt.dataset, opt.class_embedding),'rb')
            a = pickle.load(f)
            f.close()
        self.attribute = torch.from_numpy(a).float() # att / w2v / glove
        fid.close()
        
        self.feature = feature
        self.label = label

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

        #f-CLSWGAN
        self.seenclass_num = self.seenclasses.size(0)       # number of seen classes
        self.unseenclass_num = self.unseenclasses.size(0)   # number of unseen classes
        self.allclasses_num = self.allclasses.size(0)       # number of all classes
        self.seen_att = self.attribute[self.seenclasses]        # attribute of seen classes
        self.unseen_att = self.attribute[self.unseenclasses]    # attribute of unseen classes
        self.feature_dim = self.train_feature.shape[1]                  # dim of feature
        self.att_dim = self.attribute.shape[1]                          # dim of attribute
        print(self.train_label.numel())
        self.ntrain = int(self.train_label.numel())

        
        # SDGZSL 专属映射
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        #self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()



    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        label = matcontent['labels'].astype(int).squeeze() - 1
        #if self.finetune:
            #matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + "_finetuned.mat")
            # label = matcontent['labels'].astype(int).squeeze() - 1

        feature = matcontent['features'].T
        #if opt.dataset == "APY" and self.finetune:
            #feature = feature.T
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        self.attribute = torch.from_numpy(matcontent['att'].T).float()


        scaler = preprocessing.MinMaxScaler()
        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1 / mx)
        self.train_label = torch.from_numpy(label[trainval_loc]).long()
        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)

        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att = self.attribute[self.unseenclasses].numpy()


class FeatDataLayer(object):
    def __init__(self, label, feat_data,  opt):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs


class SampleFeatDataLayer(object):
    def __init__(self, feat_data, label, att, opt):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0] == att.shape[0]
        self._opt = opt
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
        if self._cur + self._opt.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batch_size]
        self._cur += self._opt.batch_size

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self._opt.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batch_size]
        self._cur += self._opt.batch_size

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        minibatch_att = np.array([self._att[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'att': minibatch_att, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label, 'att': self._att}
        return blobs
class ClsFeatDataLayer(object):
    def __init__(self, feat_data, label, att, opt):
        """Set the roidb to be used by this layer during training."""
        assert len(label) == feat_data.shape[0]
        self._opt = opt
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
        if self._cur + self._opt.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batch_size]
        self._cur += self._opt.batch_size

        return db_inds

    def forward(self):
        new_epoch = False
        if self._cur + self._opt.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batch_size]
        self._cur += self._opt.batch_size

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        minibatch_att = np.array([self._att[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels': minibatch_label, 'att': minibatch_att, 'newEpoch': new_epoch, 'idx': db_inds}
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label, 'att': self._att}
        return blobs