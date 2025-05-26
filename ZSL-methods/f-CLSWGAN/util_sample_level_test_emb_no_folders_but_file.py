#import h5py
import numpy as np
import h5py
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import pickle
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.preprocessing import normalize
import scipy.spatial.distance as scidis
from sklearn.metrics.pairwise import cosine_similarity #,euclidean_distances



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes): # 训练用的 seen classes 并不是从 1 开始往后排的，这里把它们映射成从 1 开始向后排
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

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
        if opt.matdataset == True:
            self.read_matdataset(opt)
        else:
            print('hdf5 file reading')
            self.read_h5dataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
                
    # QZHe version
    def read_h5dataset(self, opt):
        # read image feature
        #fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "_" + opt.image_embedding + ".hdf5", 'r')
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
        #f = open('/cver/qzhe/data/semantic-embedding/pkl-files/{}_{}.pkl'.format(opt.dataset, opt.class_embedding),'rb')
        f = open(opt.dataroot + "/" +'{}_{}.pkl'.format(opt.dataset, opt.class_embedding),'rb')
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

def _distance(data1, data2, dm='euclidean_distance'):
    assert data1.shape == data2.shape
    if dm == 'euclidean_distance':
        dist = np.linalg.norm(data2 - data1, axis=-1, keepdims=True)
        # pass
    elif dm =='cosine_similarity':
        dist = 1/np.diag(cosine_similarity(data2, data1)).reshape(-1,1)
        # pass
    else:
        print('Distance Metric not supported.')
        return
    return dist



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