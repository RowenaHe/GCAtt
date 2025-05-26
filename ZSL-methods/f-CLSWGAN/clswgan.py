from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util_sample_level_test_emb_no_folders_but_file as util #util_test_emb_no_folders_but_file as util 
import classifier
import classifier2
import sys
import model
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#break
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/cver/qzhe/GCAtt-opensource/data', help='path to dataset')
parser.add_argument('--matdataset', action='store_false', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
#parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
#parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
#parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

# mode-aug classification 也需要 sample-level 数据的读取
# mode-level, sample-level , mode-/cls- rectified sample-level都在这里
parser.add_argument('--sample_level', action='store_true', default=False, help='load sample-level data?') 
parser.add_argument('--save_for_visual', action='store_true', default=False, help='acc 增幅') 

# 关于 sample-level, mode-level, cls-/mode- rectified sample-level 训练数据的超参数
parser.add_argument('--sample_level_train', action='store_true', default=False, help='use sample level semantics in ZSL model training?')



# vision+semantics 分类
parser.add_argument('--semantics_for_classifier', action='store_true', default=False, help='use semantics in addition to vision for classification?') 
parser.add_argument('--cls_sample_att1', action='store_true', default=False, help='default: only seen -> GZSL| True: seen+unseen -> GZSL+TZSL')

#parser.add_argument('--mode_level_lab', default='mode_level_CLIP_att_label', help='name of the utilized mode level labels (key name in the .hdf5 file)')
opt = parser.parse_args()
# 读取的 sample-level  semantics 一定和 class-level semantics 对应，命名也一致
opt.sample_level_emb = 'sample_level_'+opt.dataset+'_'+opt.class_embedding
print(opt)



try:
    os.makedirs(opt.outf)
except OSError:
    pass
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

data = util.DATA_LOADER(opt)

def cluster_for_mode_aug_classification(cluster_source, labels, k_num_cluster):
    mode_level_label = np.zeros_like(labels, dtype = labels.dtype)
    for idx,lab in enumerate(torch.from_numpy(labels).unique()):
        use_idx = labels == lab.item()
        X = cluster_source[use_idx]
        iris_k_mean_model = KMeans(n_clusters=k_num_cluster, init='k-means++')
        iris_k_mean_model.fit(X)
        cluster_labels = iris_k_mean_model.labels_
        mode_level_label[use_idx] = lab.item()*k_num_cluster + cluster_labels
    return mode_level_label
    

# load data    
if opt.sample_level_train:
    data_layer = util.SampleFeatDataLayer(data.train_feature.numpy(), data.train_label.numpy(), data.train_sample_attribute.numpy(), opt)#
else:
    data_layer = util.ClsFeatDataLayer(data.train_feature.numpy(), data.train_label.numpy(), data.attribute.numpy(), opt)#

    

opt.nz = data.attribute.shape[-1]
opt.attSize = data.attribute.shape[-1]
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1,dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att = iclass_att.cuda().repeat(num, 1)
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(Variable(syn_noise), Variable(syn_att))
        syn_feature.narrow(0, i*num, num).copy_(output.data)
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1).cuda().expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# train a classifier on seen classes, obtain \theta of Equation (4)
#pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
#for p in pretrain_cls.model.parameters(): # set requires_grad to False
    #p.requires_grad = False
_list = []
for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    for i in tqdm(range(0, data.ntrain, opt.batch_size)):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            #sample()
            blobs = data_layer.forward()
            batch_feature = torch.from_numpy(blobs['data'])
            batch_label = torch.from_numpy(blobs['labels'])
            batch_att = torch.from_numpy(blobs['att'])
            
            input_res.copy_(batch_feature)
            input_att.copy_(batch_att)
            input_label.copy_(util.map_label(batch_label, data.seenclasses))
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)
            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()
        
        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        #c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))
        errG = G_cost #+ opt.cls_weight*c_errG
        errG.backward()
        optimizerG.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f'
              % (epoch, opt.nepoch, D_cost.item(), G_cost.item(), Wasserstein_D.item()))#, c_errG:%.4f, c_errG.item()

    netG.eval()
    if opt.gzsl:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        if opt.semantics_for_classifier:
            if opt.cls_sample_att1:
                train_XS = torch.cat((data.train_sample_attribute, data.attribute[syn_label]), 0)
            else:
                train_XS = data.attribute[train_Y]
            train_X = torch.cat((train_X, train_XS),1)
        nclass = data.allclasses_num
        cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        _list.append(cls.H)
    # Zero-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num) 
        if opt.semantics_for_classifier:
            syn_att = data.attribute[syn_label]
            syn_feature = torch.cat((syn_feature, syn_att), 1)
        cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt, opt.classifier_lr, 0.5, 50, opt.syn_num, False)
        acc = cls.acc
        print('unseen class accuracy= ', acc)
        _list.append(acc)

        
    netG.train()
    
    #plt.plot(np.arange(epoch+1), _list)
    
    #plt.xlim((0,opt.nepoch))
    #plt.ylim((0,0.9))
    #设置坐标轴名称
    #plt.xlabel('epoch')
    #plt.ylabel('acc')
    #设置坐标轴刻度
    #plt.yticks(np.arange(0,0.9,0.1))
    #plt.savefig('savefig/'+opt.dataset+'__'+opt.class_embedding+'__gzsl='+str(opt.gzsl))
print('BEST = ',max(_list))
#plt.savefig('savefig/'+opt.dataset+'__'+opt.class_embedding+'__gzsl='+str(opt.gzsl)+'__BEST='+str(max(_list).item())+'.png')
if opt.save_for_visual:
    np.save('sample-level-stability/{}_sample-level-{}.npy'.format(opt.dataset, opt.sample_level_train),np.array(_list))