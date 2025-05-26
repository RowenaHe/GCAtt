import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from util import calibrated_stacking

import copy
from sklearn.preprocessing import normalize

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, opt, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=True):
        self.train_X = _train_X
        self.train_Y = _train_Y
        if opt.sample_level_train:
            self.train_att = data_loader.train_sample_attribute
            #print('train att:',self.train_att.shape, self.train_att)
        
        # 待修改
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        
        #待修改
        self.attribute = data_loader.attribute
        self.attri_dim = self.attribute.size(1)
        self.attribute_zsl = self.prepare_attri_label(self.attribute, self.unseenclasses) # (att_dim, n_unseen_class)  ZSL 测试用
        self.attribute_seen = self.prepare_attri_label(self.attribute, self.seenclasses) # (att_dim, n_seen_class)  训练用
        self.attribute_gzsl = torch.transpose(self.attribute, 1, 0) # (att_dim, n_allclass)    GZSL 测试用
        
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX_ALE(self.input_dim, self.attri_dim)
        self.best_model = LINEAR_LOGSOFTMAX_ALE(self.input_dim, self.attri_dim)
        self.model.apply(util.weights_init)
        if opt.pretrained:
            self.model.load_state_dict(
                torch.load(os.path.join(opt.checkpointroot, "{}_{}.pth".format(opt.pretrained, "cls"))))

        self.criterion = nn.CrossEntropyLoss()

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)
        self.att = torch.FloatTensor(_batch_size, self.attri_dim)
        self.opt = opt
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
            self.attribute_zsl = self.attribute_zsl.cuda()
            self.attribute_gzsl = self.attribute_gzsl.cuda()
            self.attribute_seen = self.attribute_seen.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def prepare_attri_label(self, attribute, classes):
        classes_dim = classes.size(0)
        output_attribute = torch.FloatTensor(classes_dim, self.attri_dim)
        for i in range(classes_dim):
            output_attribute[i] = attribute[classes[i]]
        return torch.transpose(output_attribute, 1, 0)

    def clean_attribute(self, attribute, factor):
        attribute = attribute.cpu().numpy()
        attribute[attribute < factor] = 0

        attribute = normalize(attribute, axis=1)
        attribute = torch.from_numpy(attribute)
        return attribute

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8

        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                if self.opt.sample_level_train:
                    batch_input, batch_label, batch_att = self.next_batch(self.batch_size)
                    #print(batch_input.shape, batch_label.shape, batch_att.shape)
                else:
                    batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                #print('batch att:', self.att.shape, self.att)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                if self.opt.sample_level_train:
                    self.att.copy_(batch_att)
                    attv = Variable(self.att)
                if self.opt.sample_level_train:
                    batch_sample_train_att = self.attribute_seen
                    for l in self.label.unique():
                        #print(self.att[self.label==l].mean(dim=0).shape)
                        #print(batch_sample_train_att[l].shape)
                        batch_sample_train_att[:,l] = self.att[self.label==l].mean(dim=0)
                    output = self.model(inputv, batch_sample_train_att)
                else:
                    output = self.model(inputv, self.attribute_seen)
                    
                loss = self.criterion(output, labelv)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            # print('Epoch: {}, acc: {:.2%}'.format(epoch, acc.item()))
            if acc > best_acc:
                best_acc = acc
                self.best_model.load_state_dict(self.model.state_dict())
        print('acc: {:.2%}'.format(best_acc.item()))
        return best_acc

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                if self.opt.sample_level_train:
                    batch_input, batch_label, batch_att = self.next_batch(self.batch_size)
                    #print(batch_input.shape, batch_label.shape, batch_att.shape)
                else:
                    batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                
                #print('batch att:', self.att.shape, self.att)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                if self.opt.sample_level_train:
                    self.att.copy_(batch_att)
                    attv = Variable(self.att)
                if self.opt.sample_level_train:
                    batch_sample_train_att = self.attribute_seen
                    for l in self.label.unique():
                        #print(self.att[self.label==l].mean(dim=0).shape)
                        #print(batch_sample_train_att[l].shape)
                        batch_sample_train_att[:,l] = self.att[self.label==l].mean(dim=0)
                    output = self.model(inputv, batch_sample_train_att)
                else:
                    output = self.model(inputv, self.attribute_seen)

                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if (acc_seen + acc_unseen) == 0:
                H = 0
            else:
                H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            # print('Epoch: {}, U: {:.2%}, S: {:.2%}, H: {:.2%}'.format(epoch, acc_unseen, acc_seen, H))
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                self.best_model.load_state_dict(self.model.state_dict())
        print('U: {:.2%}, S: {:.2%}, H: {:.2%}'.format(best_unseen, best_seen, best_H))
        return best_seen, best_unseen, best_H

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            if self.opt.sample_level_train:
                self.train_att = self.train_att[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
                if self.opt.sample_level_train:
                    att_rest_part = self.train_att[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            if self.opt.sample_level_train:
                self.train_att = self.train_att[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if self.opt.sample_level_train:
                att_new_part = self.train_att[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                if self.opt.sample_level_train:
                    return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0), torch.cat((att_rest_part, att_new_part), 0)
                else:
                    return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                if self.opt.sample_level_train:
                    return X_new_part, Y_new_part, att_new_part
                else:
                    return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            if self.opt.sample_level_train:
                return self.train_X[start:end], self.train_Y[start:end], self.train_att[start:end]
            else:
                return self.train_X[start:end], self.train_Y[start:end]

    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(Variable(test_X[start:end].cuda()), self.attribute_zsl)
                else:
                    output = self.model(Variable(test_X[start:end]), self.attribute_zsl)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                         target_classes.size(0))
        return acc

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]

        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            with torch.no_grad():
                if self.cuda:
                    output = self.model(Variable(test_X[start:end].cuda()), self.attribute_gzsl)
                else:
                    output = self.model(Variable(test_X[start:end]), self.attribute_gzsl)
                if self.opt.calibrated_stacking:
                    output = calibrated_stacking(self.opt, output, self.opt.calibrated_stacking)

            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]).item() / torch.sum(idx).item()
        acc_per_class /= target_classes.size(0)
        return acc_per_class

        # test_label is integer

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).item() / torch.sum(idx).item()
        return acc_per_class.mean()


class LINEAR_LOGSOFTMAX_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LINEAR_LOGSOFTMAX_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax()

    def forward(self, x, attribute):
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output
