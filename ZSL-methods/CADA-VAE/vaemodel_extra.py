#vaemodel
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader_extra import DATA_LOADER as dataloader
from data_loader_extra import ClsFeatDataLayer, SampleFeatDataLayer
import final_classifier_extra as  classifier
import models
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()
        #print('classifier: (', input_dim, nclass,')')

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters, opt):
        super(Model,self).__init__()
        self.device = hyperparameters['device']
        self.class_embedding = hyperparameters['class_embedding']
        # auxiliary_data_source = str 'attributes'
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.extra_class = hyperparameters['extra_class']
        self.extra_num_per_class = hyperparameters['extra_num_per_class']
        self.filter_lower_boundary = hyperparameters['filter_lower_boundary']
        self.classifier_batch_size = 32
        # 下面 4 个参数对全部数据集都为 200-0-400-0
        self.img_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][0] # 200
        self.att_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][1] # 0
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2] # 400 
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3] # 0
        # GZSL 为 200 0 400 0； TZSL 为 0 0 200 0
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        # load-data 在这里进行。传入的两个参数都是 string
        self.dataset = dataloader(opt, self.DATASET, copy.deepcopy(self.auxiliary_data_source) , self.extra_class, self.extra_num_per_class, 
                                  self.filter_lower_boundary, device= self.device, class_embedding=self.class_embedding)
        # CADAVAE 的 dataloader 略难顶，和 f-CLSWGAN 差这么多吗？？？
        # self.dataset 
        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_unseen_classes = 50
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_unseen_classes = 72
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_unseen_classes = 10
        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        # Here, the encoders and decoders for all modalities are created and put into dict
        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype],self.device)
            # print(str(datatype) + ' ' + str(dim))
        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype],self.device)
        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize +=  list(self.encoder[datatype].parameters())
            parameters_to_optimize +=  list(self.decoder[datatype].parameters())
        self.optimizer  = optim.Adam( parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)
    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1).to(self.device)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label

    def trainstep(self, img, att):

        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################

        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])]).to(self.device)

        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])]).to(self.device)

        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])]).to(self.device)

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        loss = reconstruction_loss - beta * KLD

        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor >0:
            loss += distance_factor*distance

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_vae(self, args):
        _list = []
        losses = []
        
        if args.sample_level_train:
            data_layer = SampleFeatDataLayer(self.dataset.train_feature_extra.cpu().numpy(), self.dataset.train_label_extra.cpu().numpy(), self.dataset.train_sample_attribute_extra.cpu().numpy(), self.batch_size)
        else:
            data_layer = ClsFeatDataLayer(self.dataset.train_feature_extra.cpu().numpy(), self.dataset.train_label_extra.cpu().numpy(), self.dataset.attribute_extra.cpu().numpy(), self.batch_size)
        
        #self.dataloader = data.DataLoader(self.dataset,batch_size= self.batch_size,shuffle= True,drop_last=True)#,num_workers = 4)
        #print('self.dataloader')
        self.dataset.unseenclasses =self.dataset.unseenclasses.long().to(self.device)
        self.dataset.seenclasses_extra =self.dataset.seenclasses_extra.long().to(self.device)
        self.dataset.seenclasses =self.dataset.seenclasses.long().to(self.device)
        #leave both statements
        self.train()
        self.reparameterize_with_noise = True
        best_acc_unseen, best_acc_seen, best_acc_H, best_acc_T = 0,0,0,0
        print('train for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch
            print('epoch: ', epoch)
            #print('=====epoch:', epoch)
            i=-1
            for iters in range(0, self.dataset.ntrain_extra, self.batch_size):
                i+=1
                blobs = data_layer.forward()
                imgs, atts, label = torch.from_numpy(blobs['data']), torch.from_numpy(blobs['att']), torch.from_numpy(blobs['labels'])
                label= label.long().to(self.device)
                # 光速结束，正常
                imgs = imgs.to(self.device)
                imgs.requires_grad = False
                atts = atts.to(self.device)
                atts.requires_grad = False
                #print('dtype=====', data_from_modalities[0].dtype, data_from_modalities[1].dtype)
                loss = self.trainstep(imgs, atts)
                # 正常
                
                #if i%50==0:
                    #print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+ ' | loss ' +  str(loss)[:5]   )

                if i%50==0 and i>0:
                    losses.append(loss)
                # 正常
            print('epoch ' + str(epoch) + '\t'+ ' | loss ' +  str(loss)[:5]   )
            if self.generalized:
                unseen, seen, H = self.train_classifier(args)
                #print('H:',H,type(H))
                _list.append(round(H,4))#.cpu().item()) #
                if H > best_acc_H:
                    best_acc_seen = seen
                    best_acc_unseen = unseen
                    best_acc_H = H
            else:
                acc = self.train_classifier(args)
                _list.append(round(acc,4))#.cpu().item())
                if acc > best_acc_T:
                    best_acc_T = acc
            #print('BEST: unseen=%.4f, seen=%.4f, h=%.4f ' % (best_acc_unseen, best_acc_seen, best_acc_H))
        
        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()
        if self.generalized:
            print('BEST: unseen=%.4f, seen=%.4f, h=%.4f ' % (best_acc_unseen, best_acc_seen, best_acc_H))
        else:
            print('BEST: acc=%.4f ' % (best_acc_T))

        #plt.plot(_list)
        #plt.savefig('save/extra-use-class='+str(self.extra_class)+'_'+self.DATASET+'-'+str(self.generalized))
        return losses
        
    def train_classifier(self, args,show_plots=False):
        history = []  # stores accuracies
        cls_seenclasses = self.dataset.seenclasses
        cls_unseenclasses = self.dataset.unseenclasses


        train_seen_feat = self.dataset.train_feature.to(self.device) #self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.train_label #self.dataset.data['train_seen']['labels']

        unseen_att = self.dataset.unseen_att  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seen_att = self.dataset.seen_att

        unseen_corresponding_labels = self.dataset.unseenclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)


        # The resnet_features for testing the classifier are loaded here
        unseen_test_feat = self.dataset.test_unseen_feature
        # self.dataset.data['test_unseen']['resnet_features'].to(self.device)  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.test_seen_feature 
        # self.dataset.data['test_seen']['resnet_features'].to(self.device)  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.test_seen_label #self.dataset.data['test_seen']['labels'].to(self.device)  # self.dataset.test_seen_label.to(self.device)
        test_unseen_label = self.dataset.test_unseen_label 
        #self.dataset.data['test_unseen']['labels'].to(self.device)  # self.dataset.test_novel_label.to(self.device)

        # 肯定用不上
        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']


        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels =list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:

            unseen_corresponding_labels = self.map_label(unseen_corresponding_labels, unseen_corresponding_labels)
            test_unseen_label = self.map_label(test_unseen_label, cls_unseenclasses)
            # map cls novelclasses last
            cls_unseenclasses = self.map_label(cls_unseenclasses, cls_unseenclasses)


        if self.generalized:
            if args.semantics_for_classifier:
                clf = LINEAR_LOGSOFTMAX(self.latent_size+seen_att.shape[1], self.num_classes)
            else:
                clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
        else:
            #print('mode: zsl')
            if args.semantics_for_classifier:
                clf = LINEAR_LOGSOFTMAX(self.latent_size+seen_att.shape[1], self.num_unseen_classes)
            else:
                clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_unseen_classes)
            


        clf.apply(models.weights_init)

        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False
            #print('unseen_test_feat',unseen_test_feat.device)
            mu1, var1 = self.encoder['resnet_features'](unseen_test_feat.to(self.device))
            test_unseen_X = self.reparameterize(mu1, var1).to(self.device).data
            test_unseen_Y = test_unseen_label.to(self.device)

            mu2, var2 = self.encoder['resnet_features'](seen_test_feat.to(self.device))
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)
            if args.semantics_for_classifier:
                #print('test seen/unseen X shape (before):', test_seen_X.shape, test_unseen_X.shape)
                test_seen_X = torch.cat((test_seen_X, self.dataset.test_seen_sample_attribute.to(self.device)), 1)
                test_unseen_X = torch.cat((test_unseen_X, self.dataset.test_unseen_sample_attribute.to(self.device)), 1)
                #print('test seen/unseen X shape (before):', test_seen_X.shape, test_unseen_X.shape)
            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))]).to(self.device)).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),
                                                         dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.FloatTensor([]).to(self.device), torch.LongTensor([]).to(self.device)


            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            img_seen_feat, img_seen_label = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label,self.img_seen_samples)

            # 还是认为不应该有 train_unseen_feat/label
            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples )

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                    unseen_att, unseen_corresponding_labels,self.att_unseen_samples )

            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seen_att, seen_corresponding_labels, self.att_seen_samples)

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.FloatTensor([]).to(self.device)

            # print('img_seen_feat',img_seen_feat.device)
            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            # 继续删除
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])

            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])

            train_Z = [z_seen_img, z_unseen_img ,z_seen_att    ,z_unseen_att]
            train_L = [img_seen_label    , img_unseen_label,att_seen_label,att_unseen_label]

            # empty tensors are sorted out
            if self.generalized:
                train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
                train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]
            else:
                train_X = [train_Z[i] for i in [1,3] if train_Z[i].size(0) != 0]
                train_Y = [train_L[i] for i in [1,3] if train_Z[i].size(0) != 0]
                
            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)
            #print('generalized=', self.generalized,'\n','label unique',train_Y.unique())
            #print('test seen label unique',test_seen_Y.unique(), 'test unseen label unique',test_unseen_Y.unique())
            if args.semantics_for_classifier:
                #print('train X shape (before):', train_X.shape)
                #print(train_X.shape, self.dataset.attribute_extra[train_Y].shape)
                if self.generalized:
                    train_X = torch.cat((train_X, self.dataset.attribute_extra[train_Y].to(self.device)),1)
                else:
                    train_X = torch.cat((train_X, self.dataset.unseen_att[train_Y].to(self.device)),1)
                #print('train X shape (after):', train_X.shape)

        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################
        
        # 这里是调用 classifier.py 的内容

        cls = classifier.CLASSIFIER(clf, train_X, train_Y, test_seen_X, test_seen_Y, test_unseen_X,
                                    test_unseen_Y,
                                    cls_seenclasses, cls_unseenclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)
        best_acc = torch.tensor([0]).to(self.device)
        best_H = torch.tensor([0]).to(self.device)
        best_seen = torch.tensor([0]).to(self.device)
        best_unseen = torch.tensor([0]).to(self.device)
        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_unseen, cls.H = cls.fit()
                else:
                    cls.acc = cls.fit_zsl()

            if self.generalized:
                #print('[%.1f]     unseen=%.4f, seen=%.4f, h=%.4f , loss=%.4f' % (k, cls.acc_unseen, cls.acc_seen, cls.H, cls.average_loss))
                if cls.H >= best_H:
                    best_seen = cls.acc_seen
                    best_unseen = cls.acc_unseen
                    best_H = cls.H
            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                if cls.acc >= best_acc:
                    best_acc = cls.acc
        if self.generalized:
            print('=========================unseen=%.4f, seen=%.4f, h=%.4f ' % (best_unseen, best_seen, best_H))
            return best_unseen.cpu().item(), best_seen.cpu().item(), best_H.cpu().item()
        else:
            print('=========================acc=%.4f ' % (best_acc))
            return best_acc.cpu().item()
