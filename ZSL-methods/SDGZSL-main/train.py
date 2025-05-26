import torch.optim as optim
import glob
import argparse
import os
import random
import math
from time import gmtime, strftime
from models import *
from dataset_GBU_new_h5_no_folders_but_file import * #FeatDataLayer, DATA_LOADER
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.backends.cudnn as cudnn
import classifier
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SUN',help='dataset: CUB, AWA2, APY, FLO, SUN')
parser.add_argument('--dataroot', default='/cver/qzhe/GCAtt-opensource/data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)
parser.add_argument('--matdataset', action='store_false', default=True, help='Data in matlab format')

parser.add_argument('--gen_nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train generater')

parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')
parser.add_argument('--ga', type=float, default=15, help='relationNet weight')
parser.add_argument('--beta', type=float, default=1, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--dis', type=float, default=3, help='Discriminator weight')
parser.add_argument('--dis_step', type=float, default=2, help='Discriminator update interval')
parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')
parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')

parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_steps', type=int, default=50, help='training steps of the classifier')

parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=400)
parser.add_argument('--evl_start',  type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=5606, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--S_dim', type=int, default=1024)
parser.add_argument('--NS_dim', type=int, default=1024)

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')


# mode-aug classification 也需要 sample-level 数据的读取
# mode-level, sample-level , mode-/cls- rectified sample-level都在这里
parser.add_argument('--sample_level', action='store_true', default=False, help='load sample-level data?') 

# 关于 sample-level, mode-level, cls-/mode- rectified sample-level 训练数据的超参数
parser.add_argument('--sample_level_train', action='store_true', default=False, help='use sample level semantics in ZSL model training?')

# vision+semantics 分类
parser.add_argument('--semantics_for_classifier', action='store_true', default=False, help='use semantics in addition to vision for classification?') 
parser.add_argument('--cls_sample_att1', action='store_true', default=False, help='default: only seen -> GZSL| True: seen+unseen -> GZSL+TZSL')

parser.add_argument('--save_for_visual', action='store_true', default=False, help='acc 增幅') 
opt = parser.parse_args()
# 读取的 sample-level  semantics 一定和 class-level semantics 对应，命名也一致
opt.sample_level_emb = 'sample_level_'+opt.dataset+'_'+opt.class_embedding
print(opt)



if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")


def train():
    # 读取数据
    _acc = []
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/wd-{}_b-{}_g-{}_lr-{}_sd-{}_dis-{}_nS-{}_nZ-{}_bs-{}'.format(opt.dataset, opt.weight_decay,
                    opt.beta, opt.ga, opt.lr,
                            opt.S_dim, opt.dis, opt.nSample, opt.Z_dim, opt.batch_size)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    # 这里的 FeatDataLayer 输入读取到的 label 和 feature，后续处理
    #data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)
    
    

    # load data    
    if opt.sample_level_train:
        data_layer = SampleFeatDataLayer(dataset.train_feature.numpy(), dataset.train_label.numpy(), dataset.train_sample_attribute.numpy(), opt)#
        # print('sample-level load]]]]]]]]]]]]]]]]]]]]]]]]]]]]]')
    else:
        data_layer = ClsFeatDataLayer(dataset.train_feature.numpy(), dataset.train_label.numpy(), dataset.attribute.numpy(), opt)#


    iter_each = int(dataset.ntrain/opt.batch_size) 
    opt.niter = iter_each * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()

    model = VAE(opt).to(opt.gpu)
    relationNet = RelationNet(opt).to(opt.gpu)
    discriminator = Discriminator(opt).to(opt.gpu)
    ae = AE(opt).to(opt.gpu)
    print(model)

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(relationNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ae_optimizer = optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ones = torch.ones(opt.batch_size, dtype=torch.long, device=opt.gpu)
    zeros = torch.zeros(opt.batch_size, dtype=torch.long, device=opt.gpu)
    mse = nn.MSELoss().to(opt.gpu)


    iters = math.ceil(dataset.ntrain/opt.batch_size)
    beta = 0.01
    coin = 0
    gamma = 0
    for it in range(start_step, opt.niter+1):

        if it % iters == 0:
            beta = min(opt.kl_warmup*(it/iters), 1)
            gamma = min(opt.tc_warmup * (it / iters), 1)

        blobs = data_layer.forward()
        feat_data = blobs['data']
        labels_numpy = blobs['labels'].astype(int)
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)
        # 样本本身取 sample-level; 在整体的 encoder-decoder 架构中，每个样本级的输入。
        if opt.sample_level_train:
            C = blobs['att']
        else:
            C = np.array([dataset.train_att[i,:] for i in labels])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        X = torch.from_numpy(feat_data).to(opt.gpu)
        # 去分类的那些就直接用 class-level 吧
        # 想让的是
        sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).to(opt.gpu)
        sample_C_n = labels.unique().shape[0]
        sample_label = labels.unique().cpu()
        # 实际上 sample-level 只用于了 VAE 训练，没有用到 RelationNet 内部。该网络内部使用的还是 class-level 的兼容性得分
        x_mean, z_mu, z_var, z = model(X, C)
        loss, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batch_size, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).to(opt.gpu)

        # one_hot_labels = torch.tensor(
        #     torch.zeros(opt.batch_size, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1)).to(opt.gpu)

        x1, h1, hs1, hn1 = ae(x_mean)
        relations = relationNet(hs1, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])

        p_loss = opt.ga * mse(relations, one_hot_labels)

        x2, h2, hs2, hn2 = ae(X)
        relations = relationNet(hs2, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])

        p_loss = p_loss + opt.ga * mse(relations, one_hot_labels)

        rec = mse(x1, X) + mse(x2, X)
        if coin > 0:
            s_score = discriminator(h1)
            tc_loss = opt.beta * gamma *((s_score[:, :1] - s_score[:, 1:]).mean())
            s_score = discriminator(h2)
            tc_loss = tc_loss + opt.beta * gamma* ((s_score[:, :1] - s_score[:, 1:]).mean())

            loss = loss + p_loss + rec + tc_loss
            coin -= 1
        else:
            s, n = permute_dims(hs1, hn1)
            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h1)
            n_score = discriminator(b)
            tc_loss = opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            s, n = permute_dims(hs2, hn2)
            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h2)
            n_score = discriminator(b)
            tc_loss = tc_loss + opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            dis_optimizer.zero_grad()
            tc_loss.backward(retain_graph=True)
            dis_optimizer.step()

            loss = loss + p_loss + rec
            coin += opt.dis_step
        optimizer.zero_grad()
        relation_optimizer.zero_grad()
        ae_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        relation_optimizer.step()
        ae_optimizer.step()


        if it % iter_each == 0 and it:
            model.eval()
            ae.eval()
            gen_feat, gen_label = synthesize_feature_test(model, ae, dataset, opt)
            with torch.no_grad():
                train_feature = ae.encoder(dataset.train_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_unseen_feature = ae.encoder(dataset.test_unseen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_seen_feature = ae.encoder(dataset.test_seen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()

            train_X = torch.cat((train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)
            #print('============', dataset.train_label.unique(), gen_label.unique())

            if opt.semantics_for_classifier:
                train_XS = torch.cat((dataset.seen_att[dataset.train_label], dataset.unseen_att[gen_label]),0) # 有问题
                train_X = torch.cat((train_X, train_XS),1).to(torch.float32)
                gen_feat = torch.cat((gen_feat, dataset.unseen_att[gen_label]),1).to(torch.float32) #有问题
                test_unseen_feature = torch.cat((test_unseen_feature, dataset.test_unseen_sample_attribute),1).to(torch.float32)
                test_seen_feature = torch.cat((test_seen_feature, dataset.test_seen_sample_attribute),1).to(torch.float32)
            
            
            if opt.zsl:
                """ZSL"""
                cls = classifier.CLASSIFIER(opt, gen_feat, gen_label, dataset, test_seen_feature, test_unseen_feature,
                                            dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 20,
                                            opt.nSample, False)
                result_zsl_soft.update(it, cls.acc)
                log_print("Iter-[{}/{}]; Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                    it,opt.niter,cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)
                _acc.append(cls.acc)

            else:
                """ GZSL"""
                cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                    dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5,
                                            opt.classifier_steps, opt.nSample, True)

                result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("Iter-[{}/{}]; U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                    it,opt.niter,cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                    result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)
                _acc.append(cls.H)

            #np.save('sample-level-stability/{}_sample-level-{}.npy'.format(opt.dataset, opt.sample_level_train),np.array(_acc))
            
            model.train()
            ae.train()
if __name__ == "__main__":
    train()