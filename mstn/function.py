import numpy as np
import torch
import scipy.io as sio
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
import torch.utils.data as Data
from torch import Tensor
from torch.autograd import Variable
from operator import truediv
import time
from sklearn import metrics
import model_mstn as model
import os
import torch.nn.functional as F
import random


def load_dataset(data_path ,args):
    if args.dataset == 'Indian_pines':
        data_mat = sio.loadmat(data_path + 'Indian_pines_corrected.mat')
        data_hsi = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat(data_path + 'Indian_pines_gt.mat')
        gt_hsi = gt_mat['indian_pines_gt']

    if args.dataset == 'PaviaU':
        data_mat = sio.loadmat(data_path + 'PaviaU.mat')
        data_hsi = data_mat['paviaU']
        gt_mat = sio.loadmat(data_path + 'PaviaU_gt.mat')
        gt_hsi = gt_mat['paviaU_gt']

    if args.dataset == 'Salinas':
        data_mat = sio.loadmat(data_path + 'Salinas_corrected.mat')
        data_hsi = data_mat['salinas_corrected']
        gt_mat = sio.loadmat(data_path + 'Salinas_gt.mat')
        gt_hsi = gt_mat['salinas_gt']


    return data_hsi,gt_hsi

def pp_pca(data_hsi,args):
    d_shape = data_hsi.shape
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
    data_hsi = PCA(n_components=args.band).fit_transform(data_hsi)
    d_shape = np.array(d_shape)
    d_shape[-1] = args.band
    data_hsi = data_hsi.reshape(d_shape)
    return data_hsi



def pp_data(data_hsi,gt_hsi,args):

    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    patch_s=math.floor(args.patch/2)
    data = preprocessing.scale(data)
    data_all = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
    data_patch = np.lib.pad(data_all,((patch_s, patch_s),(patch_s, patch_s),(0, 0)),'constant',constant_values=0)
    return data_patch,data_all,gt

def pp_sample(gt,args):
    train = {}
    test = {}
    dev={}
    labels_loc = {}

    for i in range(args.classes):
        indexes = [
            j for j, x in enumerate(gt.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        num_train= max(int(args.split * len(indexes)), args.min_train)
        train[i] = indexes[:num_train]
        test[i] = indexes[num_train:]
        if args.dev_split != 1:
            num_dev= max(int(args.dev_split * len(test[i])), args.min_dev)

        else:
            num_dev = 0
        dev[i]=test[i][-num_dev:]

    train_indexes = []
    test_indexes = []
    dev_indexes = []
    for i in range(args.classes):
        train_indexes += train[i]
        test_indexes += test[i]
        dev_indexes += dev[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    np.random.shuffle(dev_indexes)
    return train_indexes, test_indexes,dev_indexes

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

class clsDataset(Data.Dataset):
    def __init__(self,Data ,Label ):

        self.Data=Data
        self.Label=Label

    def __getitem__(self, index):

        index=index
        Data=torch.from_numpy(self.Data[index]).type(torch.FloatTensor).unsqueeze(0)
        return Data ,self.Label[index]
    def __len__(self):
        return len(self.Data)

class ctrDataset(Data.Dataset):
    def __init__(self,Date,Data1,Data2,Data3):
        self.Data=Date
        self.Data1=Data1
        self.Data2 = Data2
        self.Data3 = Data3


    def __getitem__(self, index):

        index=index
        num1=self.Data1[index]

        Data1=torch.from_numpy(self.Data[num1]).type(torch.FloatTensor).unsqueeze(0)

        num2=self.Data2[index]
        Data2 = torch.from_numpy(self.Data[num2]).type(torch.FloatTensor).unsqueeze(0)
        num3 = self.Data3[index]
        Data3 = torch.from_numpy(self.Data[num3]).type(torch.FloatTensor).unsqueeze(0)

        return Data1, Data2, Data3
    def __len__(self):

        return len(self.Data1)


def generate_train_pn_iter(flag,train_indices,data_all, data_patch, gt,args):

    y_train = gt[train_indices] - 1
    patch_s = math.floor(args.patch / 2)
    x_train = select_small_cubic(len(y_train), train_indices, data_all,patch_s, data_patch, args.band)

    classi_data = clsDataset(x_train, y_train)

    cls_train_iter = Data.DataLoader(
        dataset=classi_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )
    if flag==0:
        return cls_train_iter
    if flag==1:
        data1 = []
        data2 = []
        data3 = []


        for order in range(len(y_train)):
            for k1 in range(len(y_train)):
                for k2 in range(len(y_train)):
                    if (not order == k1) and (not order == k2) and (not k1 == k2):
                       if int(y_train[order] == y_train[k1]) and int(y_train[order] != y_train[k2]):
                            data1.append(order)
                            data2.append(k1)
                            data3.append(k2)

        ctr_data = ctrDataset(x_train, data1,data2,data3)

        ctr_train_iter = Data.DataLoader(
            dataset=ctr_data,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
        )


        return cls_train_iter,ctr_train_iter


def generate_td_iter(td_indices,data_all, data_patch, gt,args):

    y_td = gt[td_indices] - 1
    patch_s = math.floor(args.patch / 2)
    x_td = select_small_cubic(len(y_td), td_indices, data_all,patch_s, data_patch, args.band)

    classi_data = clsDataset(x_td, y_td)

    cls_td_iter = Data.DataLoader(
        dataset=classi_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )
    return cls_td_iter
def generate_cand_iter(td_indices,data_all, data_patch,args):


    patch_s = math.floor(args.patch / 2)
    x_td = select_small_cubic(len(td_indices), td_indices, data_all,patch_s, data_patch, args.band)

    classi_data = clsDataset(x_td, td_indices)

    cls_td_iter = Data.DataLoader(
        dataset=classi_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )
    return cls_td_iter


class ContrastiveLoss(torch.nn.Module):

    def __init__(self,weight=1, margin=1.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight = weight
    def check_type_forward(self, in_types):
        assert len(in_types) == 3

    def forward(self, x0, x1, x2):
        self.check_type_forward((x0, x1, x2))

        diff1 = x0 - x1
        dist_sq1 = torch.sum(torch.pow(diff1, 2), 1)

        diff2 = x0 - x2
        dist_sq2 = torch.sum(torch.pow(diff2, 2), 1)
        dist2 = torch.sqrt(dist_sq2)

        mdist2 = self.margin - dist2
        dist2 = torch.clamp(mdist2, min=0.0)
        dist_sq2 =torch.pow(dist2, 2)
        loss=dist_sq1- dist_sq2+self.weight
        loss=torch.clamp(loss, min=0.0)

        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

def pre_acc_loss(data_iter, net, loss, device):

    acc_sum, n_sum = 0.0, 0
    batch_l_sum, batch_num = 0, 0
    with torch.no_grad():
        net.eval()
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat,_= net(X,0)
            y_hat = y_hat.squeeze()
            if len(y_hat.size()) == 1:
                y_hat = y_hat.reshape(1, -1)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            batch_l_sum += l
            batch_num += 1
            n_sum += y.shape[0]
    net.train()
    return [acc_sum / n_sum, batch_l_sum/batch_num]


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=0)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def train(flag,net,train_indices,  dev_indices,data_all, data_patch, gt,device,PATH_net,PATH_data,args=None):

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    dev_loss_list = []
    train_acc_list = []
    dev_acc_list = []
    best_dev_acc = 0.0
    best_train_acc=0.0
    best_train_loss=0.0
    best_epoch=0

    optimizer0 = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.00005)
    loss0 = torch.nn.CrossEntropyLoss()
    loss1 = ContrastiveLoss(weight=args.weight)

    dev_iter=generate_td_iter(dev_indices, data_all, data_patch, gt, args)
    if flag:

        clstrain_iter, ctrtrain_iter = generate_train_pn_iter(1, train_indices, data_all, data_patch, gt, args)
        r1=np.ceil(clstrain_iter.sampler.num_samples/args.batch)
        r3 =np.ceil( ctrtrain_iter.sampler.num_samples/args.batch)
        r4=min(r3,r1*40)
        lr1 = args.lr * r1 / r4

        optimizer1 = torch.optim.Adam(net.parameters(), lr=lr1, weight_decay=0)
    else:
        clstrain_iter = generate_train_pn_iter(0, train_indices, data_all, data_patch, gt, args)

    for epoch in range(args.epoch):
        train_acc_sum, train_n_sum = 0.0, 0
        batch_l_sum, batch_count = 0.0, 0
        epoch_contrastloss = 0
        time_epoch = time.time()

        net.train()
        if flag:

            for step, (x_i, x_j1, x_j2) in enumerate(ctrtrain_iter):
                x_i = x_i.to(device).float()
                x_j1 = x_j1.to(device).float()
                x_j2 = x_j2.to(device).float()

                h_i = net(x_i,1)
                h_j1 = net(x_j1,1)
                h_j2 = net(x_j2, 1)
                l1 =loss1(h_i, h_j1, h_j2)
                optimizer1.zero_grad()
                l1.backward(retain_graph=True)
                optimizer1.step()
                if step>=r4:
                    break
        for X, y in clstrain_iter:

            X = X.to(device).float()
            y = y.to(device)
            y_hat,_ = net(X,0)
            y_hat=y_hat.squeeze()
            if len(y_hat.size()) == 1:
                y_hat = y_hat.reshape(1, -1)
            l = loss0(y_hat,y.long())
            optimizer0.zero_grad()

            l.backward(retain_graph=True)


            optimizer0.step()
            batch_l_sum += l.cpu().item()
            batch_count += 1

            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            train_n_sum += y.shape[0]

        dev_acc, dev_loss = pre_acc_loss(dev_iter, net, loss0, device)

        train_loss_list.append(batch_l_sum/ batch_count)
        train_acc_list.append(train_acc_sum / train_n_sum)
        dev_loss_list.append(dev_loss)
        dev_acc_list.append(dev_acc)

        print(
            'epoch %d, train loss %.6f, train acc %.3f, dev loss %.6f, dev acc %.3f, time %.1f sec,'
            % (epoch + 1, batch_l_sum / batch_count, train_acc_sum / train_n_sum,
               dev_loss, dev_acc, time.time() - time_epoch)
        )



        if dev_acc>0 and best_dev_acc < dev_acc:
            PATHNET=PATH_net
            torch.save(net.state_dict(), PATHNET)
            best_epoch=epoch
            best_dev_acc = dev_acc
            best_train_loss=batch_l_sum / batch_count
            best_train_acc=train_acc_sum / train_n_sum
            print("best_dev_acc ={}".format(best_dev_acc))


    print('best_epoch %d, train_loss %.4f, train_acc %.3f, time %.1f sec'
          % (best_epoch, best_train_loss, best_train_acc,
             time.time() - start))

def generate_anchor(net,Dl_indices, data_all, data_patch, gt,device,args=None):
    net = net.to(device)
    clsDl_iter= generate_td_iter(Dl_indices, data_all, data_patch, gt, args)
    net.eval()
    zl=torch.zeros(0)
    y_Dl=torch.zeros(0)
    anchor =torch.zeros(0)
    with torch.no_grad():
        for X, y in clsDl_iter:
            y_Dl=torch.cat((y_Dl,y))
            X = X.to(device)
            net.eval()
            _, z = net(X.float(), 0)


            zl=torch.cat((zl, z.cpu()))
    for i in range(args.classes):
        emb = zl[y_Dl == i]
        anchor_i = emb.mean(dim=0).view(1, -1)
        anchor=torch.cat((anchor, anchor_i))
    return anchor

def find_candidate_set(net,net_mix,e_alpha,Dl_indices, Du_indices, data_all, data_patch, gt,device,args=None):


    candidate_set = torch.zeros(0, dtype=int)
    c=torch.zeros(args.classes)
    with torch.no_grad():

        net = net.to(device)
        net_mix = net_mix.to(device)
        anchor = generate_anchor(net, Dl_indices, data_all, data_patch, gt, device, args)

        cand_iter=generate_cand_iter(Du_indices, data_all, data_patch,  args)
        net.eval()
        net_mix.eval()

        eps = e_alpha / math.sqrt(anchor.size(1))
        for step,(x, index) in enumerate(cand_iter):

            x = x.to(device)
            pred_Duu,zu=net(x,0)
            pred_Duu=F.softmax(pred_Duu.cpu(),dim=1)

            zu =zu.cpu()
            pred_Du= pred_Duu.argmax(dim=1)

            with torch.enable_grad():
                var_emb = Variable(zu, requires_grad=True).to(device)
                out = net_mix(var_emb)

                loss = F.cross_entropy(out, pred_Du.to(device).long())
                grads = torch.autograd.grad(loss, var_emb)[0].cpu()


            for cls in range(args.classes):

                anchor_i=anchor[cls].repeat(len(index),1)

                z = (anchor_i - zu)
                alpha = (eps * z.norm(dim=1) / grads.norm(dim=1)).unsqueeze(dim=1).repeat(1,z.size(1)) * grads / (z + 1e-8)

                zu_mix = (1 - alpha) * zu + alpha * anchor_i
                zu_mix=zu_mix.to(device)
                pred_zu_mix=net_mix(zu_mix)
                pred_zu_mix=pred_zu_mix.cpu()
                pc = pred_zu_mix.argmax(dim=1) != pred_Du
                indexes=index[pc== True]
                pred_Duex=pred_Du[pc== True]
                candidate_set = torch.cat((candidate_set, indexes))
    candidate_set= torch.unique(candidate_set)

    return candidate_set

def Dl_Du_change(Dl1_indices, Du_indices, data_all, data_patch, gt,device,PATHNET,aa=32,args=None):

    e_alpha=args.e_alpha

    net = model.NET(args)
    net_mix=model.NET_mix(args)
    net.load_state_dict(torch.load(PATHNET))
    net_mix.load_state_dict(torch.load(PATHNET))
    Dl_indices=Dl1_indices.copy()
    while (aa):
        candidate_indices = find_candidate_set(net, net_mix, e_alpha, Dl_indices, Du_indices, data_all, data_patch, gt,
                                               device, args)
        print("aa=",len(candidate_indices))
        if len(candidate_indices)<=aa:

            Dl_indices = torch.from_numpy(np.array(Dl_indices))
            Du_indices = torch.from_numpy(np.array(Du_indices))
            Dl_indices = torch.cat((Dl_indices, candidate_indices))
            Du_indices = Du_indices[~np.isin(Du_indices, candidate_indices)]
            aa=aa-len(candidate_indices)
        else:
            Dl_indices = torch.from_numpy(np.array(Dl_indices))
            Du_indices = torch.from_numpy(np.array(Du_indices))
            Dl_indices = torch.cat((Dl_indices, candidate_indices[:aa]))
            Du_indices = Du_indices[~np.isin(Du_indices, candidate_indices[:aa])]
            aa=0

        e_alpha+=args.e_step

    Dl_indices = Dl_indices.numpy().tolist()
    Du_indices = Du_indices.numpy().tolist()

    return Dl_indices, Du_indices
def para_tensor_bvsb(Du_indices, data_all, data_patch,device,PATHNET,gt,args=None):
    gt_p=np.copy(gt)
    pred_du = torch.zeros(0, dtype=int)
    predb_du = torch.zeros(0)
    predsb_du = torch.zeros(0)
    index_du = torch.zeros(0, dtype=int)
    with torch.no_grad():
        net = model.NET(args)
        net.load_state_dict(torch.load(PATHNET))
        net = net.to(device)

        cand_iter = generate_cand_iter(Du_indices, data_all, data_patch, args)
        net.eval()

        for step, (x, index) in enumerate(cand_iter):
            index_du = torch.cat((index_du, index))
            x = x.to(device)
            pred_Duu, zu = net(x, 0)
            pred_Duu = F.softmax(pred_Duu.cpu(), dim=1)

            zu = zu.cpu()
            pred_Du = pred_Duu.argmax(dim=1)
            gt_p[index]=pred_Du+1
            y_sorted, idx_sorted = torch.sort(pred_Duu, descending=True)
            yb = y_sorted[:, 0]
            ysb = y_sorted[:, 1]
            pred_du = torch.cat((pred_du, pred_Du))
            predb_du = torch.cat((predb_du, yb))
            predsb_du = torch.cat((predsb_du, ysb))

    return  pred_du, predb_du, predb_du- predsb_du,index_du,gt_p

def cand_bvsb_percent_next(e_percent,e0_percent,pred_du, predbsb_du,index_du,args=None):
    Du_indices_percent=torch.zeros(0,dtype=int)
    predbsb_du_percent=torch.zeros(0,dtype=int)

    for i in range(args.classes):
        predbsb_du_i=predbsb_du[pred_du==i]
        index_du_i=index_du[pred_du==i]
        y_sorted, idx_sorted = torch.sort(predbsb_du_i, descending=True)
        cc=int(len(predbsb_du_i)*e_percent)
        cc1=max(3,int(len(predbsb_du_i)*e0_percent))

        predbsb_du_percent = torch.cat((predbsb_du_percent, predbsb_du_i[idx_sorted[cc-cc1:cc]]))
        Du_indices_percent = torch.cat((Du_indices_percent, index_du_i[idx_sorted[cc-cc1:cc]]))

    return Du_indices_percent,predbsb_du_percent


def Dl_Du_change_all_next(e_percent,e0_percent,Dl_indices, Du_indices, data_all, data_patch, gt,device,PATHNET,args=None):


    pred_du, predb_du, predbsb_du, index_du,gt=para_tensor_bvsb(Du_indices, data_all, data_patch,device,PATHNET,gt,args)
    candidate_bvsb_du_percent ,predbsb_du_percent= cand_bvsb_percent_next(e_percent,e0_percent,pred_du, predbsb_du, index_du, args)


    Dl_indices = torch.from_numpy(np.array(Dl_indices))
    Du_indices = torch.from_numpy(np.array(Du_indices))
    Dl_indices = torch.cat((Dl_indices, candidate_bvsb_du_percent))
    Du_indices = Du_indices[~np.isin(Du_indices, candidate_bvsb_du_percent)]

    Dl_indices = Dl_indices.numpy().tolist()
    Du_indices = Du_indices.numpy().tolist()

    return Dl_indices, Du_indices,gt
def mkdir(path):

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
       os.makedirs(path)
       return True
    else:
       return False

