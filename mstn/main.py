import numpy as np
import torch
import time
import argparse
import os
import random
from sklearn import metrics
import collections
import function as fun
import model_mstn as model
torch.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Hyperspectral Image Classification相关参数')
parser.add_argument('--net_name', type=str, default="NET")
parser.add_argument('--optimizer', default='adam', help="Name of optimizer.")
parser.add_argument('--dataset', default='Indian_pines', help="Name of dataset.")#'Indian_pines': 10249, 'PaviaU': 42776, 'Salinas': 54129
parser.add_argument('--epoch', type=int,default=50, help=" Value of epoch")
parser.add_argument('--batch', type=int,default=64, help=" Value of batch")
parser.add_argument('--lr', type=float,default=1e-4, help=" learning rate")

parser.add_argument('--split',type=float, default=0, help="Percentage of split.")
parser.add_argument('--dev_split',type=float, default=0.02, help="Percentage of dev_split.")
parser.add_argument('--min_train',type=int, default=3, help="min number train dataset classes训练集每类数量")
parser.add_argument('--min_dev',type=int, default=5, help="min number dev dataset classes验证集每类数量")
parser.add_argument('--weight', type=float, default=2)

args = parser.parse_args()

if args.dataset=='Indian_pines':
    args.total=10249
    args.band = 30
    args.patch = 25
    args.classes = 16

elif args.dataset=='PaviaU':
    args.total = 42776
    args.band = 15
    args.patch = 19
    args.classes = 9
elif args.dataset == 'Salinas':
    args.total = 54129
    args.band = 15
    args.patch = 25
    args.classes = 16

def arg():
    return args
if __name__ == '__main__':
    seed=13
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_path = './dataset/'
    PATH_HEAD="./"+args.dataset+"/"
    fun.mkdir(PATH_HEAD+"net/")
    fun.mkdir(PATH_HEAD+"data/")

    PATH_net = PATH_HEAD+"net/net_"
    PATH_data=PATH_HEAD+"data/data_"


    data_hsi, gt_hsi = fun.load_dataset(data_path, args)
    data_hsi = fun.pp_pca(data_hsi, args)
    data_patch, data_all, gt = fun.pp_data(data_hsi, gt_hsi, args)
    train_indices, test_indices, dev_indices= fun.pp_sample(gt,args)
    w=np.array(test_indices)
    for m in test_indices:
        if m in dev_indices:
            test_indices.remove(m)
    print("\ntrain dataset number is ",len(train_indices))

    net = model.NET(args)

########################################################################################################
    N = 'TN1.pt'
    print("epoch=",args.epoch)
    Dl_indices = train_indices.copy()
    Du_indices = test_indices.copy()
    fun.train(1,net, Dl_indices, dev_indices, data_all, data_patch, gt, device, PATH_net + N, PATH_data, args)
##########################################################################################################
    args.e_alpha=0.006
    args.e_step=0.0005

    aa=2*args.classes
    Dl_indices=train_indices.copy()
    Du_indices=test_indices.copy()
    Dl_indices, Du_indices=fun.Dl_Du_change(Dl_indices, Du_indices, data_all, data_patch, gt, device, PATH_net+N,aa, args=args)
    print("\ntrain dataset number is ", len(Dl_indices))
    N='TN2.pt'
    fun.train(1,net, Dl_indices, dev_indices, data_all, data_patch, gt, device, PATH_net + N, PATH_data, args)
#######################################################################################################
    e_percent = [0.5,0.6,0.7,0.8]
    e0_percent = [0.01,0.01,0.02,0.02]
    N = 'TN2.pt'
    gt_p = np.zeros(len(gt), dtype=int)
    dl = np.array(Dl_indices)
    gt_p[dl] = gt[dl]
    dev = np.array(dev_indices)
    gt_p[dev] = gt[dev]
    test_indices = Du_indices.copy()
    for i in range(4):
        Dl_indices, Du_indices, gt_p = fun.Dl_Du_change_all_next(e_percent[i],e0_percent[i],Dl_indices, Du_indices, data_all, data_patch, gt_p, device,
                                                                 PATH_net + N, args)
        print("\ntrain dataset number is ", len(Dl_indices))
        N = 'TN3.pt'
        fun.train(0,net, Dl_indices, dev_indices, data_all, data_patch, gt_p, device, PATH_net + N, PATH_data, args)

        gt_test = gt[test_indices] - 1
        net = net.to(device)
        print("------train is over --------")
        PATHNET = PATH_net +N
        net.load_state_dict(torch.load(PATHNET))

        test_iter=fun.generate_td_iter(test_indices, data_all, data_patch, gt, args)
        pred_test = []


        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(device)
                net.eval()
                y_hat,_ = net(X.float(),0)
                y = torch.squeeze(y_hat.cpu())
                if len(y.size()) == 1:
                    y = y.reshape(1, -1)
                pred_test.extend(np.array(y.argmax(axis=1)))


        collections.Counter(pred_test)
        overall_acc = metrics.accuracy_score(pred_test, gt_test)
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
        each_acc, average_acc = fun.aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, gt_test)

        print("-----------train model and patch--epoch best-----------")
        print('overall_acc:',overall_acc)
        print('each_acc, average_acc:',each_acc, average_acc)
        print('kappa:',kappa)


        print("------train is over --------")






