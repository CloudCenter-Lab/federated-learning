#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #主要步骤
    # 1 数据分发
    # 2 模型创建

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users 数据分发
        if args.iid:
            # 客户端之间的数据可能是独立同分布（IID），也可能是非独立同分布的（Non-IID）
            dict_users = mnist_iid(dataset_train, args.num_users)
            # 在这里面通过定义不同的数据划分方式，将数据划分为iid和non-iid两种，用来模拟测试fedavg 在不同场景下的性能
            # 返回的dict_users是一个字典类型，其中key是用户的id，value是用户拥有的图片id
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
            # pdb 是python自带的一个包，为python程序提供了一种交互的源代码调试功能
            # 主要特性包括设置断点，单步调试，进入函数调试，查看当前代码，查看栈片段，动态改变变量的值等
            pdb.set_trace()

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model 构建模型
    # 在这里我们需要选择models文件夹中的Nets.py文件所定义的神经网络模型
    # net_glob变量用来接收返回的网络类型，通过print(net_glob)就可以查看具体的网络结构
    # net_glob 就是初始化全局模型
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    # 注意这不是在进行模型的训练，而是要切换到训练模式
    # 在这里我们有两种模式，一个是训练模式，一个是测试模式
    # 区别就在于，测试的时候是不需要记录每一批数据的均值和方差的，也不需要进行误差等的计算；而训练模式需要信息的收集，以此来进行误差计算，更新模型
    net_glob.train()

    # copy weights 然后将net_glob的参数 复制给 w_glob
    # net_glob中的state_dict变量存放驯良过程中需要学习的权重和偏置系数
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    # 进行模型的训练，在这里我们定义的epoch是10
    for iter in range(args.epochs):
        # 在每一个epoch都要进行如下的操作
        # 首先定义一个client的损失函数
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        # m代表者随机抽取本地参与模型训练的client个数
        # 1 是用户量大的情况下，如果全都进行训练，会造成巨大的通信开销
        # 2 如果所有用户同时训练的情况下，并不能够保证所有用户在同一时刻都是在线的状态
        m = max(int(args.frac * args.num_users), 1) # args.frac * args.num_users = 0.1 * 100 = 10
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # 训练完之后返回 w 和loss
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                # w_locals的作用就是把每一个用户的w 都复制过来
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            # 复制参与本轮更新的clients的权重和 loss
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # 这里就是我们所说的FedAVG 联邦平均算法
        # 这里模拟的就是，本地模型训练好之后，将参数传递给server，交由sever进行聚合
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # 将每一个epoch的损失记录下来
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

