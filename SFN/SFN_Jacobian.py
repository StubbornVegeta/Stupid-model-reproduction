#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
##########################################################################
# File Name: SFN_Jacobian.py
# Author: stubborn vegeta
# Created Time: 2020年07月05日 星期日 12时43分47秒
##########################################################################
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os

def ReadData():
    PATH = '/home/vegeta/bcidatasetIV2a/mybci/'
    T01 = np.load(PATH+'mya01t.npz')['arr_0']
    Tlabel01 = np.load(PATH+'mya01t.npz')['arr_1']

    T02 = np.load(PATH+'mya02t.npz')['arr_0']
    Tlabel02 = np.load(PATH+'mya02t.npz')['arr_1']

    T03 = np.load(PATH+'mya03t.npz')['arr_0']
    Tlabel03 = np.load(PATH+'mya03t.npz')['arr_1']

    T04 = np.load(PATH+'mya04t.npz')['arr_0']
    Tlabel04 = np.load(PATH+'mya04t.npz')['arr_1']

    T05 = np.load(PATH+'mya05t.npz')['arr_0']
    Tlabel05 = np.load(PATH+'mya05t.npz')['arr_1']

    T06 = np.load(PATH+'mya06t.npz')['arr_0']
    Tlabel06 = np.load(PATH+'mya06t.npz')['arr_1']

    T07 = np.load(PATH+'mya07t.npz')['arr_0']
    Tlabel07 = np.load(PATH+'mya07t.npz')['arr_1']

    T08 = np.load(PATH+'mya08t.npz')['arr_0']
    Tlabel08 = np.load(PATH+'mya08t.npz')['arr_1']

    T09 = np.load(PATH+'mya09t.npz')['arr_0']
    Tlabel09 = np.load(PATH+'mya09t.npz')['arr_1']
    return T01, Tlabel01, T02, Tlabel02, T03, Tlabel03, T04, Tlabel04, T05, Tlabel05, T06, Tlabel06, T07, Tlabel07, T08, Tlabel08, T09, Tlabel09

trainData1,trainLabels1,trainData2,trainLabels2,trainData3,trainLabels3,trainData4,trainLabels4,trainData5,trainLabels5,trainData6,trainLabels6,trainData7,trainLabels7,trainData8,trainLabels8,trainData9,trainLabels9, = ReadData()
X = [trainData1, trainData2, trainData3, trainData4, trainData5, trainData6, trainData7, trainData8, trainData9 ]
Y = [trainLabels1,trainLabels2,trainLabels3,trainLabels4,trainLabels5,trainLabels6,trainLabels7,trainLabels8,trainLabels9,]
if not os.path.exists("fig_Jacobian"):
    os.makedirs("fig_Jacobian")
for i in range(1):
    X_data = X[i]
    Y_data = Y[i]
    _,trial,_ = X_data.shape
    X_data = X_data.transpose((1,0,2))

    N,M = Y_data.shape
    for n in range(N):
        for m in range(M):
            if Y_data[n,m] == 'right':
                Y_data[n,m] = 0
            elif Y_data[n,m] == 'left':
                Y_data[n,m] = 1
            elif Y_data[n,m] == 'foot':
                Y_data[n,m] = 2
            elif Y_data[n,m] == 'tongue':
                Y_data[n,m] = 3

    Y_data = Y_data[0].astype(np.int32)
    # Y_label = tf.convert_to_tensor(Y_data, dtype=tf.int32)
    # Y_label = tf.one_hot(Y_label, depth=4)

    # 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
    # seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
    np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(X_data)
    np.random.seed(116)
    np.random.shuffle(Y_data)
    tf.random.set_seed(116)

    # 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
    X_train = X_data[:-50]
    Y_train = Y_data[:-50]
    X_test = X_data[-50:]
    Y_test = Y_data[-50:]

    # 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)

    # from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
    train_db = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(16)
    test_db = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(16)


    Chans, Samples = 25, 750
    M = 12           # 选择M个通道
    O = 4            # 类别数量为O
    w1 = tf.Variable(tf.random.truncated_normal([Chans, M], stddev=0.1, seed=1))
    w2 = tf.Variable(tf.random.truncated_normal([M, O], stddev=0.1, seed=1))
    b2 = tf.Variable(tf.random.truncated_normal([O], stddev=0.1, seed=1))

    lr = 0.01  # 学习率为0.1
    train_loss_results = [1]  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
    test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
    epoch = 50     # 循环500轮
    loss_all = 0   # 每轮分4个step，loss_all记录四个step生成的4个loss的和

    mu = 100
    beta = 2
    q = tf.concat([tf.reshape(w1,[-1,1]), tf.reshape(w2,[-1,1]), tf.reshape(b2,[-1,1])], axis=0)
    for epoch in range(epoch):
        for step, (x_train, y_train) in enumerate(train_db):
            with tf.GradientTape(True) as tape:
                y = tf.matmul(tf.transpose(x_train, [0,2,1]), w1)
                y = tf.divide(y, tf.norm(w1,axis=0))
                _,variance = tf.nn.moments(y,axes=1)
                f = tf.math.log(variance)
                z = tf.add(tf.matmul(f, w2), b2)
                o = tf.nn.tanh(z)
                y_ = tf.one_hot(y_train, depth=4)
                loss = tf.reduce_mean(tf.square(y_ - o))
                e = tf.reshape((y_ - o),[-1,1])
                if loss.numpy() < train_loss_results[-1]:
                    mu = mu/beta
                else:
                    mu = mu*beta
            N_l = len(y_train)*4
            J_w1 = tf.reshape(tape.jacobian(e,w1),[N_l,-1])
            J_w2 = tf.reshape(tape.jacobian(e,w2),[N_l,-1])
            J_b2 = tf.reshape(tape.jacobian(e,b2),[N_l,-1])
            J = tf.concat([J_w1,J_w2,J_b2],axis=1)
            _,M_J = J.shape
            I = tf.eye(M_J)
            q = q - tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(J,[1,0]),J) + I*mu),tf.transpose(J,[1,0])),e)
            w1 = tf.Variable(tf.reshape(q[0:Chans*M],[Chans,M]))
            w2 = tf.Variable(tf.reshape(q[Chans*M:Chans*M+M*O],[M,O]))
            b2 = tf.Variable(tf.reshape(q[Chans*M+M*O:],[1,-1]))
            # w1 = tf.reshape(q[0:300],[25,12])
            # w2 = tf.reshape(q[300:348],[12,4])
            # b2 = tf.reshape(q[348:],[1,-1])
        # 每个epoch，打印loss信息
        print("Epoch {}, loss: {}".format(epoch, loss_all/4))
        train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
        loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

        # 测试部分
        # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
        total_correct, total_number = 0, 0
        for x_test, y_test in test_db: 
            # 使用更新后的参数进行预测
            y = tf.matmul(tf.transpose(x_test, [0,2,1]), w1)
            y = tf.divide(y, tf.norm(w1,axis=0))
            _,variance = tf.nn.moments(y,axes=1)
            f = tf.math.log(variance)
            z = tf.add(tf.matmul(f, w2),b2)
            o = tf.nn.tanh(z)
            pred = tf.argmax(o, axis=1)  # 返回y中最大值的索引，即预测的分类
            # 将pred转换为y_test的数据类型
            pred = tf.cast(pred, dtype=y_test.dtype)
            # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
            correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
            # 将每个batch的correct数加起来
            correct = tf.reduce_sum(correct)
            # 将所有batch中的correct数加起来
            total_correct += int(correct)
            # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
            total_number += x_test.shape[0]
        # 总的准确率等于total_correct/total_number
        acc = total_correct / total_number
        test_acc.append(acc)
        print("Test_acc:", acc)
        print("--------------------------")

    fig,ax1 = plt.subplots()
    ld1 = ax1.plot(train_loss_results[1,:], label="$Loss$", color='red')
    ax2 = ax1.twinx()
    ld2 = ax2.plot(test_acc, label="$Accuracy$")

    ld = ld1 + ld2
    labs = [l.get_label() for l in ld]
    plt.legend(ld, labs)
    ax1.set_xlabel('epoch')    #设置x轴标题
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    plt.savefig('fig_Jacobian/Ja-'+ str(i+1) +".png")

    with open('./README.md', 'a') as f:
        f.write(r"""
# 第"""+ str(i+1)+ """个人的分类正确率为："""+ str(acc) +"""\n
!["""+str(i+1)+"""](./fig_Jacobian/Ja-"""+str(i+1)+""".png)\n""")

