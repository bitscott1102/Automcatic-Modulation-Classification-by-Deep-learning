# -*- coding: utf-8 -*-

#%% import
import os, random
import os
from gen_log import gen_history #自建函数：生成日志
import sys
from keras.optimizers import Adam #优化器的一种，另一种常用的是SGD
import tensorflow as tf
# from keras import backend as K
import argparse
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow" #设置框架为tf
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
# import seaborn as sns
import pickle, random, sys, keras,os
import h5py
import network2  #包含的模型的文件

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from .common import floatx, epsilon

# %% 创建一个路径用的
def create_file(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    else:
        pass
# from keras import backend as K
# K.categorical_crossentropy(y_true, y_pred)
        
#%% 公式8     后面有多余部分
def snr_loss(y_true, y_pred):
    y_snr = (y_true[:, 11:])
    print(y_snr.get_shape())
    # y_snr = np.transpose(Y_train[:, 11:])
    target = y_true[:, :11]
    output = y_pred
    output /= tf.reduce_sum(output,
                            axis=len(output.get_shape()) - 1,
                            keep_dims=True)
    _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)#最小值，用来防止出现nan
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    #print('改的是这里11', output.get_shape(), target.get_shape())
    return - tf.reduce_sum(y_snr * target * tf.log(output),
                           axis=len(output.get_shape()) - 1)

#%% 数据类型转换 
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

#%% main()
def main(args):
    dim = args.data_dim
    time_step = int(256 / dim)
    name = args.model_name
    
    ##%% 对信号按照信噪比进行分区，并且将I、Q两路信号放在X与X_low_snr中，把对应的snr and mod 放在lbl与lbl_low_snr中
    # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
    with open(r"F:\AMC\data\data/RML2016.10a_dict.dat", 'rb') as xd1:  # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
        Xd = pickle.load(xd1, encoding='latin1') #从文件中，读取字符串，将他们反序列化转换为python的数据对象，可以正常像操作数据类型的这些方法来操作它们
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = [] #data，放你仿真的数据
    X_low_snr=[]
    lbl = []
    lbl_low_snr=[]
    for mod in mods:
        for snr in snrs:
            if snr >=0:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
            else:
                X_low_snr.append(Xd[(mod, snr)]) #在X_low_snr中加入信噪比低于阈值的信号的I路和Q路
                for i in range(Xd[(mod, snr)].shape[0]):  lbl_low_snr.append((mod, snr))
    X = np.vstack(X) #对数据进行按照行堆叠
    X_low_snr=np.vstack(X_low_snr)
    
    ##%% 对预处理好的数据制作成投入网络训练的格式，并进行one-hot编码
    ##%% 这两行代码还不知道是什么作用
    if dim == 22:
        X = X.transpose((0, 2, 1))
    np.random.seed(2016)  #随机种子
    n_examples = X.shape[0] #大于snr阈值的数据总量
    n_train = n_examples * 0.8  # 对数据进行八二开
    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
    X_train = X[train_idx]#对数据进行随机“八二”划分，将x分为X_train and X_test
    X_test = X[test_idx]
    classes = mods

    ##%% to_onehot函数的作用是将代表调至种类的数字（0，1，2...10）转换为（10000000000，01000000000...)
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))#字符串变成数字，0-10代表十一种调制模式
    Y_train = to_onehot(trainy)#调制模式，见图
    
    ####没看懂以下三行是干什么用的###
    low_idx=range(0,110000)#改snr阈值的话，这里要修改，要修改成，低于阈值的点的个数220000*（1-XX%）
    train_low_snry=list(map(lambda x: mods.index(lbl_low_snr[x][0]), low_idx))###对调制模式进行划分：110000个点，每一万为一个区间，定义为0，1，2.。。。10
    Y_low_snr=to_onehot(train_low_snry)##对调制模式进行（0，1，2...10）one-hot成([10000000000],[01000000000]...[00000000001])
    ####没看懂以上三行是干什么用的###

     # print(Y_train)
    def f(x):#把信噪比归一化转变为权重
        return min(1,math.e**(x/10.))

    Y_train_snr = list(map(lambda x: (lbl[x][1]), train_idx))#导出信噪比给Y_train_snr

    Y_train_snr = list(map(lambda x: f(x), Y_train_snr))#对信噪比赋予权重，snr大于0均为1，小于0，用f(x)替代
    
    #如果要修改snr阈值的话，其中，0.8是训练集占整体数据集的大小，110000是高于SNR阈值的数量220000*XX%
    Y_train = np.hstack((Y_train, np.reshape(Y_train_snr, (int(110000*0.8), 1))))#在后面增加一列信噪比，一会看一下加入snr小于0的部分，后面是什么
    # y_snr = np.transpose(Y_train[:, 11:])
    # print(np.shape(y_snr),y_snr)

    # print(Y_train)
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    Y_test_snr = list(map(lambda x: (lbl[x][1]), test_idx))
    Y_test_snr = list(map(lambda x: f(x), Y_test_snr))
    # print(np.shape(Y_test_snr),np.shape(Y_test))
    Y_test = np.hstack((Y_test, np.reshape(Y_test_snr, (int(110000*0.2), 1))))#其中，0.2是测试集占整体数据集的大小，110000是高于SNR阈值的数量220000*XX
    # print((Y_train[:,:-1]))

    in_shp = list(X_train.shape[1:])#输入数据的矩阵大小
    print(X_train.shape, Y_test.shape,in_shp)
#到此为止，数据准备结束，验证集测试集已经划分，label已经准备好

#%%
    classes = mods#调制模式
    # Set up some params
    epochs = 10  # number of epochs to train on
    batch_size = 500  # training batch size default1024


    model.load_weights(filepath)

    # %%
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    print(score)
    gen_history(
        name='output/' + name + str(time_step) + '_' + str(dim) + '/' + str(model.name) + '_' + str(dim) + '.txt',
        dict=history.history, X_train=X_train, X_test=X_test,
        score=score, acc=acc, dim=dim)

# %% 画出识别信号的matrix图    
    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[], dir=''):  
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        picture_path = dir + title + ".png"
        print(picture_path)
        plt.savefig(picture_path)
