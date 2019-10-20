# -*- coding: utf-8 -*-
import os, random
import os
from gen_log import gen_history
import sys
from keras.optimizers import Adam
import tensorflow as tf
# from keras import backend as K
import argparse
import math
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"
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
import network2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from .common import floatx, epsilon
# %%

def create_file(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    else:
        pass


# from keras import backend as K
# K.categorical_crossentropy(y_true, y_pred)
def snr_loss(y_true, y_pred):
    y_snr = (y_true[:, 11:])
    print(y_snr.get_shape())
    # y_snr = np.transpose(Y_train[:, 11:])
    target = y_true[:, :11]
    output = y_pred
    output /= tf.reduce_sum(output,
                            axis=len(output.get_shape()) - 1,
                            keep_dims=True)
    _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    print('改的是这里11', output.get_shape(), target.get_shape())
    return - tf.reduce_sum(y_snr * target * tf.log(output),
                           axis=len(output.get_shape()) - 1)
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def main(args):
    dim = args.data_dim
    time_step = int(256 / dim)


    name = args.model_name
    print(dim, time_step, name)
    # with open(r"C:\Users\niejinbo\Desktop\learning\AMR\AMR\data\RML2016.10a_dict.dat\CNN+RNN/RML2016.10a_dict.dat", 'rb') as xd1:  # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
    with open(r"F:\AMC\data/RML2016.10a_dict.dat", 'rb') as xd1:  # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
        Xd = pickle.load(xd1, encoding='latin1')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    X_low_snr=[]
    lbl = []
    lbl_low_snr=[]
    for mod in mods:
        for snr in snrs:
            if snr >=0:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
            else:
                X_low_snr.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):  lbl_low_snr.append((mod, snr))
    X = np.vstack(X)
    X_low_snr=np.vstack(X_low_snr)
    # %%
    if dim == 22:
        X = X.transpose((0, 2, 1))
    np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
    n_examples = X.shape[0]
    n_train = n_examples * 0.8  # 对半
    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
    X_train = X[train_idx]
    X_test = X[test_idx]
    classes = mods

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_train = to_onehot(trainy)
    low_idx=range(0,110000)
    train_low_snry=list(map(lambda x: mods.index(lbl_low_snr[x][0]), low_idx))
    Y_low_snr=to_onehot(train_low_snry)

     # print(Y_train)
    def f(x):
        return min(1,math.e**(x/10.))

    Y_train_snr = list(map(lambda x: (lbl[x][1]), train_idx))

    Y_train_snr = list(map(lambda x: f(x), Y_train_snr))

    Y_train = np.hstack((Y_train, np.reshape(Y_train_snr, (int(110000*0.8), 1))))
    # y_snr = np.transpose(Y_train[:, 11:])
    # print(np.shape(y_snr),y_snr)

    # print(Y_train)
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    Y_test_snr = list(map(lambda x: (lbl[x][1]), test_idx))
    Y_test_snr = list(map(lambda x: f(x), Y_test_snr))
    # print(np.shape(Y_test_snr),np.shape(Y_test))
    Y_test = np.hstack((Y_test, np.reshape(Y_test_snr, (int(110000*0.2), 1))))
    # print((Y_train[:,:-1]))

    # %%
    in_shp = list(X_train.shape[1:])
    print(X_train.shape, Y_test.shape,in_shp)

    classes = mods
    # Set up some params
    epochs = 1  # number of epochs to train on
    batch_size = 500  # training batch size default1024
    # %%
    from keras.callbacks import LearningRateScheduler
    def scheduler(epoch):
        if epoch % 5 == 0 and epoch != 0 and epoch<20:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * .9)
            print("lr changed to {}".format(lr * .9))
        if epoch % 10 == 0 and epoch != 0 and epoch>20:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * .9)
            print("lr changed to {}".format(lr * .9))
        return K.get_value(model.optimizer.lr)

    lr_decay = LearningRateScheduler(scheduler)

    kind = 11
    outdir = 'output/' + name + str(time_step) + '_' + str(dim) + '/picture/'
    tensorboardir = 'output/' + name + str(time_step) + '_' + str(dim) + '/' + 'logs'
    create_file(tensorboardir)
    create_file(outdir)
    # nerwork choose
    net = network2.Both_Net(time_step=time_step, dim=dim, kind=kind, model=name, classes=classes)
    model = net.output

    def snr_loss(y_true, y_pred):
        y_snr = (y_true[:, 11:])
        print(y_snr.get_shape())
        # y_snr = np.transpose(Y_train[:, 11:])
        target = y_true[:, :11]
        output = y_pred
        output /= tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keep_dims=True)
        _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        print('改的是这里11', output.get_shape(), target.get_shape())
        return - tf.reduce_sum(y_snr * target * tf.log(output),
                               axis=len(output.get_shape()) - 1)
    #'categorical_crossentropy'
    model.compile(loss=snr_loss, optimizer=Adam(lr=0.001), metrics=['accuracy'])
    filepath = 'output/' + name + str(time_step) + '_' + str(dim) + '/' + name + '_input_shape' + str(
        time_step) + '_' + str(dim) + ".h5"
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(X_test, Y_test),
                        callbacks=[  # 回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
                            keras.callbacks.ModelCheckpoint(filepath, verbose=2, save_best_only=True,
                                                            ),
                            keras.callbacks.EarlyStopping(patience=20, verbose=2, ),
                            lr_decay,
                            #keras.callbacks.TensorBoard(log_dir=tensorboardir, batch_size=batch_size,histogram_freq=0, write_graph=False)
                        ])  # EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
    model.load_weights(filepath)

    # %%
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
    print(score)
    gen_history(
        name='output/' + name + str(time_step) + '_' + str(dim) + '/' + str(model.name) + '_' + str(dim) + '.txt',
        dict=history.history, X_train=X_train, X_test=X_test,
        score=score, acc=acc, dim=dim)

    # %%
    # def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[], dir=''):
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(labels))
    #     plt.xticks(tick_marks, labels, rotation=45)
    #     plt.yticks(tick_marks, labels)
    #     plt.tight_layout()
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')
    #     picture_path = dir + title + ".png"
    #     print(picture_path)
    #     plt.savefig(picture_path)

    # %%
    # Plot confusion matrix 画图
    acc = {}
    fobj = open(outdir+name+'snr_acc.txt', 'w')
    for snr in snrs:
        if snr<-10:
            # extract classes @ SNR
            test_SNRs = list(map(lambda x: lbl_low_snr[x][1], low_idx))
            #    print(test_SNRs)
            test_X_i = X_low_snr[np.where(np.array(test_SNRs) == snr)]
            test_Y_i = Y_low_snr[np.where(np.array(test_SNRs) == snr)]

            # estimate classes
            test_Y_i_hat = model.predict(test_X_i)
            conf = np.zeros([len(classes), len(classes)])
            confnorm = np.zeros([len(classes), len(classes)])
            for i in range(0, test_X_i.shape[0]):
                j = list(test_Y_i[i, :]).index(1)
                k = int(np.argmax(test_Y_i_hat[i, :]))
                conf[j, k] = conf[j, k] + 1
            for i in range(0, len(classes)):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plt.figure()
            # plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr), dir=outdir)
            plt.close()
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            print("SNR is ",str(snr),"Overall Accuracy: ", cor / (cor + ncor))
            fobj.write('SNR is: ' + str(snr) + ' ' + str(cor / (cor + ncor)) + '\n')
            acc[snr] = 1.0 * cor / (cor + ncor)
        else:
            # extract classes @ SNR
            test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
            #    print(test_SNRs)
            test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
            test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

            # estimate classes
            test_Y_i_hat = model.predict(test_X_i)
            conf = np.zeros([len(classes), len(classes)])
            confnorm = np.zeros([len(classes), len(classes)])
            for i in range(0, test_X_i.shape[0]):
                j = list(test_Y_i[i, :]).index(1)
                k = int(np.argmax(test_Y_i_hat[i, :]))
                conf[j, k] = conf[j, k] + 1
            for i in range(0, len(classes)):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plt.figure()
            # plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr), dir=outdir)
            plt.close()
            cor = np.sum(np.diag(conf))
            ncor = np.sum(conf) - cor
            print("SNR Overall Accuracy: ", cor / (cor + ncor))
            fobj.write('SNR is: ' + str(snr)+' '+ str( cor / (cor + ncor)) + '\n')
            acc[snr] = 1.0 * cor / (cor + ncor)
    fobj.close()
    # %%
    # Plot accuracy curve
    plt.figure()
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
    plt.savefig(outdir + name + "_acc.png")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='model_name', type=str, help='Choose the model you use:', default='lstmmodel')
    parser.add_argument(dest='data_dim', type=int, help='Choose the model you use:', default=2)

    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.argv = ['k', 'resnet_50_FPN', '2']
    main(parse_arguments(sys.argv[1:]))