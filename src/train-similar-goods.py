#coding:utf-8
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_th_4 import load_data
import h5py
import argparse
from keras.models import model_from_json
#import numpy as np
#np.random.seed(1337)  # for reproducibility

#Number of samples per gradient update
batch_size = 50
#分类个数
nb_classes = 2
#the number of times to iterate over the training data arrays
nb_epoch = 15
#训练CNN的个数
nb_loop = 5

# input image dimensions
img_rows, img_cols = 64, 32
# the images are RGB
img_channels = 3

ap = argparse.ArgumentParser()
ap.add_argument("--train1", required=True, help="Path to the positive training set")
ap.add_argument("--train0", required=True, help="Path to the negative training set")
ap.add_argument("--test1", required=True, help="Path to the positive testing set")
ap.add_argument("--test0", required=True, help="Path to the negative testing set")
args = vars(ap.parse_args())

#加载数据
path_positive_train = args["train1"]
path_negative_train = args["train0"]
path_positive_valid_test = args["test1"]
path_negative_valid_test = args["test0"]
X_train,Y_train,X_valid,Y_valid,X_test,Y_test = load_data(path_positive_train,path_negative_train,path_positive_valid_test,path_negative_valid_test)
print(X_train.shape[0], 'training samples')
print(X_valid.shape[0],'validation samples')
print(X_test.shape[0],'testing samples')

#label为0,1两个类别，keras要求格式为binary class matrices,转化一下，调用keras提供的函数
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

#dist保存每个cnn的index和test accurary，存储形式为{"1":0.95,"2":0.90}
dist={}

for i in range(nb_loop):
    ###############
    #开始建立CNN模型
    ###############
    model = Sequential()
    #第一个卷积层，32个卷积核，每个卷积核大小5*5
    #border_mode为valid
    #激活函数用relu
    model.add(Convolution2D(32, 5, 5,
                    border_mode='valid',
                    activation='relu',
                    input_shape=(img_channels, img_rows, img_cols)))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #第二个卷积层，32个卷积核，每个卷积核大小5*5
    #采用maxpooling,poolsize为（2,2）
    #使用Dropout
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #第三个卷积层，64个卷积核，每个卷积核大小3*3
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #全连接层
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #Softmax回归，输出2个类别
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    ##############
    #开始训练模型
    ##############

    #lr: float >= 0. Learning rate.
    #momentum: float >= 0. Parameter updates momentum.
    #decay: float >= 0. Learning rate decay over each update.
    #nesterov: boolean. Whether to apply Nesterov momentum.
    #model.compile里的参数loss就是损失函数(目标函数),binary_crossentropy即为logistic loss
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


    #Stop training when a monitored quantity has stopped improving.
    #monitor: quantity to be monitored.
    #patience: number of epochs with no improvement after which training will be stopped.
    #verbose: verbosity mode.
    #mode: one of {auto, min, max}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing.
    earlyStopping=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath="../model/model/model_weights_"+str(i)+".h5", verbose=1, save_best_only=True)
    
    #调用fit方法
    #shuffle=True，数据经过随机打乱
    #verbose=1，训练过程中输出的信息，1为输出进度条记录
    #validation_data，指定验证集
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid,Y_valid),callbacks=[checkpointer,earlyStopping])

    #load the best model
    #model = model_from_json(open('../model/model/model_architecture_1.json').read())
    model.load_weights('../model/model/model_weights_'+str(i)+'.h5')

    score = model.evaluate(X_train, Y_train, verbose=1)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])

    score = model.evaluate(X_valid, Y_valid, verbose=1)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])

    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    dist[i]=score[1]

######################################
#保存CNN模型
######################################
path_save_model = '/home/cad/model'
json_string = model.to_json()
open(path_save_model+'/model_architecture.json','w').write(json_string)

##根据dist.values()做一个排序，选择前5个或10个效果较好的cnn model单独保存,应用于sliding window最后判断
print(dist)
import subprocess
for i in range(3):
	index = dist.keys()[dist.values().index(max(dist.values()))]
	subprocess.call(["mv", "../model/model/model_weights_"+str(index)+".h5",path_save_model+"/model_weights_"+str(index)+".h5"])
	dist.pop(index)
