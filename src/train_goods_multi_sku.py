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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import PReLU
from data_multi_classes import load_data
import h5py
import argparse
from keras.models import model_from_json
#import numpy as np
#np.random.seed(1337)  # for reproducibility
#from keras.utils.visualize_util import plot
#import tensorflow as tf
#import theano
#theano.config.device = 'gpu1'
#theano.config.floatX = 'float32'

#Number of samples per gradient update
batch_size = 100
#分类个数
nb_classes = 3
#the number of times to iterate over the training data arrays
nb_epoch = 50
#训练CNN的个数
nb_loop = 10

# input image dimensions
img_rows, img_cols = 64, 32
# the images are RGB
img_channels = 3
'''
ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="Path to the training set")
ap.add_argument("--valid", required=True, help="Path to the validation set")
ap.add_argument("--test", required=True, help="Path to the testing set")
ap.add_argument("--0", required=True, help="Path to the negative set")
args = vars(ap.parse_args())

#加载数据
path_train = args["train"]
path_valid = args["valid"]
path_test = args["test"]
path_0 = args["0"]
'''
#path_train_1='/home/cad/dataset/pinkall/train_1'
#path_valid_1='/home/cad/dataset/pinkall/valid_test_1'
#path_train_2='/home/cad/dataset/bigyellow/train_1'
#path_valid_2='/home/cad/dataset/bigyellow/valid_test_1'
#path_train_0='/home/cad/dataset/negative/pink_yellow_158'
#path_valid_0='/home/cad/dataset/bigyellow/valid_test_0'

data_path = '/home/cad/dataset_new'
X_train,Y_train_label,X_valid,Y_valid_label = load_data(data_path)
print(X_train.shape[0], 'training samples')
print(X_valid.shape[0],'validation samples')
#print(X_test.shape[0],'testing samples')

#label为0,1两个类别，keras要求格式为binary class matrices,转化一下，调用keras提供的函数
Y_train = np_utils.to_categorical(Y_train_label, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid_label, nb_classes)
#Y_test = np_utils.to_categorical(Y_test_label, nb_classes)

dist={}
#prelu=PReLU(init='zero', weights=None)
for i in range(nb_loop):
    ###############
    #开始建立CNN模型
    ###############
    model = Sequential()
    #第一个卷积层，32个卷积核，每个卷积核大小5*5
    #border_mode为valid
    #激活函数用relu
    model.add(Convolution2D(32, 5, 5,
            #        border_mode='valid',
            #        activation='relu',
            #        W_regularizer=l2(0.01),
            #        b_regularizer=l2(0.01),
            #       activity_regularizer=activity_l2(0.001),
                    input_shape=(img_channels, img_rows, img_cols)))
                    #input_shape=(X_train.shape[1:])))
    model.add(Activation('relu'))
#    model.add(prelu)
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.5))

    #第二个卷积层，32个卷积核，每个卷积核大小5*5
    #采用maxpooling,poolsize为（2,2）
    #使用Dropout
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #第三个卷积层，64个卷积核，每个卷积核大小3*3
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #全连接层
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #Softmax回归，输出nb_classes个类别
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
    sgd = SGD(lr=0.004, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #this will do preprocessing and realtime data augmentation
    '''
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    ''' 
    #datagen.fit(X_train)
    #Stop training when a monitored quantity has stopped improving.
    #monitor: quantity to be monitored.
    #patience: number of epochs with no improvement after which training will be stopped.
    #verbose: verbosity mode.
    #mode: one of {auto, min, max}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing.
    earlyStopping= EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(filepath="../model/model/model_weights_"+str(i)+".h5", verbose=1, save_best_only=True)
#    class_weight = {0:1,1:5} 
    #调用fit方法
    #shuffle=True，数据经过随机打乱
    #verbose=1，训练过程中输出的信息，1为输出进度条记录
    #validation_data，指定验证集
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid,Y_valid),callbacks=[checkpointer,earlyStopping])
    '''
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size,
			shuffle=True),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
			verbose=1,
                        validation_data=(X_valid, Y_valid),
			callbacks=[checkpointer,earlyStopping])
    '''
    #load the best model
    model.load_weights('../model/model/model_weights_'+str(i)+'.h5')


    score = model.evaluate(X_train, Y_train, verbose=1)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    count = 0
    result = model.predict(X_train, batch_size=100, verbose = 0)
    for j in range(X_train.shape[0]):
	for label in range(nb_classes):
	    if label == 0:
		if Y_train_label[j] == 0 and result[j][0] > 0.01:
			count += 1
	    else:
            	if Y_train_label[j] == label and result[j][label] > 0.99:
                	count += 1
    print(('Strict Train accuracy:', float(count)/X_train.shape[0]))


    score = model.evaluate(X_valid, Y_valid, verbose=1)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    count = 0
    result = model.predict(X_valid, batch_size=100, verbose = 0)
    for j in range(X_valid.shape[0]):
	for label in range(nb_classes):
	    if label == 0:
		if Y_valid_label[j] == 0 and result[j][0] > 0.01:
            		count += 1
            else:
	    	if Y_valid_label[j] == label and result[j][label] > 0.99:
            		count += 1
    print(('Strict Validation accuracy:', float(count)/X_valid.shape[0]))

    
    '''
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    count = 0
    result = model.predict(X_test, batch_size=100, verbose = 0)
    for j in range(X_test.shape[0]):
        if Y_test_label[j] == 1 and result[j][1] > 0.9:
            count += 1
        elif Y_test_label[j] == 0 and result[j][0] > 0.5:
            count += 1
    print(('Strict Test accuracy:', float(count)/X_test.shape[0]))
    '''
#    dist[i]=score[0]
    dist[i]=float(count)/X_valid.shape[0]

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
