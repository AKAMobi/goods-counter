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
from data_th import load_data
import h5py
import argparse
from keras.models import model_from_json
#from keras.utils.visualize_util import plot

batch_size = 60
nb_classes = 2
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 64, 32
# the images are RGB
img_channels = 3

ap = argparse.ArgumentParser()
ap.add_argument("--positive", required=True, help="Path to the positive training set")
ap.add_argument("--negative", required=True, help="Path to the negative training set")
args = vars(ap.parse_args())
		
#加载数据
path_positive = args["positive"]
path_negative = args["negative"]
X_train,Y_train,X_valid,Y_valid,X_test,Y_test = load_data(path_positive,path_negative)
print(X_train.shape[0], 'training samples')
print(X_valid.shape[0],'validation samples')
print(X_test.shape[0],'testing samples')
		
#label为0,1两个类别，keras要求格式为binary class matrices,转化一下，调用keras提供的函数
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

###############
#开始建立CNN模型
###############
model = Sequential()

#第一个卷积层，32个卷积核，每个卷积核大小5*5
#激活函数用relu
model.add(Convolution2D(32, 5, 5, 
		border_mode='valid',
        	input_shape=(img_channels, img_rows, img_cols)))
		#input_shape=(X_train.shape[1:])))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='tf'))
#model.add(Dropout(0.5))

#第二个卷积层，32个卷积核，每个卷积核大5*5
#采用maxpooling,poolsize为（2,2）
#使用Dropout
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

#第三个卷积层，64个卷积核，每个卷积核大小3*3
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

#第四个卷积层，64个卷积核，每个卷积核大小3*3
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
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
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# visualize model layout with pydot_ng
#plot(model, to_file='../model/model.png', show_shapes=True)	
	
#调用fit方法
#shuffle=True，数据经过随机打乱
#verbose=1，训练过程中输出的信息，1为输出进度条记录
#validation_data，指定验证集
#Stop training when a monitored quantity has stopped improving.
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid,Y_valid),callbacks=[earlyStopping]) 
	
score = model.evaluate(X_train, Y_train, verbose=1)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(X_valid, Y_valid, verbose=1)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

######################################
#保存CNN模型
######################################

json_string = model.to_json()
open('../model/model_similar_architecture.json','w').write(json_string)
model.save_weights('../model/model_similar_weights.h5')
