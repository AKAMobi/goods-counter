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
from data import load_data
import h5py 
from keras.models import model_from_json
import tensorflow as tf

batch_size = 64
nb_classes = 2
nb_epoch = 3

# input image dimensions
img_rows, img_cols = 64, 32
# the images are RGB
img_channels = 3
		
#加载数据
data, label = load_data('/home/ubuntu/dataset/train_bigpink')
print(data.shape[0], ' samples')
		
#X_test, Y_test = load_data('./train')
#print(X_test.shape[0], 'test samples')
#Y_test = np_utils.to_categorical(Y_test, 2)
#label为0,1两个类别，keras要求格式为binary class matrices,转化一下，调用keras提供的函数
label = np_utils.to_categorical(label, nb_classes)

###############
#开始建立CNN模型
###############
#分配计算任务所使用设备
with tf.device('/cpu:0'):
	model = Sequential()
	#第一个卷积层，32个卷积核，每个卷积核大小3*3
	#border_mode可以为valid或者same
	#激活函数用relu
	#dim_ordering为tensorflow
	model.add(Convolution2D(32, 5, 5, 
				border_mode='valid',
        	                input_shape=(img_rows, img_cols, img_channels),
				dim_ordering='tf'))
	model.add(Activation('relu'))
#	model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='tf'))
#        model.add(Dropout(0.5))

	#第二个卷积层，32个卷积核，每个卷积核大小3*3
	#采用maxpooling,poolsize为（2,2）
	#使用Dropout
	model.add(Convolution2D(32, 5, 5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='tf'))
	model.add(Dropout(0.5))

	#第三个卷积层，64个卷积核，每个卷积核大小3*3
	#model.add(Convolution2D(64, 3, 3, border_mode='same'))
	#model.add(Activation('relu'))

	#第四个卷积层，64个卷积核，每个卷积核大小3*3
	#model.add(Convolution2D(64, 3, 3))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='tf'))
	#model.add(Dropout(0.25))

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
	#使用SGD + momentum
	#model.compile里的参数loss就是损失函数(目标函数)
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
		
	#调用fit方法
	#shuffle=True，数据经过随机打乱
	#verbose=1，训练过程中输出的信息，1为输出进度条记录
	#validation_split=0.2，将20%的数据作为验证集
	model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_split=0.2)

######################################
#保存CNN模型
######################################

json_string = model.to_json()
open('../model/model_architecture_super_bigpink_more1_add0fp2.json','w').write(json_string)
model.save_weights('../model/model_weights_super_bigpink_more1_add0fp2.h5')

#score = model.evaluate(X_test, Y_test, verbose=1)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
