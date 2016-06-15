#coding:utf-8

import os
from PIL import Image
import numpy as np
		
#读取图片，分为训练集、验证集、测试集，比例6:2:2
#若图片为灰度图，则为1通道，data[i,:,:,:] = arr
#如果是将彩色图作为输入,则为3通道，data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data(path_positive_train, path_negative_train, path_positive_valid_test, path_negative_valid_test):
	imgs_positive_train = os.listdir(path_positive_train)
	imgs_negative_train = os.listdir(path_negative_train)
	imgs_positive_valid_test = os.listdir(path_positive_valid_test)
	imgs_negative_valid_test = os.listdir(path_negative_valid_test)
        num_positive_train = len(imgs_positive_train)
	num_negative_train = len(imgs_negative_train)
	num_positive_valid_test = len(imgs_positive_valid_test)
	num_negative_valid_test = len(imgs_negative_valid_test)
	X_valid = np.empty((num_positive_valid_test/2 + num_negative_valid_test/2,3,64,32),dtype="float32")
	X_test = np.empty((num_positive_valid_test - num_positive_valid_test/2 + num_negative_valid_test - num_negative_valid_test/2,3,64,32),dtype="float32")
	X_train = np.empty((num_positive_train + num_negative_train,3,64,32),dtype="float32")
	Y_valid = np.empty((num_positive_valid_test/2 + num_negative_valid_test/2,),dtype="uint8")
	Y_test = np.empty((num_positive_valid_test - num_positive_valid_test/2 + num_negative_valid_test - num_negative_valid_test/2,),dtype="uint8")
	Y_train = np.empty((num_positive_train + num_negative_train,),dtype="uint8")

	for i in range(num_positive_train):
		img = Image.open(path_positive_train+'/'+imgs_positive_train[i])
		arr = np.asarray(img,dtype="float32")
		X_train[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
		Y_train[i] = 1
	for i in range(num_negative_train):
                img = Image.open(path_negative_train+'/'+imgs_negative_train[i])
                arr = np.asarray(img,dtype="float32")
                X_train[num_positive_train+i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                Y_train[num_positive_train+i] = 0
	for i in range(num_positive_valid_test):
                img = Image.open(path_positive_valid_test+'/'+imgs_positive_valid_test[i])
                arr = np.asarray(img,dtype="float32")
		if i%2==1:
                	X_valid[i/2,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                	Y_valid[i/2] = 1
		else:
			X_test[i/2,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                        Y_test[i/2] = 1
	for i in range(num_negative_valid_test):
                img = Image.open(path_negative_valid_test+'/'+imgs_negative_valid_test[i])
                arr = np.asarray(img,dtype="float32")
                if i%2==1:
                        X_valid[num_positive_valid_test/2+i/2,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                        Y_valid[num_positive_valid_test/2+i/2] = 0
                else:
                        X_test[num_positive_valid_test-num_positive_valid_test/2+i/2,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                        Y_test[num_positive_valid_test-num_positive_valid_test/2+i/2] = 0

#	X_valid = X_valid.reshape(len(X_valid),64,32,3)
#	X_test = X_test.reshape(len(X_test),64,32,3)
#	X_train = X_train.reshape(len(X_train),64,32,3)
	#X_valid = np.transpose(X_valid / 255, (0, 2, 3, 1))
	#X_test = np.transpose(X_test / 255, (0, 2, 3, 1))
	#X_train = np.transpose(X_train / 255, (0, 2, 3, 1))
	X_valid /= 255
	X_test /= 255
	X_train /= 255
	return X_train,Y_train,X_valid,Y_valid,X_test,Y_test
