#coding:utf-8
import os
from PIL import Image
import numpy as np
import random

#读取图片，分为训练集、验证集、测试集，比例6:2:2
#若图片为灰度图，则为1通道，data[i,:,:,:] = arr
#如果是将彩色图作为输入,则为3通道，data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data(path_train, path_valid, path_train_0, path_valid_0):
        imgs_train = os.listdir(path_train)
        imgs_valid = os.listdir(path_valid)
        #imgs_test = os.listdir(path_test)
        imgs_train_0 = os.listdir(path_train_0)
        imgs_valid_0 = os.listdir(path_valid_0)
        #imgs_test_0 = os.listdir(path_test_0)
        num_train = len(imgs_train)        
        num_valid = len(imgs_valid)
        #num_test = len(imgs_test)
        num_train_0 = len(imgs_train_0)
        num_valid_0 = len(imgs_valid_0)
        #num_test_0 = len(imgs_test_0)
        X_valid = np.empty((num_valid+num_valid_0,3,64,32),dtype="float32")
        #X_test = np.empty((num_test+num_test_0,3,64,32),dtype="float32")
        X_train = np.empty((num_train + num_train_0,3,64,32),dtype="float32")
        Y_valid = np.empty((num_valid+num_valid_0,),dtype="uint8")
        #Y_test = np.empty((num_test+num_test_0,),dtype="uint8")
        Y_train = np.empty((num_train + num_train_0,),dtype="uint8")

        for i in range(num_train):
                img = Image.open(path_train+'/'+imgs_train[i])
                arr = np.asarray(img,dtype="float32")
                X_train[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                Y_train[i] = 1
        for i in range(num_valid):
                img = Image.open(path_valid+'/'+imgs_valid[i])
                arr = np.asarray(img,dtype="float32")
                X_valid[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                Y_valid[i] = 1
	'''
        for i in range(num_test):
                img = Image.open(path_test+'/'+imgs_test[i])
                arr = np.asarray(img,dtype="float32")
                X_test[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                Y_test[i] = 1
	'''
        for i in range(num_train_0):
                img = Image.open(path_train_0+'/'+imgs_train_0[i])
                arr = np.asarray(img,dtype="float32")      
                X_train[num_train+i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                Y_train[num_train+i] = 0
        for i in range(num_valid_0):
                img = Image.open(path_valid_0+'/'+imgs_valid_0[i])
                arr = np.asarray(img,dtype="float32")
                X_valid[num_valid+i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                Y_valid[num_valid+i] = 0
	'''
        for i in range(num_test_0):
                img = Image.open(path_test_0+'/'+imgs_test_0[i])
                arr = np.asarray(img,dtype="float32")
                X_test[num_test+i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                Y_test[num_test+i] = 0
	'''
        X_valid /= 255
        #X_test /= 255
        X_train /= 255
#	print(X_train[len(X_train)-1])
#	X_train_mean = np.mean(X_train)
#	print(X_train_mean)
#	X_train -= X_train_mean
#	X_valid -= X_train_mean
#	X_test -= X_train_mean

#	X_train_shuffle = np.empty((num_train + num_train_0 - 2*(num_train_0/5),3,64,32),dtype="float32")
#	Y_train_shuffle = np.empty((num_train + num_train_0 - 2*(num_train_0/5),),dtype="uint8")
#	list_train = range(X_train.shape[0])
#        random.shuffle(list_train)
#	for i in range(X_train.shape[0]):
#		X_train_shuffle[i] = X_train[list_train[i]]
#		Y_train_shuffle[i] = Y_train[list_train[i]]
        return X_train,Y_train,X_valid,Y_valid
