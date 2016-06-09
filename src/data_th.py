#coding:utf-8

import os
from PIL import Image
import numpy as np
		
#读取图片，分为训练集、验证集、测试集，比例6:2:2
#若图片为灰度图，则为1通道，data[i,:,:,:] = arr
#如果是将彩色图作为输入,则为3通道，data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data(path_positive,path_negative):
	imgs_positive = os.listdir(path_positive)
	imgs_negative = os.listdir(path_negative)
        num_positive = len(imgs_positive)
	num_negative = len(imgs_negative)
	X_valid = np.empty((num_positive/5 + num_negative/5,3,64,32),dtype="float32")
	X_test = np.empty((num_positive/5 + num_negative/5,3,64,32),dtype="float32")
	X_train = np.empty((num_positive + num_negative - len(X_valid) - len(X_test),3,64,32),dtype="float32")
	Y_valid = np.empty((num_positive/5 + num_negative/5,),dtype="uint8")
	Y_test = np.empty((num_positive/5 + num_negative/5,),dtype="uint8")
	Y_train = np.empty((num_positive + num_negative - len(Y_valid) - len(Y_test),),dtype="uint8")

	for i in range(num_positive):
		img = Image.open(path_positive+'/'+imgs_positive[i])
		arr = np.asarray(img,dtype="float32")
		if i % 5 == 3 and i / 5 < num_positive / 5:
			X_valid[i/5,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                	Y_valid[i/5] = 1
		elif i % 5 == 4 and i / 5 < num_positive / 5:
			X_test[i/5,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                        Y_test[i/5] = 1
		else:
			X_train[i - 2 * (i/5),:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
			Y_train[i - 2 * (i/5)] = 1
	for i in range(num_negative):
                img = Image.open(path_negative+'/'+imgs_negative[i])
                arr = np.asarray(img,dtype="float32")
		if i % 5 == 3 and i / 5 < num_negative / 5:
                        X_valid[num_positive / 5 + i/5,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                        Y_valid[num_positive / 5 + i/5] = 0
                elif i % 5 == 4 and i / 5 < num_negative / 5:
                        X_test[num_positive / 5 + i/5,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                        Y_test[num_positive / 5 + i/5] = 0
                else:
                        X_train[num_positive - 2 * (num_positive / 5) + i - 2 * (i/5),:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                        Y_train[num_positive - 2 * (num_positive / 5) + i - 2 * (i/5)] = 0

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
