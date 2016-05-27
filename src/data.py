#coding:utf-8

import os
from PIL import Image
import numpy as np
		
#读取文件夹train下的图片，若图片为灰度图，则为1通道，data[i,:,:,:] = arr
#如果是将彩色图作为输入,则为3通道，data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data(path):
	imgs = os.listdir(path)
        num = len(imgs)
	data = np.empty((num,3,64,32),dtype="float32")
	label = np.empty((num,),dtype="uint8")

	for i in range(num):
		img = Image.open(path+'/'+imgs[i])
		arr = np.asarray(img,dtype="float32")
		data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
		label[i] = int(imgs[i].split('.')[0])
	data = data.reshape(num,64,32,3)
	data /= 255
	return data,label
