import os
from PIL import Image
import numpy as np

def load_data(data_path):
	num_train,num_valid = 0,0
        for dirs1 in os.listdir(data_path):
                num_train += len(os.listdir(data_path+'/'+dirs1+'/train'))
                num_valid += len(os.listdir(data_path+'/'+dirs1+'/valid'))

        X_valid = np.empty((num_valid,3,64,32),dtype="float32")
        X_train = np.empty((num_train,3,64,32),dtype="float32")
        Y_valid = np.empty((num_valid,),dtype="uint8")
        Y_train = np.empty((num_train,),dtype="uint8")

        i,j = 0,0
        for dirs1 in os.listdir(data_path):
                for dirs2 in os.listdir(data_path+'/'+dirs1):
                        if dirs2 == 'train':
                                for files in os.listdir(data_path+'/'+dirs1+'/train'):
                                        img = Image.open(data_path+'/'+dirs1+'/train/'+files)
                                        arr = np.asarray(img,dtype="float32")
                                        X_train[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                                        Y_train[i] = int(dirs1)
                                        i += 1
                        if dirs2 == 'valid':
                                for files in os.listdir(data_path+'/'+dirs1+'/valid'):
                                        img = Image.open(data_path+'/'+dirs1+'/valid/'+files)
                                        arr = np.asarray(img,dtype="float32")
                                        X_valid[j,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
                                        Y_valid[j] = int(dirs1)
                                        j += 1

	print i,j
        X_valid /= 255
        X_train /= 255

        return X_train,Y_train,X_valid,Y_valid

