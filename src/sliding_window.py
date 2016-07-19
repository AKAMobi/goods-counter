import sys
import os
if not "/home/cad/git/goods-counter/lib/pyimagesearch/" in sys.path:
	sys.path.append("/home/cad/git/goods-counter/lib/pyimagesearch/")
from nms import non_max_suppression_fast,expansion,judge,readtxt
from helpers import sliding_window
from imutils import resize,exifrotate
import argparse
import cv2
import h5py
from keras.models import model_from_json
import numpy as np
from PIL import Image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
path = args["image"]
files = os.listdir(path)
#X_train_mean = 0.505046

#load model
#model2 = model_from_json(open('/home/cad/model/model_architecture.json').read())
#model2.load_weights('/home/cad/model/model_weights_8.h5')
#model = model_from_json(open('/home/cad/cnnmodel/pinkall/model1/model_architecture.json').read())
#model.load_weights('/home/cad/cnnmodel/pinkall/model1/model_weights_55.h5')
model = model_from_json(open('/home/cad/model/model_architecture.json').read())
model.load_weights('/home/cad/model/model_weights_7.h5')

#set parameters
scale = 1.15
stepsize = 5
threshold = 0.99

for filename in files:
	print filename
	# load the image and resize
	exifrotate(path+'/'+filename)
	image = cv2.imread(path+'/'+filename)
	if image.shape[1] > image.shape[0]:
		image = resize(image,width = 800)
	else:
		image = resize(image,width = 550)	
	# define initial window size
	(winW, winH) = (32,64)
	
	detect_list = []
	#nb_size = 0
	#detect_num = 0
	# loop
	while winW < 75:
		# loop over the sliding window for each windowSize
		print 'winW=',winW,'winH=',winH
		list_tmp = []
		for (x, y, window) in sliding_window(image, stepSize=stepsize, windowSize=(winW, winH)):
			# if the window does not meet the desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
	
			# THIS IS WHERE TO PROCESS THE WINDOW
			# APPLYING CNN MODEL CLASSIFIER TO CLASSIFY THE OBJECT OF THE WINDOW
			window_resize = cv2.resize(window,(32,64),interpolation=cv2.INTER_CUBIC)
			data = np.empty((1,3,64,32),dtype="float32")
			#cv2.imwrite('/home/cad/tmp.png',window_resize)
			#img = Image.open('/home/cad/tmp.png')
                	#arr = np.asarray(img,dtype="float32")
			window_resize_RGB = cv2.cvtColor(window_resize, cv2.COLOR_BGR2RGB)
			arr = np.asarray(window_resize_RGB,dtype="float32")
			
			data[0,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
			data /= 255
			#data -= X_train_mean
			result = model.predict(data, batch_size=1, verbose = 0)
			#result2 = model2.predict(data, batch_size=1, verbose = 0)
			if result[0][1] > threshold:
			#	result2 = model2.predict(data, batch_size=1, verbose = 0)
			#	if result2[0][1] > 0.99:
				# detect_list.append([x,y,x+winW,y+winH])
					list_tmp.append([x,y,x+winW,y+winH])
			#		cv2.imwrite('/home/cad/dataset/images/valid/'+filename.split('.')[0]+'_'+str(detect_num)+'.png',window_resize)
			#		detect_num += 1
		print(len(list_tmp))
		if len(list_tmp) < 8 and winW > 41:
			break
		for idx in range(len(list_tmp)):
                        detect_list.append(list_tmp[idx])
		winW = int(winW * scale)
		winH = 2 * winW

	if len(detect_list) > 0:
		#save initial detect result
		img = image.copy()
		for (x1,y1,x2,y2) in detect_list:
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.imwrite('/home/cad/dataset/images/result/'+filename,img)
	
		#expansion
		print "[x] %d initial bounding boxes" % (len(detect_list))
		detect_list = np.array(detect_list) 
		detect_list = expansion(detect_list)
		'''
		img = image.copy()
		for (x1,y1,x2,y2) in detect_list:
				cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.imwrite('/home/cad/dataset/images/result/'+filename.split('.')[0]+'_expansion.jpg',img)
		'''
		#Apply non-maximum Suppression and save the final result
		boundingBoxes = np.array(detect_list)
		print "[x] after applying expansion, %d bounding boxes" % (len(boundingBoxes))
		pick = non_max_suppression_fast(boundingBoxes, 0.43)
		print "[x] after applying non-maximum, %d bounding boxes" % (len(pick))
	
		pick_src = readtxt(path='/home/cad/dataset/images/location.txt',filename = filename)
		if len(pick_src) > 0: 
			pick_right = judge(np.array(pick),np.array(pick_src),0.4)
	
			for (startX, startY, endX, endY) in pick:
				cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			for (startX, startY, endX, endY) in pick_right:
				cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cv2.imwrite('/home/cad/dataset/images/result/'+filename.split('.')[0]+'_result.jpg',image)
			TP = len(pick_right)
			FP = len(pick)-len(pick_right)
			TN = len(pick_src)-len(pick_right)
			print "TP = %d" % (TP)
			print "FP = %d" % (FP)
			print "TN = %d" % (TN)
			print "accuracy = %s" % (float(TP)/(TP+TN))
			print "error = %s" % (float(FP)/(TP+TN))
		else:
			for (startX, startY, endX, endY) in pick:
				cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cv2.imwrite('/home/cad/dataset/images/result/'+filename.split('.')[0]+'_result.jpg',image)
	else:
		pick_src = readtxt(path='/home/cad/dataset/images/location.txt',filename = filename)
		print "TP = %d" % (0)
		print "FP = %d" % (0)
		print "TN = %d" % (len(pick_src))
		print "accuracy = %s" % (0)
		print "error = %s" % (0)
cv2.waitKey(0)
