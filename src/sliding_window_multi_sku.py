import sys
import os
if not "../lib/pyimagesearch/" in sys.path:
	sys.path.append("../lib/pyimagesearch/")
from nms import non_max_suppression_slow,expansion
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

#X_train_mean = 0.508538
nb_classes = 3
BGR = [(147,20,255),(0,255,255)]

files = os.listdir(path)
for filename in files:
	print filename
	# load the image and define the window width and height
	exifrotate(path+'/'+filename)
	image = cv2.imread(path+'/'+filename)
	if image.shape[1]>image.shape[0]:
		image = resize(image,width = 800)
	else:
		image = resize(image,width = 600)
	# set parameters
	(winW, winH) = (32,64)
	scale = 1.3
	stepsize = 6
	threshold = 0.99
	#load model
	model = model_from_json(open('/home/cad/model/model_architecture.json').read())
	model.load_weights('/home/cad/model/model_weights_0.h5')
	#model = model_from_json(open('/home/cad/cnnmodel/middlepink/model4/model_architecture.json').read())
        #model.load_weights('/home/cad/cnnmodel/middlepink/model4/model_weights_6.h5')

	detect_lists = [[] for i in range(nb_classes)]

	# loop
	while winW < image.shape[1] and winH < image.shape[0]:
		# loop over the sliding window for each windowSize
		print 'winW=',winW,'winH=',winH
		for (x, y, window) in sliding_window(image, stepSize=stepsize, windowSize=(winW, winH)):
			# if the window does not meet the desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
	
			# THIS IS WHERE TO PROCESS THE WINDOW
			# APPLYING CNN MODEL CLASSIFIER TO CLASSIFY THE OBJECT OF THE WINDOW
			window_resize = cv2.resize(window,(32,64),interpolation=cv2.INTER_CUBIC)
			data = np.empty((1,3,64,32),dtype="float32")
			window_resize = cv2.cvtColor(window_resize, cv2.COLOR_BGR2RGB)
			arr = np.asarray(window_resize,dtype="float32")
			data[0,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
			data /= 255
			#data -= X_train_mean
			result = model.predict(data, batch_size=1, verbose = 0)
			for i in xrange(1,nb_classes):
				if result[0][i] > threshold:
					detect_lists[i].append([x,y,x+winW,y+winH])
				#cv2.imwrite('/home/cad/dataset/images/valid/c_'+str(detect_num)+'.png',window_resize)
				#detect_num += 1
		winW = int(winW * scale)
		winH = 2 * winW
		if winW > 70:
			break

	#save initial detect result
	img = image.copy()
	for i in xrange(1,nb_classes):
		for (x1,y1,x2,y2) in detect_lists[i]:
			cv2.rectangle(img, (x1, y1), (x2, y2), BGR[i-1], 2)
	cv2.imwrite('/home/cad/dataset/images/result/'+filename,img)

	'''
	#expansion
	img = image.copy()
	for i in xrange(1,nb_classes+1):
		if len(detect_list[i]) > 0:
			boundingBoxes = np.array(detect_list[i])
			pick = expansion(boundingBoxes)
			for (x1,y1,x2,y2) in pick:
				cv2.rectangle(img, (x1, y1), (x2, y2), BGR[i-1], 2)
	cv2.imwrite('/home/cad/dataset/images/result/'+filename.split('.')[0]+'_expansion.jpg',img)
	'''

	#Apply non-maximum Suppression and save the final result
	for i in xrange(1,nb_classes):
		boundingBoxes = np.array(detect_lists[i])
		print "[x] %d initial bounding boxes" % (len(boundingBoxes))
		pick = non_max_suppression_slow(boundingBoxes,0.5)
		print "[x] after applying non-maximum, %d bounding boxes" % (len(pick))
		for (startX, startY, endX, endY) in pick:
			cv2.rectangle(image, (startX, startY), (endX, endY), BGR[i-1], 2)
	cv2.imwrite('/home/cad/dataset/images/result/'+filename.split('.')[0]+'_result.jpg',image)
