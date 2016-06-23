# USAGE
# python sliding_window.py --image path-to-image 
#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

import sys
if not "/home/cad/git/goods-counter/lib/pyimagesearch/" in sys.path:
	sys.path.append("/home/cad/git/goods-counter/lib/pyimagesearch/")

from nms import non_max_suppression_slow,expansion,judge,readtxt
from helpers import sliding_window
from imutils import resize,exifrotate
import argparse
import cv2
import h5py
from keras.models import model_from_json
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and define the window width and height
exifrotate(args["image"])
image = cv2.imread(args["image"])
if image.shape[1]>image.shape[0]:
	image = resize(image,width = 900)
else:
	image = resize(image,width = 600)
#cv2.imwrite('/home/cad/dataset/images/result/src_image.jpg',image)

# set parameters
(winW, winH) = (32,64)
scale = 1.2
stepsize = 4
threshold = 0.9

#load model
model = model_from_json(open('/home/cad/model/model_architecture.json').read())
model.load_weights('/home/cad/model/model_weights_7.h5')

imagepath = args["image"].split('/')
filename = imagepath[len(imagepath)-1]
#print filename

detect_list = []
detect_num = 0
#detect_0 = 0
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
		arr = np.asarray(window_resize,dtype="float32")
		data[0,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
      	        data /= 255
		result = model.predict(data, batch_size=1, verbose = 0)
		if result[0][1] > threshold:
			detect_list.append([x,y,x+winW,y+winH])
			print x,y,winW,winH
			print result[0]
		#	cv2.imwrite('/home/cad/dataset/images/valid/a_'+str(detect_num)+'.png',window_resize)
		#	detect_num += 1
	winW = int(winW * scale)
	winH = 2 * winW
	if winW > 90:
		break

if len(detect_list) > 0:
	#save initial detect result
	img = image.copy()
	for (x1,y1,x2,y2) in detect_list:
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.imwrite('/home/cad/dataset/images/result/src_image.jpg',img)

	#expansion
	print "[x] %d initial bounding boxes" % (len(detect_list))
	detect_list = np.array(detect_list) 
	detect_list = expansion(detect_list)
	img = image.copy()
	for (x1,y1,x2,y2) in detect_list:
	        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv2.imwrite('/home/cad/dataset/images/result/expansion.jpg',img)

	#Apply non-maximum Suppression and save the final result
	boundingBoxes = np.array(detect_list)
	print "[x] after applying expansion, %d bounding boxes" % (len(boundingBoxes))
	pick = non_max_suppression_slow(boundingBoxes, 0.35)
	print "[x] after applying non-maximum, %d bounding boxes" % (len(pick))

	pick_src = readtxt(filename)
	if len(pick_src) > 0: 
		pick_right = judge(np.array(pick),np.array(pick_src),0.35)

		for (startX, startY, endX, endY) in pick:
			if (startX, startY, endX, endY) in pick_right:
				cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
			else:
				cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.imwrite('/home/cad/dataset/images/result/result.jpg',image)
		cv2.waitKey(0)
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
                cv2.imwrite('/home/cad/dataset/images/result/result.jpg',image)
                cv2.waitKey(0)
else:
	pick_src = readtxt(filename)
	print "TP = %d" % (0)
        print "FP = %d" % (0)
        print "TN = %d" % (len(pick_src))
        print "accuracy = %s" % (0)
        print "error = %s" % (0)
