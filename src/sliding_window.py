# USAGE
# python sliding_window.py --image images/1.jpg 
import sys
sys.path.append("/home/ubuntu/git/goods-counter/lib/pyimagesearch/")
from nms import non_max_suppression_slow
from helpers import sliding_window
from imutils import resize
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
image = cv2.imread(args["image"])
image = resize(image,width = 600)
# set parameters
(winW, winH) = (32,64)
scale = 2 ** 0.5
stepsize = 8
threshold = 0.9

#load model
model = model_from_json(open('../model/model_architecture_super_bigpink_more1_add0fp2.json').read())
model.load_weights('../model/model_weights_super_bigpink_more1_add0fp2.h5')

detect_list = []
#c = 0
# loop
while winW < image.shape[1] and winH < image.shape[0]:
	# loop over the sliding window for each windowSize
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
		data = data.reshape(1,64,32,3)
       	        data /= 255
		result = model.predict(data, batch_size=1, verbose = 0)
		#print result
		#result = model.predict_on_batch(data)
		if result[0][1] > threshold:
			detect_list.append([x,y,x+winW,y+winH])
			print x,y,winW,winH
			print result[0]
			#cv2.imwrite('images/valid/'+str(c)+args["image"].split('/')[1],window)
			#c += 1

	winW = int(winW * scale)
	winH = 2 * winW
	print('winW=%d,winH=%d',winW,winH)
	#if winW > image.shape[1] / 6:
	#	break

img = image.copy()
for (x1,y1,x2,y2) in detect_list:
	cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imwrite('~/dataset/images/result/src_image.jpg',img)
boundingBoxes = np.array(detect_list)
print "[x] %d initial bounding boxes" % (len(boundingBoxes))
pick = non_max_suppression_slow(boundingBoxes, 0.3)
print "[x] after applying non-maximum, %d bounding boxes" % (len(pick))
for (startX, startY, endX, endY) in pick:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imwrite('~/dataset/images/result/'+args["image"].split('/')[1],image)
cv2.waitKey(0)
