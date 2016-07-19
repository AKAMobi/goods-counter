# import the necessary packages
import numpy as np

#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]

			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
	return boxes[pick]

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def expansion(boundingBoxes):
	bigRect = []

	x1 = boundingBoxes[:,0]
	x2 = boundingBoxes[:,2]
	y1 = boundingBoxes[:,1]
	y2 = boundingBoxes[:,3]

	width_avg = sum(x2 - x1) / len(boundingBoxes)
	height_avg = sum(y2 - y1) / len(boundingBoxes)
#	print width_avg,height_avg
	idxs_y2 = np.argsort(y2-(y2-y1)/2)

	while len(idxs_y2) > 0:
		pick = []
		index = []
		length = len(idxs_y2)
		pick.append(idxs_y2[0])
		index.append(0)
		if(length > 1):
			for pos in xrange(1, length):
				pick.append(idxs_y2[pos])
				index.append(pos)
				if (y2[idxs_y2[pos]]-(y2[idxs_y2[pos]]-y1[idxs_y2[pos]])/2) - (y2[idxs_y2[pos-1]]-(y2[idxs_y2[pos - 1]]-y1[idxs_y2[pos - 1]])/2) >= 0.8*height_avg:
					pick.pop()
					index.pop()
					break
		idxs_y2 = np.delete(idxs_y2, index)

		idx_x1 = np.argsort(x1[pick])
		pick_sort = []
		for i in range(0,len(pick)):
			pick_sort.append(pick[idx_x1[i]])

		while len(pick_sort) > 0:
			pickx = []
			index = []
			length = len(pick_sort)
			pickx.append(pick_sort[0])
			index.append(0)
			if(length > 1):
				for pos in xrange(1, length):
					pickx.append(pick_sort[pos])
					index.append(pos)
					if x1[pick_sort[pos]] - x1[pick_sort[pos - 1]] > 2.5 * width_avg:
						pickx.pop()
						index.pop()
						break
			pick_sort = np.delete(pick_sort, index)

			if len(pickx) >= 5:
				for ok in pickx:
					bigRect.append([x1[ok],y1[ok],x2[ok],y2[ok]])
	return bigRect

def readtxt(path,filename):
    list=[]
    for line in open(path):
        name = line.split(':')[0]
        if name == filename:
            location = line.split(':')[1].split(' ')
            for i in range(location.__len__()):
                point = []
                for j in range(4):
                    point.append(int(location[i].split(',')[j]))
                list.append(point)
    if len(list) == 0:
        return []
    else:
        images = [(np.array(list))]
        return images[0]

def judge(pick,pick_src,overlap):
    if len(pick)==0 or len(pick_src)==0:
        return []
    index = []

    x1 = pick[:,0]
    y1 = pick[:,1]
    x2 = pick[:,2]
    y2 = pick[:,3]

    src_x1 = pick_src[:,0]
    src_y1 = pick_src[:,1]
    src_x2 = pick_src[:,2]
    src_y2 = pick_src[:,3]

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (src_x2 - src_x1) * (src_y2 - src_y1)

    for i in range(len(pick)):
        for j in range(len(pick_src)):
            endx = max(x2[i],src_x2[j])
            startx = min(x1[i],src_x1[j])
            width = (x2[i] - x1[i]) + (src_x2[j] - src_x1[j]) - (endx - startx)

            endy = max(y2[i],src_y2[j])
            starty = min(y1[i],src_y1[j])
            height = (y2[i] - y1[i]) + (src_y2[j] - src_y1[j]) - (endy - starty)

            if width <= 0 or height <= 0:
                continue
            area = width * height
            if float(area)/area2[j] > overlap and float(area)/area1[i] > overlap:
                index.append(i)
                break
    return pick[index]
