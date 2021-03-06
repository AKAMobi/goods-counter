
import imutils

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def sliding_window_variable_stepsize(start_x, start_y, image, stepSize, windowSize):
        # slide a window across the image
        for y in xrange(start_y, image.shape[0], stepSize):
                for x in xrange(start_x, image.shape[1], stepSize):
                        # yield the current window
                        yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def sliding_window_1D(image, stepSize, windowWidth):
	#slide a window 1D
	for x in xrange(0, image.shape[1], stepSize):
		yield (x, image[0:image.shape[0],x:x + windowWidth])
