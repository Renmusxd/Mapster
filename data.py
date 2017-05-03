import numpy
import scipy.ndimage

class BatchManager:
	def __init__(self,datapath):
    	# load in image as array
    	
    	self.img = scipy.ndimage.imread(os.path.join(datapath), True)/255.0


	def next_batch(self,data_shape=100,pred_size=10,batch_size=64):

    	img = self.img

		m,n = numpy.shape(img)

		xs = []  # list of length batch_size containing samples of size data_shape
		ys = []  # list of length batch_size containing samples of size batch_size

		for b in range(batch_size):
			i = numpy.random.randint(1,m-pred_size-1)
			j = numpy.random.randint(1,n-pred_size-1)
			r = numpy.random.random()
			if (r < 0.25):
				sample = img[i:i+pred_size, j:j+pred_size]
				xs.append(sample.reshape((data_shape,)))

				ysample = img[i-1, j:j+pred_size]
				ys.append(ysample)
			elif (0.25 <= r and r < 0.5):
				# rotate twice
				sample = img[i:i+pred_size, j:j+pred_size]
				sample = numpy.rot90(sample, 2)
				xs.append(sample.reshape((data_shape,)))

				ysample = img[i+pred_size, j:j+pred_size]
				ys.append(ysample)
			elif (0.5 <= r and r < 0.75):
				# rotate once
				sample = img[i:i+pred_size, j:j+pred_size]
				sample = numpy.rot90(sample, 1)
				xs.append(sample.reshape((data_shape,)))

				ysample = img[i:i+pred_size, j-1]
				ys.append(ysample)
			else:
				# rotate three times
				sample = img[i:i+pred_size, j:j+pred_size]
				sample = numpy.rot90(sample, 3)
				xs.append(sample.reshape((data_shape,)))

				ysample = img[i:i+pred_size, j+pred_size]
				ys.append(ysample)
		return numpy.array(xs), numpy.array(ys)

if __name__ == "__main__":
	b = BatchManager(None)
	b.next_batch()