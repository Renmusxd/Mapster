import numpy
import scipy.ndimage
from matplotlib import pyplot


class BatchManager:
    def __init__(self,datapath):
        # load in image as array
        self.img = scipy.ndimage.imread(datapath, True)/255.0
        self.img[self.img>0.8] = 1
        self.img[self.img<0.8] = 0

    def next_batch(self,pred_size=10,batch_size=64):
        m,n = numpy.shape(self.img)
        xs = []  # list of length batch_size containing samples of size data_shape
        ys = []  # list of length batch_size containing samples of size batch_size

        for b in range(batch_size):
            i = numpy.random.randint(1,m-pred_size-1)
            j = numpy.random.randint(1,n-pred_size-1)

            context = self.img[i-1:i+pred_size+1, j-1:j+pred_size+1]
            context = numpy.rot90(context,numpy.random.randint(0,4))

            subimg = context[1:-1, 1:-1]  # NxN subimage for x vector
            ysample = context[0,1:-1]  # Top line for y vector
            xsample = subimg.reshape((pred_size*pred_size,))

            xs.append(xsample)
            ys.append(ysample)
        return numpy.array(xs), numpy.array(ys)

if __name__ == "__main__":
    import sys
    b = BatchManager("image/map2.jpg")
    b.next_batch(batch_size=int(sys.argv[1]))
