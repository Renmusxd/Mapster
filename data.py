import numpy
import scipy.ndimage
from matplotlib import pyplot


class BatchManager:
    def __init__(self,datapath,x_shape=(10,10),y_shape=(1,10),pregen=0):
        """
        A batch manager for making data batches
        :param datapath: 
        :param pregen: 
        """
        # load in image as array
        self.img = scipy.ndimage.imread(datapath, True)/255.0
        self.img[self.img<0.5] = 0
        self.x_shape = x_shape
        self.y_shape = y_shape
        if self.y_shape[1] > self.x_shape[1]:
            raise Exception("Width of prediction cannot surpass width of input")

        if pregen > 0:
            self.xs, self.ys = self.make_batch(batch_size=pregen)
            self.next_batch = self.get_batch
        else:
            self.xs, self.ys = None, None
            self.next_batch = self.make_batch

    def get_batch(self,batch_size=64):
        rs = numpy.random.randint(0,len(self.xs),(batch_size,))
        return self.xs[rs], self.ys[rs]

    def get_data(self,size=None):
        if size is None and self.xs is None:
            raise Exception("Need pregen or defined size")
        elif size is None and self.xs is not None:
            size = self.xs.shape[0]
        elif size is not None and self.xs is None:
            self.xs, self.ys = self.make_batch(size)
        elif size > self.xs.shape[0]:
            new_xs, new_ys = self.make_batch(size - self.xs.shape[0])
            self.xs = numpy.concatenate((self.xs,new_xs))
            self.ys = numpy.concatenate((self.ys,new_ys))
        return self.xs[:size], self.ys[:size]

    def make_batch(self,batch_size=64):
        m,n = numpy.shape(self.img)
        xs = []  # list of length batch_size containing samples of size data_shape
        ys = []  # list of length batch_size containing samples of size batch_size

        for b in range(batch_size):
            # Up-Down or Left-Right
            if numpy.random.random()<0.5:
                i = numpy.random.randint(0, m - (self.x_shape[0] + self.y_shape[0]) )
                j = numpy.random.randint(0, n - self.x_shape[1])
                sample = self.img[i:i+self.x_shape[0]+self.y_shape[0],j:j+self.x_shape[1]]

            else:
                i = numpy.random.randint(0, m - self.x_shape[1])
                j = numpy.random.randint(0, n - (self.x_shape[0] + self.y_shape[0]))
                sample = self.img[i:i + self.x_shape[1], j:j + self.x_shape[0] + self.y_shape[0]]
                sample = numpy.rot90(sample)  # Align vertically

            sample = numpy.rot90(sample,numpy.random.randint(0,2)*2) # Flip or do not flip

            ximg = sample[self.y_shape[0]:,:self.x_shape[1]]  # Bottom of image
            yimg = sample[:self.y_shape[0],:self.y_shape[1]]  # Top shape of image

            xs.append(ximg.flatten())
            ys.append(yimg.flatten())
        return numpy.array(xs), numpy.array(ys)

if __name__ == "__main__":
    import sys
    b = BatchManager("image/map2.jpg",x_shape=(2,8),y_shape=(1,8))
    b.next_batch(batch_size=int(sys.argv[1]))
