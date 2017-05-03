import numpy
import scipy.ndimage
from matplotlib import pyplot


class BatchManager:
    def __init__(self, datapath):
        # load in image as array
        self.img = scipy.ndimage.imread(datapath, True) / 255.0
        pyplot.imshow(self.img)
        pyplot.show()

    def next_batch(self, data_shape=100, pred_size=10, batch_size=64):

        img = self.img
        m, n = img.shape

        xs = []  # list of length batch_size containing samples of size data_shape
        ys = []  # list of length batch_size containing samples of size batch_size

        for b in range(batch_size):
            i = numpy.random.randint(1, m - pred_size - 1)
            j = numpy.random.randint(1, n - pred_size - 1)
            sample = img[i:i + pred_size, j:j + pred_size]
            xs.append(sample.reshape((data_shape,)))
            r = numpy.random.random()
            if (r < 0.25):
                ys.append(img[i - 1, j:j + pred_size])
            elif (0.25 <= r and r < 0.5):
                ys.append(img[i + pred_size, j:j + pred_size])
            elif (0.5 <= r and r < 0.75):
                ys.append(img[i:i + pred_size, j - 1])
            else:
                ys.append(img[i:i + pred_size, j + pred_size])

        return numpy.array(xs), numpy.array(ys)


if __name__ == "__main__":
    b = BatchManager(None)
    b.next_batch()
