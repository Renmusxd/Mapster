import numpy

class BatchManager:
    def __init__(self,datapath):
        pass

    def next_batch(self,data_shape=100,pred_size=10,batch_size=64):
        xs = numpy.random.rand(batch_size,data_shape)
        ys = numpy.random.rand(batch_size,pred_size)

        return xs, ys