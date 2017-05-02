import numpy

class BatchManager:
    def __init__(self,datapath):
        pass

    def next_batch(self,data_shape=(10,10),pred_size=10,batch_size=64):

        xs = []  # list of length batch_size containing samples of size data_shape
        ys = []  # list of length batch_size containing sameples of size batch_size

        return xs, ys