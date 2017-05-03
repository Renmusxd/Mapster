import tensorflow as tf
import numpy
import os
from matplotlib import pyplot

class MapModel:
    def __init__(self, input_layer, output_placeholder,
                 shape, cube_shape=10,
                 learning_rate=0.001):
        self.input_placeholder = input_layer
        self.output_placeholder = output_placeholder
        self.shape = list(shape)
        self.cube_shape = cube_shape

        last_size = cube_shape*cube_shape
        weights = []
        biases = []
        layers = []
        layer = self.input_placeholder
        for s in list(self.shape):
            w = tf.Variable(tf.random_normal([last_size, s]))
            b = tf.Variable(tf.random_normal([s]))
            layer = tf.nn.relu(tf.add(tf.matmul(layer, w), b))

            weights.append(w)
            biases.append(b)
            layers.append(layer)
            last_size = s
        w = tf.Variable(tf.random_normal([last_size, self.cube_shape]))
        b = tf.Variable(tf.random_normal([self.cube_shape]))
        out_layer = tf.add(tf.matmul(layer, w), b)

        weights.append(w)
        biases.append(b)

        self.out_layer = out_layer
        self.weights = weights
        self.biases = biases

        self.cost = tf.reduce_mean(tf.squared_difference(self.out_layer, output_placeholder))
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            centered=True,
            momentum=0.05).minimize(self.cost)

    def train(self,sess,batch_maker,training_epochs,total_batch):
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_x, batch_y = batch_maker.next_batch(pred_size=self.cube_shape)
                _, c = sess.run([self.optimizer, self.cost],
                                feed_dict={self.input_placeholder: batch_x,
                                           self.output_placeholder: batch_y})
                avg_cost += c / total_batch
            print("Epoch: {}\tcost: {:.9f}".format(epoch, avg_cost))

    def predict(self,sess,xs):
        y = sess.run([self.out_layer], feed_dict={self.input_placeholder: xs})
        return y

    def make_single_model(self):
        pass

    def make_training_model(self):
        pass

    def export_weights(self,sess):
        ws = sess.run(self.weights)
        bs = sess.run(self.biases)
        return ws, bs

    def export_diagnostics(self,sess,directory):
        ws, bs = self.export_weights(sess)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        sqrt_last_size = self.cube_shape

        ws_list = []
        for i in range(ws[0].shape[1]):
            v = ws[0][:,i].reshape((sqrt_last_size,sqrt_last_size))
            ws_list.append((numpy.sum(numpy.abs(v)),v))
        ws_major = [w[1] for w in sorted(ws_list,key=lambda x:x[0])]

        fig, axes = pyplot.subplots(10,10)
        for i,ax in enumerate(axes.flatten()):
            im = ax.imshow(ws_major[i],vmin=-3,vmax=3)
        pyplot.colorbar(im,ax=axes.ravel().tolist())
        pyplot.savefig(os.path.join(directory,'topW.png'))
        pyplot.clf()

    def make_map(self,sess,init,shape=(100,100)):
        '''
        Make a map
        :param init: initial seed for map
        :param shape: how big of a map
        :return: A map!
        '''
        map = numpy.zeros(shape)
        init[init<0] = 0
        init[init>1] = 1
        map[:init.shape[0],:init.shape[1]] = init
        for col_pos in range(0,shape[1],self.cube_shape):
            for row_pos in range(shape[0]-init.shape[0]):
                xs = [numpy.rot90(map[row_pos:row_pos + self.cube_shape, col_pos:col_pos + self.cube_shape],2)
                          .reshape((self.cube_shape*self.cube_shape,))]
                ys = self.predict(sess,xs)[0]
                ys[ys<0] = 0
                ys[ys>1] = 1
                map[row_pos + self.cube_shape,col_pos:col_pos + self.cube_shape] = ys
            if col_pos < shape[1]-init.shape[1]:
                for col_help in range(self.cube_shape):
                    xs = [numpy.rot90(map[0:self.cube_shape, col_pos+col_help:col_pos+col_help + self.cube_shape], 1)
                              .reshape((self.cube_shape*self.cube_shape,))]
                    ys = self.predict(sess, xs)[0]
                    ys[ys < 0] = 0
                    ys[ys > 1] = 1
                    map[0:self.cube_shape, col_pos+col_help+self.cube_shape] = numpy.flip(ys,0)
        # Now do 5 y changes
        pyplot.imshow(map)
        pyplot.show()






