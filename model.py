import tensorflow as tf
import numpy
from matplotlib import pyplot

class MapModel:
    def __init__(self, input_layer, output_placeholder,
                 input_size, shape, output_size,
                 learning_rate=0.001):
        self.input_placeholder = input_layer
        self.output_placeholder = output_placeholder
        self.shape = [input_size] + list(shape) + [output_size]

        last_size = self.shape[0]
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
        w = tf.Variable(tf.random_normal([last_size, self.shape[-1]]))
        b = tf.Variable(tf.random_normal([self.shape[-1]]))
        out_layer = tf.nn.relu(tf.add(tf.matmul(layer, w), b))

        weights.append(w)
        biases.append(b)

        self.out_layer = out_layer
        self.weights = weights
        self.biases = biases

        self.cost = tf.reduce_mean(tf.squared_difference(self.out_layer, output_placeholder))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def train(self,sess,batch_maker,training_epochs,total_batch):
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_x, batch_y = batch_maker.next_batch()
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

    def make_map(self,sess,init,shape=(100,100)):
        '''
        Make a map
        :param init: initial seed for map
        :param shape: how big of a map
        :return: A map!
        '''
        map = numpy.zeros(shape)
        map[:init.shape[0],:init.shape[1]] = init

        for col_pos in range(0,shape[1],10):
            print("Colp:",col_pos)
            for row_pos in range(shape[0]-init.shape[0]):
                print("Rowp",row_pos)
                #TODO: Deal with magic numbers
                xs = [numpy.rot90(map[row_pos:row_pos + 10, col_pos:col_pos + 10],2).reshape((100,))]
                ys = self.predict(sess,xs)
                map[row_pos + 10,col_pos:col_pos + 10] = ys[0]
            if col_pos < shape[1]-init.shape[1]:
                for col_help in range(10):
                    print("Colh:",col_help)
                    xs = [numpy.rot90(
                                map[row_pos:row_pos + 10,
                                    col_pos+col_help:col_pos+col_help + 10],
                                3).reshape((100,))]
                    ys = self.predict(sess, xs)
                    map[row_pos:row_pos + 10, col_pos+col_help] = ys[0]
        # Now do 5 y changes
        pyplot.imshow(map)
        pyplot.show()

        map = numpy.minimum(0,numpy.maximum(1,map))




