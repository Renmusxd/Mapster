import tensorflow as tf
import numpy
import os
from matplotlib import pyplot

class MapModel:
    def __init__(self, input_layer, output_placeholder,
                 shape, cube_shape=(10,10), pred_shape=(1,10),
                 learning_rate=0.001):
        self.shape = list(shape)
        self.cube_shape = cube_shape
        self.pred_shape = pred_shape
        self.input_placeholder = input_layer
        self.output_placeholder = output_placeholder
        self.keep_prop = tf.placeholder(tf.float32)

        last_size = numpy.prod(self.cube_shape)
        weights = []
        biases = []
        layers = []
        layer = self.input_placeholder
        for s in list(self.shape):
            w = tf.Variable(tf.random_normal([last_size, s]))
            b = tf.Variable(tf.random_normal([s]))
            layer = tf.nn.relu(
                tf.matmul(layer, w) + b
            )
            layer = tf.nn.dropout(layer,self.keep_prop)
            weights.append(w)
            biases.append(b)
            layers.append(layer)
            last_size = s
        w = tf.Variable(tf.random_normal([last_size, numpy.prod(self.pred_shape)]))
        b = tf.Variable(tf.random_normal([numpy.prod(self.pred_shape)]))
        out_layer = tf.matmul(layer, w) + b

        weights.append(w)
        biases.append(b)

        self.out_layer = out_layer
        self.weights = weights
        self.biases = biases

        self.cost = tf.reduce_mean(tf.squared_difference(self.out_layer,self.output_placeholder))

        # Regularization if wanted
        #tf.reduce_mean([tf.nn.l2_loss(weights[i]) for i in range(len(weights))])*0.0001

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            centered=True,
            momentum=0.1).minimize(self.cost)

    def train(self,sess,batch_maker,training_epochs,total_batch):
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_x, batch_y = batch_maker.next_batch()
                _, c = sess.run([self.optimizer, self.cost],
                                feed_dict={self.input_placeholder: batch_x,
                                           self.output_placeholder: batch_y,
                                           self.keep_prop: 1.0})
                avg_cost += c / total_batch
            print("Epoch: {}/{}\tcost: {:.9f}".format(epoch, training_epochs, avg_cost))

    def predict(self,sess,xs):
        y = sess.run([self.out_layer], feed_dict={self.input_placeholder: xs,
                                                  self.keep_prop: 1.0})
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
        print("[*] Exporting Weights...")
        ws, bs = self.export_weights(sess)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        ws_list = []
        for i in range(ws[0].shape[1]):
            v = ws[0][:,i].reshape(self.cube_shape)
            ws_list.append((numpy.sum(numpy.abs(v)),v))
        ws_major = [w[1] for w in reversed(sorted(ws_list,key=lambda x:x[0]))]

        print("\tGenerating figure...")
        fig, axes = pyplot.subplots(10,10,figsize=(11,10))
        for i,ax in enumerate(axes.flatten()):
            im = ax.imshow(ws_major[i],vmin=-3,vmax=3)
        pyplot.colorbar(im,ax=axes.ravel().tolist())
        print("\tSaving...")
        pyplot.savefig(os.path.join(directory,'topW.png'))
        pyplot.clf()
        print("[+] Done!")


    def make_map(self,sess,init,filename='outmap.png',shape=(100,100)):
        '''
        Make a map
        :param init: initial seed for map
        :param shape: how big of a map
        :return: A map!
        '''
        print("[*] Making map...")
        map = numpy.zeros(shape)
        init[init<0] = 0
        init[init>1] = 1
        map[:init.shape[0],:init.shape[1]] = init
        print(init.shape,self.cube_shape)
        # Complete init square (init section should be square-like)
        for row_pos in range(0,self.pred_shape[1]-self.cube_shape[0],self.pred_shape[0]):
            xs = [numpy.rot90(
                    map[row_pos:row_pos+self.cube_shape[0],
                        :self.cube_shape[1]],
                    2).reshape((numpy.prod(self.cube_shape),))
                  ]
            ys = self.predict(sess,xs)[0].reshape(self.pred_shape)
            ys[ys < 0] = 0
            ys[ys > 1] = 1

            map[
                row_pos+self.cube_shape[0] : row_pos+self.cube_shape[0]+self.pred_shape[0],
                : self.pred_shape[1]
            ] = numpy.rot90(ys,-2)
        # Now we have a self.pred_shape[1] by self.pred_shape[1] square in the top left

        # Move forward by width of prediction
        for col_pos in range(0,shape[1],self.pred_shape[1]):
            # Fill out the prediction column
            for row_pos in range(self.pred_shape[1]-self.cube_shape[0],shape[0]-init.shape[0]):
                xs = [numpy.rot90(
                        map[row_pos:row_pos + self.cube_shape[0],
                            col_pos:col_pos + self.cube_shape[1]],
                        2).reshape((numpy.prod(self.cube_shape),))
                      ]
                ys = self.predict(sess,xs)[0]

                # ys = ys * 0 + row_pos * 1.0/shape[0] + col_pos * 1.0/shape[1]
                ys[ys<0] = 0
                ys[ys>1] = 1
                map[row_pos + self.cube_shape[0]:row_pos + self.cube_shape[0] + self.pred_shape[0],
                    col_pos:col_pos + self.pred_shape[1]] = numpy.rot90(ys,-2)
            # Make the next self.pred_shape[1] x self.pred_shape[1] square
            if col_pos < shape[1]-init.shape[1]:
                for col_help in range(self.pred_shape[1]):
                    xs = [numpy.rot90(
                            map[0:self.cube_shape[1],
                                col_pos+col_help:col_pos+col_help + self.cube_shape[0]],
                            1).reshape((numpy.prod(self.cube_shape),))
                          ]
                    ys = self.predict(sess, xs)[0].reshape(self.pred_shape)

                    # ys = ys * 0 + col_help / self.pred_shape[1]
                    ys[ys < 0] = 0
                    ys[ys > 1] = 1
                    map[0:self.pred_shape[1],
                        col_pos+self.cube_shape[1]+col_help:
                            col_pos+self.cube_shape[1]+col_help+self.pred_shape[0]] = numpy.rot90(ys,-1)

        print("\tSaving figure...")
        pyplot.imshow(map,cmap='gray')
        pyplot.savefig(filename)
        pyplot.clf()
        print("[+] Done!")







