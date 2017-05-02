import tensorflow as tf

class MapModel:

    def __init__(self,input_layer,output_placeholder,
                 input_size,shape,output_size,
                 learning_rate=0.001):
        self.input_layer = input_layer
        self.shape = [input_size] + list(shape) + [output_size]

        last_size = self.shape[0]
        weights = []
        biases = []
        layers = []
        layer = self.input_layer
        for s in list(self.shape):
            w = tf.Variable(tf.random_normal([last_size, s]))
            b = tf.Variable(tf.random_normal([s]))
            layer = tf.nn.relu(tf.add(tf.matmul(layer,w),b))

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

    def make_single_model(self):
        pass

    def make_training_model(self):
        pass