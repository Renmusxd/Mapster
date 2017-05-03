import tensorflow as tf
from model import MapModel
from data import BatchManager
import numpy

DAT_SHAPE = 100
PRED_SHAPE = 10

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
total_batch = 1000

x = tf.placeholder("float", [None, DAT_SHAPE])
y = tf.placeholder("float", [None, PRED_SHAPE])

data = BatchManager('image/map1.jpg')
model = MapModel(x, y, DAT_SHAPE, [256,256], PRED_SHAPE)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    model.train(sess,data,training_epochs,total_batch)
    model.make_map(sess,data.next_batch(batch_size=1)[0])
