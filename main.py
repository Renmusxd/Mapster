import tensorflow as tf
from model import MapModel
from data import BatchManager
import numpy

PRED_SHAPE = 20

learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1
total_batch = 1000

x = tf.placeholder("float", [None, PRED_SHAPE*PRED_SHAPE])
y = tf.placeholder("float", [None, PRED_SHAPE])

data = BatchManager('image/map2.jpg')
model = MapModel(x, y, [512,256], cube_shape=PRED_SHAPE)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    model.train(sess,data,training_epochs,total_batch)
    model.make_map(sess,data.next_batch(pred_size=PRED_SHAPE,batch_size=1)[0].reshape((PRED_SHAPE,PRED_SHAPE)))
