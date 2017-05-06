import tensorflow as tf
import numpy
from model import MapModel
from data import BatchManager
import sys
import os

X_SHAPE = (3,10)
Y_SHAPE = (1,10)

learning_rate = 0.001
training_epochs = 50
batch_size = 64
total_batch = 1000

if __name__ == "__main__":
    if len(sys.argv) > 1:
        training_epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        PRED_SHAPE = int(sys.argv[2])
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])

    x = tf.placeholder("float", [None, numpy.prod(X_SHAPE)],name="x_placeholder")
    y = tf.placeholder("float", [None, numpy.prod(Y_SHAPE)],name="y_placeholder")

    data = BatchManager('image/map2.jpg',x_shape=X_SHAPE,y_shape=Y_SHAPE,pregen=10000)
    model = MapModel(x, y, [1024,512],
                     cube_shape=X_SHAPE,
                     pred_shape=Y_SHAPE)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.exists('checkpoints/checkpoint') and len(sys.argv)==1:
            print("[*] Restoring model")
            saver.restore(sess,'checkpoints/model.ckpt')
        else:
            sess.run(init)
            model.train(sess, data, training_epochs, total_batch)

            print("[*] Saving model....")
            saver.save(sess,'checkpoints/model.ckpt')
            model.export_diagnostics(sess, "diag")

        init = data.make_batch(batch_size=1)[0]\
                   .reshape(X_SHAPE)
        model.make_map(sess, init)
