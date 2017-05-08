import tensorflow as tf
import numpy
from model import MapModel
from data import BatchManager
import sys
import os

X_SHAPE = (5,10)
Y_SHAPE = (2,10)

learning_rate = 0.001
training_epochs = 50
out_each = 50
batch_size = 64
total_batch = 1000

if __name__ == "__main__":
    if len(sys.argv) > 1:
        training_epochs = int(sys.argv[1])
    if len(sys.argv) > 2:
        out_each = int(sys.argv[2])

    data = BatchManager('image/map2.jpg',x_shape=X_SHAPE,y_shape=Y_SHAPE,pregen=10000)

    x = tf.placeholder("float", [None, numpy.prod(X_SHAPE)],name="x_placeholder")
    y = tf.placeholder("float", [None, numpy.prod(Y_SHAPE)],name="y_placeholder")

    model = MapModel(x, y, [1024,512],
                     cube_shape=X_SHAPE,
                     pred_shape=Y_SHAPE)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.exists('checkpoints/checkpoint') and len(sys.argv)==1:
            print("[*] Restoring model")
            saver.restore(sess,'checkpoints/model.ckpt')
            init = data.make_batch(batch_size=1)[0] \
                .reshape(X_SHAPE)
            model.make_map(sess, init, filename="output/map.png")
        else:
            sess.run(init)
            init = data.make_batch(batch_size=1)[0] \
                .reshape(X_SHAPE)
            for i in range(0,training_epochs,out_each):
                model.make_map(sess, init, filename="output/map_{}.png".format(i))
                model.train(sess, data, out_each, total_batch,
                            epoch_offset=i,total_epochs=training_epochs)
                print("[*] Saving model....")
                saver.save(sess,'checkpoints/model.ckpt')
                # model.export_diagnostics(sess, "diag")
            model.make_map(sess, init, filename="output/map_{}.png".format(training_epochs))