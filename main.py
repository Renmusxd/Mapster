import tensorflow as tf
from model import MapModel
from fake_data import BatchManager

DAT_SHAPE = 100
PRED_SHAPE = 10

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
total_batch = 1000

x = tf.placeholder("float", [None, DAT_SHAPE])
y = tf.placeholder("float", [None, PRED_SHAPE])

data = BatchManager('some/data/path')
model = MapModel(x, y, DAT_SHAPE, [256,256], PRED_SHAPE)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(total_batch):
            batch_x, batch_y = data.next_batch()
            _, c = sess.run([model.optimizer, model.cost],
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("Epoch: {}\tcost: {:.9f}".format(epoch,avg_cost))
