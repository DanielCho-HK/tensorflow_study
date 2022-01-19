import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size
print(n_batch)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(dtype=tf.float32)

W1 = tf.Variable(initial_value=tf.truncated_normal(shape=[784, 2000], stddev=0.1))
b1 = tf.Variable(initial_value=tf.zeros(shape=[2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(initial_value=tf.truncated_normal(shape=[2000, 2000], stddev=0.1))
b2 = tf.Variable(initial_value=tf.zeros(shape=[2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(initial_value=tf.truncated_normal(shape=[2000, 1000], stddev=0.1))
b3 = tf.Variable(initial_value=tf.zeros(shape=[1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(initial_value=tf.truncated_normal(shape=[1000, 10], stddev=0.1))
b4 = tf.Variable(initial_value=tf.zeros(shape=[10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# loss = tf.reduce_mean(tf.square(y - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss=loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, dimension=1), tf.argmax(prediction, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print('Iter: ' + str(epoch) + ", Testing Accuracy: " + str(test_acc) + ", Training Accuracy: " + str(train_acc))




















