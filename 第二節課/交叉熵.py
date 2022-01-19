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

W = tf.Variable(initial_value=tf.zeros(shape=[784, 10]))
b = tf.Variable(initial_value=tf.zeros(shape=[10]))
# prediction = tf.nn.softmax(tf.matmul(x, W) + b)
prediction = tf.matmul(x, W) + b

# loss = tf.reduce_mean(tf.square(y - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss=loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, dimension=1), tf.argmax(prediction, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter: ' + str(epoch) + ", Testing Accuracy: " + str(acc))




















