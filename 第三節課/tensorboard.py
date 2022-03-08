from unicodedata import name
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size
print(n_batch)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name='mean', tensor=mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x-input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y-input')

with tf.name_scope('layer'):
    with tf.name_scope('weight'):
        W = tf.Variable(initial_value=tf.zeros(shape=[784, 10]), name='w')
        variable_summaries(W)
    with tf.name_scope('bias'):
        b = tf.Variable(initial_value=tf.zeros(shape=[10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)
    # prediction = tf.matmul(x, W) + b

# loss = tf.reduce_mean(tf.square(y - prediction))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar(name='loss', tensor=loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss=loss)


init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, dimension=1), tf.argmax(prediction, dimension=1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    counter = 0
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run(fetches=[merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
            
            counter = counter + 1
            writer.add_summary(summary, counter)
        # writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter: ' + str(epoch) + ", Testing Accuracy: " + str(acc))

