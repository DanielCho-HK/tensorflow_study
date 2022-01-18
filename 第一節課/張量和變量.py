import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.Variable(initial_value=3.0, dtype=tf.float32)
b = tf.constant(value=2, dtype=tf.float32)
c = tf.add(a, b)

init = tf.global_variables_initializer()


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init)
    print(sess.run(c))


w1 = tf.Variable(initial_value=tf.random_normal(shape=[10, ], stddev=0.5, seed=10, dtype=tf.float32, name='w1'))
a = tf.constant(value=2, dtype=tf.float32)
w2 = tf.Variable(initial_value=w1.initialized_value() * a, dtype=tf.float32, name='w2')

init_op = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init_op)
    result = sess.run(fetches=[w1, w2])
    print(result[0])
    print(result[1])





