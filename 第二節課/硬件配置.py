import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.device('/gpu:0'):
    a = tf.Variable(initial_value=[2, 3, 4], dtype=tf.int32, name='a')
    b = tf.constant(value=2, name='c')
    c = tf.add(a, b, name='c')


init_op = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init_op)
    print(sess.run(c))




