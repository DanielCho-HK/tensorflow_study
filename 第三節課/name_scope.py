import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    with tf.variable_scope(name_or_scope='foo', initializer=tf.constant_initializer(3.0)) as foo:
        v = tf.get_variable(name='v', shape=[1])
        w = tf.get_variable(name='w', shape=[1], initializer=tf.constant_initializer(2.0))

        with tf.variable_scope(name_or_scope='bar'):
            l = tf.get_variable(name='l', shape=[1])

            with tf.variable_scope(name_or_scope=foo):
                h = tf.get_variable(name='h', shape=[1])
                g = v + w + l + h

    sess.run(tf.global_variables_initializer())
    print('\n{}----{}'.format(v.name, v.eval()))
    print('\n{}----{}'.format(w.name, w.eval()))
    print('\n{}----{}'.format(l.name, l.eval()))
    print('\n{}----{}'.format(h.name, h.eval()))
    print('\n{}----{}'.format(g.name, g.eval()))
