import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def my_fun(x):
    w1 = tf.Variable(initial_value=tf.random_normal(shape=[2]))[0]
    b1 = tf.Variable(tf.random_normal(shape=[1]))[0]
    r1 = w1 * x + b1

    w2 = tf.Variable(initial_value=tf.random_normal(shape=[2]))[0]
    b2 = tf.Variable(tf.random_normal(shape=[1]))[0]
    r2 = w2 * r1 + b2

    return w1, b1, r1, w2, b2, r2

def my_fun2(x):
    w = tf.get_variable(name='w', shape=[2], initializer=tf.random_normal_initializer())[0]
    b = tf.get_variable(name='b', shape=[1], initializer=tf.random_normal_initializer())[0]
    r = w * x + b
    return w, b, r

def fun2(x):
    with tf.variable_scope(name_or_scope='op1', reuse=tf.AUTO_REUSE):
        r1 = my_fun2(x)
    with tf.variable_scope(name_or_scope='op2', reuse=tf.AUTO_REUSE):
        r2 = my_fun2(r1[2])
    return r1, r2


x = tf.constant(value=3, name='x', dtype=tf.float32)
# result = my_fun(x)
result = fun2(x)

x2 = tf.constant(value=4, name='x2', dtype=tf.float32)
result2 = fun2(x2)

init_op = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init_op)
    print(sess.run(fetches=[result, result2]))








