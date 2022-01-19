import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.placeholder(dtype=tf.int32)
result = tf.Variable(initial_value=1, dtype=tf.int32, name='result')

init_op = tf.global_variables_initializer()
assign_op = tf.assign(ref=result, value=tf.multiply(result, a))

#控制依賴
with tf.control_dependencies([assign_op]):
    result = tf.Print(result, data=[result, result.read_value()], message='result:')

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(1, 7):
        r = sess.run(result, feed_dict={a: i})
    print(r)



