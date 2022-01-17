

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32, name='a')
b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2], name='b')
c = tf.matmul(a, b, name='Mat_Mul')

# 默認圖
print(c.graph is tf.get_default_graph())

# 創建新的圖
graph1 = tf.Graph()
with graph1.as_default():
    e = tf.constant(5.0, name='e')
    print(e.graph is graph1)

sess = tf.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(log_device_placement=True))
result = sess.run(fetches=[c])
print(result)
print(type(result))

sess.close()


# with tf.Session() as sess:
#     pass

