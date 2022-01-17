import tensorflow as tf

print(tf.__version__)

a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32, name='a')
b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2], name='b')
print(type(a))
print(a)
print(b)

c = tf.matmul(a, b)
print(c)
print(type(c))

# 默認圖
print(c.graph is tf.get_default_graph())

# 創建新的圖
graph1 = tf.Graph()
with graph1.as_default():
    e = tf.constant(5.0, name='e')
    print(e.graph is graph1)

