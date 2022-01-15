import tensorflow as tf

m1 = tf.constant([[3, 3]])
print(m1)
m2 = tf.constant([[2], [3]])
print(m2)

product = tf.matmul(m1, m2)
print(product)

# sess = tf.Session()
#
# result = sess.run(product)
# print(result)
#
# sess.close()

with tf.Session() as sess:
    result = sess.run(product)
    print(result)


