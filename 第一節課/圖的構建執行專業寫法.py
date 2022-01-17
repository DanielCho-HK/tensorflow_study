import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


a = tf.constant('10', tf.string, name='a_const')
b = tf.string_to_number(a, out_type=tf.float64, name='str_2_double')
c = tf.to_double(5.0, name='to_double')
d = tf.add(b, c, name='add')

gpu_options = tf.GPUOptions()
gpu_options.per_process_gpu_memory_fraction = 0.5
gpu_options.allow_growth = True

optimizer = tf.OptimizerOptions(
    do_common_subexpression_elimination=True,
    do_constant_folding=True,
    opt_level=0
)

graph_options = tf.GraphOptions(optimizer_options=optimizer)

config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                              graph_options=graph_options, gpu_options=gpu_options)

with tf.Session(config=config_proto) as sess:
    print(d.eval())
    print(sess.run(d))

