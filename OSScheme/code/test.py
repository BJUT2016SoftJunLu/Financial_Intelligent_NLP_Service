import tensorflow as tf
import numpy as np


data = [np.array([0.]), np.array([-0.10381421, 0.09874513])]
print(np.array(data).shape)
value = tf.constant(np.array(data),dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(value))