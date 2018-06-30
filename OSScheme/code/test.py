# /usr/bin/env python
# coding=utf-8

import tensorflow as tf

# new_saver = tf.train.import_meta_graph("../model/model.ckpt-0.meta")
#
# all_vars = tf.trainable_variables()
# for v in all_vars:
#     print v.name
#     #print v.name,v.eval(self.sess) # v 都还未初始化，不能求值
# # 加载模型 参数变量 的 值
# with tf.Session() as sess:
#     new_saver.restore(sess, tf.train.latest_checkpoint('../model/'))
#     self.input_x = graph.get_tensor_by_name('input_x:0')
#     all_vars = tf.trainable_variables()
#     for v in all_vars:
#         print v.name,v.eval(sess)


