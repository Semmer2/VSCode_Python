import tensorflow as tf
import numpy as np

A = tf.constant(5,shape=[5])
B = tf.constant(4,shape=[4,5])

C = tf.add(B,A)
with tf.Session() as sess:
    print(sess.run(C))
    