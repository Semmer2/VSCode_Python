import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":
    hello = tf.constant('Hello World')
    with tf.Session() as sess:
        print(sess.run(hello))
        sess.close()