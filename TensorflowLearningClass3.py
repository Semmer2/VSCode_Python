import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":
    hello = tf.constant('Hello World')
    with tf.Session() as sess:
        print(sess.run(hello))
        sess.close()

#这是一个测试更改，看github上是否有正确的修改