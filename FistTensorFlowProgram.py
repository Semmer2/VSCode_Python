import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if __name__ == '__main__':
    '''a = tf.constant(2,name='A')
    b = tf.constant(3,name='B')
    x = tf.add(a,b,name='ADD')

    writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())
    with tf.Session() as sess:
        print(sess.run(x))
        sess.close()
    writer.close()'''
    '''
    a = tf.constant([[2,3],[4,5]],shape=[2,2],verify_shape=True)
    with tf.Session() as sess:
        print(a.eval())'''


    '''
    a = tf.constant([1,2])
    b = tf.constant([3,4])

    Add1 = tf.add(a,b)
    Add2 = tf.add_n([a,b,b])
    Mul1 = tf.multiply(a,b)
    sess = tf.Session()
    print("Add 1: ",sess.run(Add1))
    print("Add 2: ",sess.run(Add2))
    print("Mul 1: ",sess.run(Mul1))'''

    '''
    VIni = tf.Variable(10)
    AOp = VIni.assign(100)
    with tf.Session() as sess:
        sess.run(VIni.initializer)
        print(VIni.eval())
        sess.run(AOp)
        print("After assigned:",VIni.eval())
    '''
    '''W = tf.Variable(20)
    sess2 = tf.Session()
    sess2.run(W.initializer)

    with tf.Session() as sess1:
        sess1.run(W.initializer)
        print(sess1.run(W.assign_add(10)))
        print(sess2.run(W.assign_sub(5)))
        print(W.eval())
        print(sess1.run(W.assign_add(20)))
        print(sess2.run(W.assign_sub(10)))

    sess2.close()'''
    

    '''
    sess = tf.InteractiveSession()
    W1 = tf.Variable(3)
    W2 = tf.Variable(2*W1)
    W3 = 3*W1.initialized_value()
    sess.run(W2.initializer)
    print(W2.eval())
    print(W3.eval())
    '''

    a = tf.placeholder(tf.float32,shape=[3])
    b = tf.constant([1,2,3],tf.float32)
    c = a + b
    with tf.Session() as sess:
        print(sess.run(c,{a:[1,2,3]}))
    