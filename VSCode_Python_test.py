import tensorflow as tf

if __name__=='__main__':
    hello=tf.constant('Hello tensorflow')
    print(hello.graph)
    print(tf.get_default_graph())
    sess=tf.Session()

    gp=tf.Graph()
    print('gp:',gp)
    with gp.as_default():
        t=tf.constant('T')
        print(t.graph)

    #print(sess.run(hello))
    #sess.close()
