import tensorflow as tf 
import numpy as np
import pickle as pk 
from PIL import Image

ciefar_10_folder = "D:\Projects\Python_Projects\Database\cifar-10-batches-py"
ciefar_10_meta = "batches.meta"
ciefar_10_batch = "data_batch_"
ciefar_10_test = "test_batch"

learning_rate = 0.00001
training_epochs = 5
batch_size = 500

def load_CIEFAR_batch(filename):
    #load single batch file
    with open(filename,'rb') as file:
        datadict = pk.load(file,encoding = "bytes")
        RawImg = datadict[b'data']
        ImgLabel = datadict[b'labels']
        RawImg = RawImg.reshape([10000,3,32,32])
        ImgLabel = np.array(ImgLabel)
        return RawImg,ImgLabel


def load_CIRFAR_label(filename):
    with open(filename,'rb') as file:
        #lines = [tmp for tmp in file.readlines()]
        labeldict = pk.load(file,encoding = "bytes")
        return labeldict
    
def get_str_label_from_byte(byte_labels):
    #所有提取出来的labe会有相关 b'airplane' byte的注释，手动去除
    for i in range(len(byte_labels)):
        strlabel = str(byte_labels[i])
        byte_labels[i] = strlabel[2:-1]
    return byte_labels

def extract_byte_ciefar_10_to_imgs():
    meta_file_name = ciefar_10_folder + "\\" + ciefar_10_meta
    labels = load_CIRFAR_label(meta_file_name)
    labels = labels[b'label_names']
    labels = get_str_label_from_byte(labels)
    
    for i in range(5):
        batch_file_1 = ciefar_10_folder + "\\" + ciefar_10_batch + str(i+1)
        imgs,imglabels = load_CIEFAR_batch(batch_file_1)
        for j in range(10000):
            img = imgs[j]
            imgR = img[0]
            imgG = img[1]
            imgB = img[2]
            newImg = Image.merge("RGB",(Image.fromarray(imgR),Image.fromarray(imgG),Image.fromarray(imgB)))
            img_save_folder = ciefar_10_folder+"\\"+"extract_pic\\"+"train"+"\\"+labels[imglabels[j]]+"\\"+"train_batch_"+str(i+1)+"_num_"+str(j+1)+".png"
            newImg.save(img_save_folder,"PNG")
        print("No."+str(i+1)+" finished")
    

    batch_file_test = ciefar_10_folder + "\\" + ciefar_10_test
    testimgs,testimglabels = load_CIEFAR_batch(batch_file_test)
    for j in range(10000):
        testimg = testimgs[j]
        imgR = testimg[0]
        imgG = testimg[1]
        imgB = testimg[2]
        newImg = Image.merge("RGB",(Image.fromarray(imgR),Image.fromarray(imgG),Image.fromarray(imgB)))
        img_save_folder = ciefar_10_folder+"\\"+"extract_pic\\"+"test"+"\\"+labels[imglabels[j]]+"\\"+"train_batch_"+str(i+1)+"_num_"+str(j+1)+".png"
        newImg.save(img_save_folder,"PNG")
        if (j+1)%1000 == 0 :
            print("No."+str(j+1)+" finished")
    
    print("All extract finished")

def label_to_y_tag(imglabels):
    label_tag = np.zeros([10000,10])
    for i in range(10000):
        label_tag[i][imglabels[i]] = 1
    return label_tag

    


if __name__ == "__main__":
    
    #extract_byte_ciefar_10_to_imgs()
    X = tf.placeholder(tf.float32,[None,3072])
    Y = tf.placeholder(tf.float32,[None,10])
    
    w = tf.Variable(tf.zeros([3072,10]))
    b = tf.Variable(tf.zeros([10]))
    #w = tf.get_variable(name="weight",shape=(3072,10),initializer=tf.random_normal_initializer())
    #b = tf.get_variable(name="bias",shape=(1,10),initializer=tf.zeros_initializer())
    
    Y_pred = tf.matmul(X,w)+b
    Y_pred = tf.nn.softmax(Y_pred)
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_pred),axis=1))
    
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    meta_file_name = ciefar_10_folder + "\\" + ciefar_10_meta
    labels = load_CIRFAR_label(meta_file_name)
    labels = labels[b'label_names']
    labels = get_str_label_from_byte(labels)
    #print(labels)
    
    batch_file_test = ciefar_10_folder + "\\" + ciefar_10_test
    testimgs,testimglabels = load_CIEFAR_batch(batch_file_test)
    testimgs = testimgs.reshape([10000,3072])
    testimglabels = label_to_y_tag(testimglabels)
    
    #imgs = np.zeros([5,10000,3,32,32])
    #imglabels = np.zeros([5,10000])
    total_loss = 0
    
    with tf.Session() as sess:
        sess.run(init)
        for n in range(training_epochs+1):
            for i in range(5):
                if(n<training_epochs):
                    batch_file_folder = ciefar_10_folder + "\\" + ciefar_10_batch + str(i+1)
                else:
                    batch_file_folder = ciefar_10_folder + "\\" + ciefar_10_test
                    print(batch_file_folder)
                imgs,imglabels = load_CIEFAR_batch(batch_file_folder)
                imgs = imgs.reshape([10000,3072])
                imglabels = label_to_y_tag(imglabels)
                imgs = np.asarray(imgs,dtype=np.float32)
                total_loss = 0
                n_batch = int(10000/batch_size)
                
                for j in range(n_batch):
                    if (j+1)*batch_size>len(imgs):
                        print("batch out of range")
                    else :
                        start = j*batch_size
                        end = j*batch_size+batch_size
                        X_batch = imgs[start:end]
                        Y_batch = imglabels[start:end]
                    _,batch_loss = sess.run([optimizer,cost],{X:X_batch,Y:Y_batch})
                    total_loss += batch_loss/n_batch
                    
                #print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batch))
            
            print('Average loss epoch {0}: {1}'.format(n+1, total_loss/n_batch))
        
        Y_pred.eval({X:X_batch,Y:Y_batch})
        #print("load No."+str(i+1)+" finished")
    
    
    
    
    

    
    