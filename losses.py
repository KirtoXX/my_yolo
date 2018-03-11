import keras.backend as K
import tensorflow as tf
from keras import layers
import numpy as np

def custom_loss(y_true, y_pred):
    # shape of tnesor: None,13,13,5,8
    # order x,y,w,h,confidence,c
    xy_predict = y_pred[:,:,:,:,0:2]
    wh_predict = y_pred[:,:,:,:,2:4]
    confidence_predict = y_pred[:,:,:,:,4]
    c_predict = y_pred[:,:,:,:,5:]

    xy_true = y_true[:, :, :, :, 0:2]
    wh_true = y_true[:, :, :, :, 2:4]
    confidence_true = y_true[:, :, :, :, 4]
    c_true = y_true[:, :, :, :, 5:]

    #xy loss
    loss1 = tf.square((xy_true-xy_predict))
    loss1 = tf.reduce_sum(loss1,axis=-1)
    loss1 = tf.multiply(confidence_true,loss1)

    #wh loss
    wh_predict1 = tf.sqrt(wh_predict)
    wh_true1 = tf.sqrt(wh_true)
    loss2 = wh_predict1 - wh_true1
    loss2 = tf.square(loss2)
    loss2 = tf.reduce_sum(loss2,axis=-1)
    loss2 = tf.multiply(confidence_true,loss2)

    #confidence loss
    loss3 = tf.square(confidence_true-confidence_predict)

    #class loss
    loss4 = tf.square(c_predict-c_true)
    loss4 = tf.reduce_sum(loss4, axis=-1)
    loss4 = tf.multiply(confidence_true,loss4)

    #get total loss
    C1 = 2.
    C2 = 0.3
    total_loss = loss1 + loss2 + loss3 + C1*loss3 + C2*loss4
    total_loss = layers.Flatten()(total_loss)
    total_loss = tf.reduce_sum(total_loss,axis=-1)

    return total_loss

def main():
    l1 = np.zeros([1,13,13,5,8])
    l2 = np.zeros([1,13,13,5,8])
    y1 = tf.placeholder(dtype=np.float32,shape=[None,13,13,5,8])
    y2 = tf.placeholder(dtype=np.float32,shape=[None, 13, 13,5,8])
    loss = custom_loss(y1,y2)
    with tf.Session() as sess:
        l = sess.run(loss,feed_dict={y1:l1,y2:l2})
        print(l)

if __name__ == '__main__':
    main()


