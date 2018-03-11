from keras.applications.densenet import DenseNet
from keras.layers import Input
from keras.layers import Conv2D,BatchNormalization,Activation,Reshape,Lambda
from keras import layers
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np

classs = 3
anchor_box = 5
down_sample = 32
input_shape = 416
output_shape = int(input_shape/down_sample)

def _customlayer(output):
    # 5-dimision
    # x,y (0,1)
    pos = K.sigmoid(output[:, :, :, :, 0:2])
    # w,h >0
    shape = K.exp(output[:, :, :, :, 2:4])
    # confidence (0,1)
    confidence = K.expand_dims(output[:, :, :, :, 4], axis=-1)
    confidence = K.sigmoid(confidence)
    # class:
    c = K.softmax(output[:, :, :, :, 5:])
    result = K.concatenate([pos, shape, confidence, c], axis=-1)
    #list(x,y,w,h,confidence,c)
    return result

def _customlayer_output_shape(input_shape):
    return input_shape

myCustomLayer = Lambda(_customlayer, output_shape=_customlayer_output_shape)

def inference(inputs):
    model = DenseNet(blocks=[6, 12, 24, 16],
                    include_top=False,
                    weights=None,
                    input_shape=(input_shape,input_shape,3))
    model.load_weights(filepath='pre_weight/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
    conv0 = model(inputs)

    with tf.name_scope('conv1') as scope:
        conv1 = Conv2D(kernel_size=(3,3),
                       filters=256,
                       use_bias=False,
                       name=scope+'c1',padding='same')(conv0)
        conv1 = BatchNormalization(name=scope+'bn')(conv1)
        conv1 = Activation('relu')(conv1)

    with tf.name_scope('output') as scope:
        conv2 = Conv2D(kernel_size=(1,1),
                       filters=anchor_box*(5+classs),
                       use_bias=False,
                       name=scope+'c1',padding='same')(conv1)

        output = Reshape([output_shape,output_shape,anchor_box,5+classs])(conv2)
        #print(output.shape)
        result = myCustomLayer(output)

    return result

def main():
    inputs = Input(shape=(input_shape,input_shape,3))
    logit = inference(inputs)
    model = Model(inputs,logit)
    model.summary()


if __name__ == '__main__':
    main()