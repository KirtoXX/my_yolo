from __future__ import print_function
from keras.models import Model
from keras import layers
import losses
import yolo
import dataset
import tensorflow as tf
from keras import callbacks
import numpy as np

classs = 3
anchor_box = 5
down_sample = 32
input_shape = 416
output_shape = int(input_shape/down_sample)

flags = tf.app.flags
flags.DEFINE_integer('batch_size',1,'batch_size')
flags.DEFINE_string('weight','false','weights')
flags.DEFINE_integer('epoch',50,'epoch')
flags.DEFINE_float('lr',1e-5,'lr')
FLAGS = flags.FLAGS

def main(_):
    batch_size = FLAGS.batch_size
    epoch = FLAGS.epoch
    weight = FLAGS.weight
    lr = FLAGS.lr
    #input = Input(shape=(416,416,3))
    m = len(dataset.get_id())
    steps_per_epoch = int(np.ceil(m/ (float(batch_size))))

    ckpt = callbacks.ModelCheckpoint(filepath='weights/last_weight.h5',
                                     period=1,
                                     save_weights_only=True)


    img,lable = dataset.get_dataset(epoch=epoch,
                                    batch_size=batch_size)

    inputs = layers.Input(tensor=img)
    logit = yolo.inference(inputs)
    model = Model(inputs,logit)

    model.compile(optimizer='Rmsprop',
                  loss=losses.custom_loss,
                  target_tensors=[lable])

    if weight != 'false':
        model.load_weights(weight)

    model.fit(epochs=epoch,
              steps_per_epoch=steps_per_epoch,
              verbose=1,
              callbacks=[ckpt])

if __name__ == '__main__':
    tf.app.run()