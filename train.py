from darknet19 import inference
from keras import layers
from keras.models import Model
from loss import custom_loss
from keras.optimizers import RMSprop
import keras.backend as k
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import parse_annotation
from utils import data_gen
from cfg import *


image_dir = 'F:/object-detection-crowdai/JPEGImages/'
ann_dir = 'F:/object-detection-crowdai/Annotations/'

def main():
    input_image = layers.Input(shape=(416,416, 3))
    true_boxes = layers.Input(shape=(1, 1, 1,50, 4))
    logit = inference(input_image)
    model = Model(input_image,logit)
    opt = RMSprop()
    model.compile(loss=custom_loss,optimizer=opt)
    model.summary()

    sess = k.get_session()
    writer = tf.summary.FileWriter('log/', sess.graph)

    anns, labels = parse_annotation(ann_dir)


    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

    model.fit_generator(data_gen(anns, BATCH_SIZE),
                            int(len(anns) / BATCH_SIZE),
                            epochs=100,
                            verbose=1,
                            callbacks=[early_stop],
                            max_q_size=3)

if __name__ == '__main__':
    main()


