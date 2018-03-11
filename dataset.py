import numpy as np
from  pre_processing import build_data
import os
import tensorflow as tf


def _resize_function(image_decoded,lable):
    image_decoded.set_shape([None, None, None]) #amazing it works
    img = tf.image.resize_images(image_decoded,[416,416])
    return img,lable

def _process(id_str):
    id = id_str[:-4]
    id = int(id)
    return id

def get_id():
    id = os.listdir('data/lable')
    id = list(map(_process,id))
    return id

def get_dataset(epoch=50, batch_size=32):
    id_list = get_id()
    dataset = tf.data.Dataset.from_tensor_slices(id_list)
    #custom python procsssing func
    dataset = dataset.map(
        lambda id_list:tf.py_func(build_data,[id_list],[tf.uint8,tf.float32])
    )
    #tensorflow op
    dataset = dataset.map(_resize_function)
    #setting dataset
    dataset = dataset.shuffle(buffer_size=512)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    img, lable = iterator.get_next()
    return img, lable

def main():
    with tf.Session() as sess:
        img,lable = get_dataset()
        img_t,lable_t = sess.run((img,lable))
        print(img_t.shape,lable_t.shape)

if __name__ == '__main__':
    main()