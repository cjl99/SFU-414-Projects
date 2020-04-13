import os

import tensorflow as tf
from skimage import color

from utils import *
from re_cnn import Model
import numpy as np
from skimage.io import imsave
from skimage.transform import resize
import cv2


def probability2img(probTensor, color_space='rgb'):
    eps = 1e-4
    output_dim = 256
    TEMPERATURE = 0.38
    batch_sz = tf.shape(probTensor)[0]
    CLASS_MAP_R = tf.constant(np.asarray([32 * i + 16 for i in range(8)] * 64), dtype=tf.float32)
    CLASS_MAP_G = tf.constant(np.asarray([32 * int(i / 8) + 16 for i in range(64)] * 8), dtype=tf.float32)
    CLASS_MAP_B = tf.constant(np.asarray([32 * int(i / 64) + 16 for i in range(512)]), dtype=tf.float32)

    # unnormalized = tf.exp((tf.log(probTensor)) / TEMPERATURE)
    # probabilities = unnormalized / tf.reduce_sum(unnormalized, axis=2, keep_dims=True)
    # probabilities = tf.nn.softmax(probabilities)
    probabilities = probTensor
    print(probabilities)
    print(CLASS_MAP_R)
    if color_space == 'rgb':
        out_img = tf.stack((tf.reduce_sum(CLASS_MAP_R * probabilities, axis=3),
                                tf.reduce_sum(CLASS_MAP_G * probabilities, axis=3),
                                tf.reduce_sum(CLASS_MAP_B * probabilities, axis=3)), axis=3)
        print(out_img)
        out_img = tf.reshape(out_img, shape=[batch_sz, output_dim, output_dim, 3])

    return out_img

common_params, dataset_params, net_params, solver_params = get_params('model.cfg')
img = cv2.imread('sketch.jpg')
img = cv2.resize(img,(256,256))
if len(img.shape) == 3:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = img[None, :, :, None]
data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

#data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
autocolor = Model(train=False,common_params=common_params, net_params=net_params)

conv8_313 = autocolor.conv_net(data_l)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './model.ckpt-0')
    conv8_313 = sess.run(conv8_313)

# img_rgb = decode(data_l, conv8_313, 2.63)
img_rgb = probability2img(conv8_313)
sess = tf.Session()
with sess.as_default():
    img_rgb = img_rgb.eval()

print(img_rgb.shape)
print(img_rgb[0].shape)
print(img_rgb[0])
imsave('color.jpg', img_rgb[0])