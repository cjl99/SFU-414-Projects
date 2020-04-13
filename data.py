from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
if float(tf.__version__[0:3]) >= 2:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
import threading

from PIL import Image
import os
from PIL import ImageFilter


class Dataset(object):

    def __init__(self, common_params, dataset_params):
        if common_params:
            self.image_size = int(common_params['image_size'])
            self.batch_size = int(common_params['batch_size'])

        if dataset_params:
            self.data_path = str(dataset_params['path'])
            # self.thread_num = int(int(dataset_params['thread_num']) / 2)
            # self.thread_num2 = int(int(dataset_params['thread_num']) / 2)

        # self.batches_original = list()
        # self.batches_processed = list()
        # self.oneBatch = list()
        self.counter = 0

    def processed_color(self, img):
        # 图像组成：红绿蓝  （RGB）三原色组成    亮度（255,255,255）
        # image = "./images/10.png"
        # img_all = "素描" + image
        new = Image.new("L", img.size, 255)
        width, height = img.size
        img = img.convert("L")

        # 定义画笔的大小
        Pen_size = 2
        # 色差扩散器
        Color_Diff = 6
        for i in range(Pen_size + 1, width - Pen_size - 1):
            for j in range(Pen_size + 1, height - Pen_size - 1):
                # 原始的颜色
                originalColor = 255
                lcolor = sum([img.getpixel((i - r, j)) for r in range(Pen_size)]) // Pen_size
                rcolor = sum([img.getpixel((i + r, j)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(lcolor - rcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)

                ucolor = sum([img.getpixel((i, j - r)) for r in range(Pen_size)]) // Pen_size
                dcolor = sum([img.getpixel((i, j + r)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(ucolor - dcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)

                acolor = sum([img.getpixel((i - r, j - r)) for r in range(Pen_size)]) // Pen_size
                bcolor = sum([img.getpixel((i + r, j + r)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(acolor - bcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)

                qcolor = sum([img.getpixel((i + r, j - r)) for r in range(Pen_size)]) // Pen_size
                wcolor = sum([img.getpixel((i - r, j + r)) for r in range(Pen_size)]) // Pen_size

                # 通道----颜料
                if abs(qcolor - wcolor) > Color_Diff:
                    originalColor -= (255 - img.getpixel((i, j))) // 4
                    new.putpixel((i, j), originalColor)

        # new.save(img_all)
        new.resize((self.image_size, self.image_size))
        # new.save("sketch.jpg")
        return new

    def convert_original(self, img):
        classes_num = 8 * 8 * 8
        # print(img.shape)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        R = (R / 32).astype(int)
        G = (G / 32).astype(int)
        B = (B / 32).astype(int)
        # print("R=",R.shape)
        prob = np.zeros([self.image_size, self.image_size, classes_num])
        # print("prob=",prob.shape)
        classes = 4 * B + 2 * G + R
        for i in range(classes.shape[0]):
            for j in range(classes.shape[1]):
                prob[i][j][classes[i][j]] = 1
        return prob

    def generate_batches(self):
        # batch_xs = image_space = [batch_size, height, width, 3]
        # batch_ys = prob
        self.counter += 1
        image_path = './images/' + str(self.counter) + ".png"
        img = Image.open(image_path)
        img.resize((self.image_size, self.image_size))
        
        images = self.processed_color(img)
        images = np.resize(images, [1, self.image_size, self.image_size, 1])
        images = np.array(images)
        img = np.array(img)
        probs = self.convert_original(img)
        probs = np.resize(probs, [1, self.image_size, self.image_size, 512])
        while self.counter % (self.batch_size) != 0:
            self.counter += 1
            image_path = './images/' + str(self.counter) + ".png"
            img = Image.open(image_path)
            img.resize((self.image_size, self.image_size))
            image = self.processed_color(img)
            image = np.array(image)
            image = np.resize(image, [1, self.image_size, self.image_size, 1])
            img = np.array(img)
            prob = self.convert_original(img)
            prob = np.resize(prob, [1, self.image_size, self.image_size, 512])
            # print("prob shape:", prob.shape)
            # print("probs shape:", probs.shape)
            # print("image shape:", image.shape)
            # print("images shape:", images.shape)
            images = np.concatenate((images, image), axis=0)
            probs = np.concatenate((probs, prob), axis=0)
        return images, probs