from moviepy.editor import VideoFileClip
import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

class Video():

    def __init__(self, pathToVideo, targetPath, sess, logits, keep_prob, image_pl, image_shape):
        self.sess = sess
        self.logits = logits
        self.keep_prob = keep_prob
        self.image_pl = image_pl
        self.image_shape = image_shape
        self.videoClip = VideoFileClip(pathToVideo)
        self.targetPath = targetPath

    def process_image_video(self, image):
        image = scipy.misc.imresize(image, self.image_shape)

        im_softmax = self.sess.run(
            [tf.nn.softmax(self.logits)],
            {self.keep_prob: 1.0, self.image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        return np.array(street_im)

    def process_video(self):
        seg_clip = self.videoClip.fl_image(self.process_image_video)
        seg_clip.write_videofile(self.targetPath, audio=False)
