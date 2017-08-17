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


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def calculate_iou(sess, logits, keep_prob, image_pl, image_batch, label_batch, image_shape):
    """
    Calculate IoU for a batch of images and labels
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param image_batch: Batch of images
    :param label_batch: Batch of labels
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    """ From @jendrik (Slack) """
    """
    crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
    self.crossEntropy = tf.reduce_mean(crossEntropy)
    loss = tf.reduce_mean(tf.multiply(crossEntropy, self.weights))
    tf.add_to_collection(‘my_losses’, tf.multiply(.1,loss))

    imu, self.imuOp = tf.metrics.mean_iou(self.labels, tf.argmax(logits, axis = 2), numberOfClasses,
    self.weights, name = ‘meanIMU’)
    with tf.control_dependencies([self.imuOp]):
           self.imu = tf.subtract(tf.constant(1.), imu)
           tf.add_to_collection(‘my_losses’, self.imu)
    self.loss = tf.reduce_sum(tf.stack(tf.get_collection(‘my_losses’)))
    self.trainStep = tf.train.AdamOptimizer(5e4).minimize(self.loss)
    """

    # For testing
    if logits is None:
        return 0.0

    sum_iou = 0.0
    batch_size = len(label_batch)

    # Prepare Ops
    softmax_logits = tf.reshape(tf.squeeze(tf.nn.softmax(logits)), [-1])
    indices_to_gather = tf.constant(np.asarray(range(1, batch_size*image_shape[0]*image_shape[1]*2, 2)))
    softmax_part = tf.gather(softmax_logits, indices_to_gather, name="gather_sm_part")
    softmax_reshaped = tf.reshape(softmax_part, (batch_size, image_shape[0], image_shape[1]), name="reshape_softmax")
    gt05 = tf.greater(softmax_reshaped, 0.5)
    tf_segmentation_ = tf.reshape(gt05, (batch_size, image_shape[0], image_shape[1]), name="reshape_gt05_to_seg")
    label_pl = tf.placeholder(tf.float32, (batch_size, image_shape[0], image_shape[1]), name="label_pl")
    iou_, iou_op_ = tf.metrics.mean_iou(label_pl, tf_segmentation_, 2)

    # Run
    label_batch_formatted = label_batch[:,:,:,1]
    sess.run(tf.local_variables_initializer())
    sess.run(iou_op_, {keep_prob: 1.0, image_pl: image_batch, label_pl: label_batch_formatted})
    return sess.run(iou_, {keep_prob: 1.0, image_pl: image_batch, label_pl: label_batch_formatted})

def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
