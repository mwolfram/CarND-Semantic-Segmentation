import os.path
import tensorflow as tf
import helper
import glob
import warnings
import sys
import os
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load vgg model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get required tensors
    vgg_input_tensor = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    """ From Class """

    """ 1x1 convolution
    The correct use is tf.layers.conv2d(x, num_outputs, 1, 1, weights_initializer=custom_init).
    num_outputs defines the number of output channels or kernels
    The third argument is the kernel size, which is 1.
    The fourth argument is the stride, we set this to 1.
    We use the custom initializer so the weights in the dense and convolutional layers are identical.
    This results in the a matrix multiplication operation that preserves spatial information.
    """

    """ deconvolution
    One possible answer is using tf.layers.conv2d_transpose(x, 3, (2, 2), (2, 2)) to upsample.
    The second argument 3 is the number of kernels/output channels.
    The third argument is the kernel size, (2, 2). Note that the kernel size could also be (1, 1) and the output shape would be the same. However, if it were changed to (3, 3) note the shape would be (9, 9), at least with 'VALID' padding.
    The fourth argument, the number of strides, is how we get from a height and width from (4, 4) to (8, 8). If this were a regular convolution the output height and width would be (2, 2).
    Now that you've learned how to use transposed convolution, let's learn about the third technique in FCNs.
    """

    """ weight initializer
        def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
            return tf.random_normal(shape, dtype=dtype, seed=seed)
    """

    print("vgg3_shape: " + str(vgg_layer3_out.get_shape())) # (?, 56, 56, 256)
    print("vgg4_shape: " + str(vgg_layer4_out.get_shape())) # (?, 28, 28, 512) (?, 14, 14, 512) (?, 7, 7, 512)
    print("vgg7_shape: " + str(vgg_layer7_out.get_shape())) # (?, 1, 1, 4096)

    vgg_layer3_num_outputs = vgg_layer3_out.get_shape()[3]
    vgg_layer4_num_outputs = vgg_layer4_out.get_shape()[3]
    vgg_layer7_num_outputs = vgg_layer7_out.get_shape()[3]

    # TODO weights are never initialized (or I don't know how)

    # 1x1 convolution
    conv_1x1_layer = tf.layers.conv2d(vgg_layer7_out,
                                      vgg_layer7_num_outputs,
                                      kernel_size=1,
                                      strides=1,
                                      name="conv_1x1_layer")

    print("conv_1x1_layer_shape: " + str(conv_1x1_layer.get_shape()))

    # first deconvolution using conv2d_transpose
    conv_transposed_layer_1 = tf.layers.conv2d_transpose(conv_1x1_layer,
                                                         vgg_layer4_num_outputs,
                                                         kernel_size=(4, 4),
                                                         strides=(2, 2),
                                                         padding="same",
                                                         name="conv_transposed_layer_1")

    print("conv_transposed_layer_1_shape: " + str(conv_transposed_layer_1.get_shape())) # supposed to be (?, ?, ?, 512)

     # skip layer
    skip_layer_1 = tf.add(conv_transposed_layer_1, vgg_layer4_out, name='skip_layer_1')

    # second deconvolution
    conv_transposed_layer_2 = tf.layers.conv2d_transpose(skip_layer_1,
                                                         vgg_layer3_num_outputs,
                                                         kernel_size=(4, 4),
                                                         strides=(2, 2),
                                                         padding="same",
                                                         name='conv_transposed_layer_2')

    print("conv_transposed_layer_2_shape: " + str(conv_transposed_layer_2.get_shape())) # supposed to be (?, ?, ?, 256)

    # skip layer
    skip_layer_2 = tf.add(conv_transposed_layer_2, vgg_layer3_out, name='skip_layer_2')

    # third deconvolution
    conv_transposed_layer_3 = tf.layers.conv2d_transpose(skip_layer_2,
                                                         num_classes,
                                                         kernel_size=(16, 16),
                                                         strides=(8, 8),
                                                         padding="same",
                                                         name='conv_transposed_layer_3')

    print("conv_transposed_layer_3_shape: " + str(conv_transposed_layer_3.get_shape())) # supposed to be (?, ?, ?, ?)

    return conv_transposed_layer_3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate, can also be a float
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # IOU: built-in
    """ and this does not work at all
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    mean_iou, update_op = tf.metrics.mean_iou(correct_label, logits, num_classes)
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(tf.reduce_mean(mean_iou))
    """

    # IOU from: http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    """ this works basically, but the results don't look good at all """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    intersection = tf.reduce_sum(tf.multiply(logits, correct_label))
    union = tf.reduce_sum(tf.subtract(tf.add(logits, correct_label), tf.multiply(logits, correct_label)))
    #loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.divide(intersection, union))
    loss = tf.divide(intersection, union)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss)

    # Pixel-wise cross-entropy-loss
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    """

    return logits, train_op, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param vgg_keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate, can also be a float
    """

    """ measure accuracy:
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    """

    print("Start training")

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('tensorboard/train', sess.graph)

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        print("Epoch", str(epoch), "|", end="")
        sys.stdout.flush()
        i = 0
        for sample_batch, label_batch in get_batches_fn(batch_size):
            # TODO
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: sample_batch, correct_label: label_batch, keep_prob: 0.5})
            #train_writer.add_summary(summary, i)
            print("=", end="")
            sys.stdout.flush()
            i = i + 1
        print ("| Loss: ", loss)

tests.test_train_nn(train_nn)


def run():

    tests_only = os.getenv("TESTS_ONLY", False)
    if tests_only:
        print("TESTS_ONLY environment variable set to True, skipping run.")
        return

    # configuration
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    epochs = 2 # was 15
    batch_size = 10 # was 2
    learning_rate = 0.0005
    # TODO define keep_prob here as well

    # check if Kitti dataset is available
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # TODO OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # TODO OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # load VGG
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # create TF Placeholder for labels
        # TODO other placeholders?
        correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 2))

        # add layers
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # define optimizer
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        # TODO passing learning rate directly
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input, correct_label, vgg_keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        # TODO OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
