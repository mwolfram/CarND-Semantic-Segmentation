import os.path
import tensorflow as tf
import helper
import glob
import warnings
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
    # TODO no activation so far)

    # 1x1 convolution
    conv_1x1_layer = tf.layers.conv2d(vgg_layer7_out,
                                      vgg_layer7_num_outputs,
                                      kernel_size=1,
                                      strides=1,
                                      name="conv_1x1_layer")

    """
    tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [1,6,19,512] vs. [1,10,36,512]
	 [[Node: skip_layer_1 = Add[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"](conv_transposed_layer_1/BiasAdd, layer4_out)]]
    """

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
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    #logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #mean_iou, update_op = tf.metrics.mean_iou(correct_label, nn_last_layer, num_classes)
    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(mean_iou)

    # from http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    trn_labels=tf.reshape(correct_label, (-1, num_classes))
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
    train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # For inference/visualization
    #valid_prediction=tf.reshape(logits,tf.shape(vgg.up))

# TODO return cross_entropy_loss
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
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    rate = 0.001
    dropout = 0.5

    print("Start training")
    for epoch in range(epochs):
        print("Epoch", str(epoch), "|", end="")
        for images, labels in get_batches_fn(batch_size):
            sess.run(train_op, feed_dict={input_image: images, correct_label: labels, keep_prob: dropout, learning_rate: rate})
            loss = sess.run(cross_entropy_loss, feed_dict={input_image: images, correct_label: labels, keep_prob: 1.0})
            print("=", end="")
        print()

tests.test_train_nn(train_nn)

"""
    def evaluate(X_data, y_data, batch_size=BATCH_SIZE):
        num_examples = len(X_data)
        total_accuracy = 0.0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_data[offset:end], y_data[offset:end]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train_model(X_train, y_train, X_valid, y_valid, model_path, epochs=EPOCHS, batch_size=BATCH_SIZE):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)
            print("Target model name prefix is " + MODEL_PATH)
            print("Training in " + str(epochs) + " epochs, with batch size " + str(batch_size) + "...")
            for i in range(epochs):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, batch_size):
                     end = offset + batch_size
                     batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                     sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                training_accuracy = evaluate(X_train, y_train, batch_size)
                validation_accuracy = evaluate(X_valid, y_valid, batch_size)
                print("EPOCH {} ...".format(i+1))
                print("Training accuracy = {:.3f}".format(training_accuracy))
                print("Validation accuracy = {:.3f}".format(validation_accuracy))
                saver.save(sess, str(model_path) + '_after_epoch_' + str(i+1))
                print("Model saved")
                print()
    """

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    epochs = 2
    batch_size = 5

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # create placeholders for session
    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # load VGG
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # add layers
        conv_transposed_layer_3 = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # optimize function
        logits, train_op, cross_entropy_loss = optimize(conv_transposed_layer_3, correct_label, learning_rate, num_classes)

        #  Train NN using the train_nn function
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
