"""Image extractor ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim


def vgg16(images,
          trainable=True,
          is_training=True,
          weight_decay=0.00004,
          stddev=0.1,
          dropout_keep_prob=0.5,
          use_batch_norm=True,
          batch_norm_params=None,
          add_summaries=True,
          scope="vgg_16"):
    """Builds an Oxford Net VGG-16 subgraph for image feature extractor.

    Args:
        images: A float32 Tensor of shape [batch, height, width, channels].
            trainable: Whether the vgg submodel should be trainable or
            not.
        is_training: Boolean indicating training mode or not.
        weight_decay: Coefficient for weight regularization.
        stddev: The standard deviation of the trunctated normal weight
            initializer.
        dropout_keep_prob: Dropout keep probability.
        use_batch_norm: Whether to use batch normalization.
        batch_norm_params: Parameters for batch normalization. See
            tf.contrib.layers.batch_norm for details.
        add_summaries: Whether to add activation summaries.
        scope: Optional Variable scope.

    Returns:
      net: The built VGG-16
    """
    is_model_training = trainable and is_training

    if use_batch_norm:
        # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
    else:
        batch_norm_params = None

    # Add regularizer
    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    with tf.variable_scope(scope, 'vgg_16', [images]) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=weights_regularizer,
                            outputs_collections=end_points_collection,
                            trainable=trainable):

            # Pretrained vgg-16 layers
            with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation_fn=tf.nn.relu):

                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3],scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='SAME')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4', padding='SAME')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)

    # Add summaries.
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)

    return net
