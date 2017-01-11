"""Helper functions for image preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


VGG_MEAN = [103.939, 116.779, 123.68]


def process_image(encoded_image,
                  mode, width=1200,
                  image_format="jpeg"):
    """Decode an image, resize and apply random distortions.

    In training, images are distorted slightly differently depending on thread_id.

    Args:
        encoded_image: String Tensor containing the image.
        mode: "train", "eval" or "inference".
        width: Width of the output image (only use in inference mode), the
            height of output image will be computed so that the output image
            keeps the same width-height ratio.
        image_format: "jpeg" or "png".

    Returns:
        A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

    Raises:
        ValueError: If image_format is invalid.
    """

    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.
    def image_summary(name, image):
        tf.summary.image(name, tf.expand_dims(image, 0))

    with tf.device("/cpu:0"):
        # Decode image into a float32 tensor of shape [?, ?, 3] with values in [0, 1).
        with tf.name_scope("decode", values=[encoded_image]):
            if image_format == "jpeg":
                image = tf.image.decode_jpeg(encoded_image, channels=3)
            elif image_format == "png":
                image = tf.image.decode_png(encoded_image, channels=3)
            else:
                raise ValueError("Invalid image format: %s" % image_format)

        # Convert image to float32 type
        image = tf.to_float(image)
        image_summary("original_image", image)

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=image)
        image = tf.concat(axis=2, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        # Resize the image when doing inference
        if mode == "inference":
            old_width = tf.shape(image)[1]
            old_height = tf.shape(image)[0]

            def scale_h():
                h = tf.constant(1000, dtype=tf.int32)
                w = tf.to_int32(
                    (tf.to_float(h) / tf.to_float(old_height)) *
                    tf.to_float(old_width))
                return h, w

            def scale_w():
                w = tf.constant(1000, dtype=tf.int32)
                h = tf.to_int32(
                    (tf.to_float(w) / tf.to_float(old_width)) *
                    tf.to_float(old_height))
                return h, w

            new_height, new_width = tf.cond(old_width > old_height,
                                            scale_h, scale_w)

            image = tf.expand_dims(image, 0)
            image = tf.image.resize_images(image, [new_height, new_width])

            return image, old_height, old_width

    image_summary("final_image", image)

    return image
