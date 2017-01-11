"""Converts ICDAR 2013 data to TFRecord file format with SequenceExample protos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import random
import threading
import numpy as np
import tensorflow as tf

from collections import namedtuple
from datetime import datetime
from tqdm import tqdm

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from tools.data_labelling import AnchorLabeller

tf.flags.DEFINE_string("train_imgs_dir",
                       "/home/ubuntu/DATA/synth_text/train_500k_scaled_imgs",
                       "Image directory.")
tf.flags.DEFINE_string("train_gts_dir",
                       "/home/ubuntu/DATA/synth_text/train_500k_scaled_gts",
                       "Ground truths directory.")
tf.flags.DEFINE_string("test_imgs_dir",
                       "./data/icdar_2013/Challenge2_Test_Task12_Images",
                       "Image directory.")
tf.flags.DEFINE_string("test_gts_dir",
                       "./data/icdar_2013/Challenge2_Test_Task1_GT",
                       "Ground truths directory.")
tf.flags.DEFINE_string("output_dir",
                       "/home/ubuntu/DATA/synth_text/tf_records_synth_500k",
                       "Output data directory.")
tf.flags.DEFINE_string("output_info_file",
                       "/home/ubuntu/DATA/synth_text/output_info_synth_500k.txt",
                       "Output dataset information file.")

tf.flags.DEFINE_integer("num_threads", 16,
                        "Number of threads to preprocess the images.")
tf.flags.DEFINE_integer("train_shards", 4096,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 1024,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 1024,
                        "Number of shards in test TFRecord files.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["filename", "bboxes"])


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._encoded_jpeg, channels=3
        )

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        height = image.shape[0]
        width = image.shape[1]
        return image, height, width

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting an float Feature into a SequenceExample proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _float_feature_list(values):
    """Wrapper for inserting an float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _to_sequence_example(image, decoder, labeller):
    """Builds a SequenceExample proto for an image-word pair.

    Args:
        image: An ImageMetadata object.
        decoder: An ImageDecoder object.
        vocab: A Vocabulary object.

    Returns:
        A SequenceExample proto.
    """
    print("Processing image %s" % image.filename)

    # Load image
    with tf.gfile.FastGFile(image.filename, "r") as f:
        encoded_image = f.read()

    # Check image
    try:
        _, height, width = decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    # Generate anchors and label them
    try:
        labels = labeller.generate_anchors(height, width, image.bboxes)
    except Exception:
        return None

    anchors = labels["anchors"].tolist()
    classes = labels["classes"].astype("int64").tolist()
    coord_ids = labels["coord_ids"].astype("int64").tolist()
    coords = labels["coords"].tolist()
    side_classes = labels["side_classes"].astype("int64").tolist()
    side_ids = labels["side_ids"].astype("int64").tolist()
    offsets = labels["offsets"]

    # Build sequence examples
    context = tf.train.Features(feature={
        "image/filename": _bytes_feature(image.filename),
        "image/encoded": _bytes_feature(encoded_image),
        "image/height": _int64_feature(height),
        "image/width": _int64_feature(width),
    })

    feature_lists = tf.train.FeatureLists(feature_list={
        "image/anchors/anchors": _float_feature_list(anchors),
        "image/anchors/classes": _int64_feature_list(classes),
        "image/coords/ids": _int64_feature_list(coord_ids),
        "image/coords/coords": _float_feature_list(coords),
        "image/sides/side_classes": _int64_feature_list(side_classes),
        "image/sides/ids": _int64_feature_list(side_ids),
        "image/sides/offsets": _float_feature_list(offsets),
    })

    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    print("Done image %s" % image.filename)

    return sequence_example

def _process_image_files(thread_index, ranges, name, images, decoder, labeller,
                         num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      decoder: An ImageDecoder object.
      labeller: A AnchorLabeller object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g.
        # 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, labeller)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d images to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d images to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, labeller, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      labeller: A AnchorLabeller object.
      num_shards: Integer number of shards for the output files.
    """
    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]]
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG imagese)
    decoder = ImageDecoder()

    # Launch a thread for each batch
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, labeller, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all threads to terminate
    coord.join(threads)

    print("%s: Finished processing all %d images in data set '%s'." %
          (datetime.now(), len(images), name))

def parse_bbox_icdar_2013(imgs_dir, gts_dir):
    """Load images and processes the ground truths.

    Args:
        imgs_dir: Directory containing the image files.
        gts_dir: Directory containing the ground truth files.

    Returns:
        A list of ImageMetadata
    """
    imgs = glob.glob(os.path.join(imgs_dir, "*.jpg"))
    gts = glob.glob(os.path.join(gts_dir, "*.txt"))
    gts.sort()
    imgs.sort()

    metadata = []
    for img, gt in zip(imgs, gts):
        filename = img
        bboxes = []

        with open(gt) as f:
            for line in f:
                _left, _top, _right, _bottom, word = [item for item in
                                                  line.rstrip('\r\n').split(" ")]
                left, top, right, bottom = float(_left), float(_top), float(_right), float(_bottom)
                bboxes.append([left, top, right, bottom])
        bboxes = np.array(bboxes, dtype=np.float)

        metadata.append(ImageMetadata(filename, bboxes))

    return metadata


def _load_and_process_metadata(imgs_dir, gts_dir):
    """Loads image metadata from a file and processes the ground truths.

    Args:
        image_dir: Directory containing the image files.
        gts_dir: Directory containing the ground truth files.

    Returns:
        A list of ImageMetadata.
    """
    # Extract data and combine the data into a list of ImageMetadata
    print("Processing raw data")
    image_metadata = parse_bbox_icdar_2013(imgs_dir, gts_dir)

    print("Finished processing %d images in %s" %
          (len(image_metadata), imgs_dir))

    return image_metadata

def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert FLAGS.train_imgs_dir, "--train_imgs_dir is required"
    assert FLAGS.train_gts_dir, "--train_gts_dir is required"
    # assert FLAGS.test_imgs_dir, "--test_imgs_dir is required"
    # assert FLAGS.test_gts_dir, "--test_gts_dir is required"
    assert FLAGS.output_dir, "--output_dir is required"
    assert FLAGS.output_info_file, "--output_info_file is required"

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    # Load image metadata
    dataset = _load_and_process_metadata(FLAGS.train_imgs_dir,
                                         FLAGS.train_gts_dir)
    # test_dataset = _load_and_process_metadata(FLAGS.test_imgs_dir,
    #                                           FLAGS.test_gts_dir)

    # Distribute the dataset into:
    #   train_dataset = 80%
    #   val_dataset = 20%
    train_cutoff = int(0.8 * len(dataset))
    train_dataset = dataset[0:train_cutoff]
    val_dataset = dataset[train_cutoff:]

    # Save dataset info to file
    with open(FLAGS.output_info_file, "w") as f:
        f.writelines("%s\n" % datetime.now())
        f.writelines("num_train_samples: %d\n" % len(train_dataset))
        f.writelines("num_val_samples: %d\n" % len(val_dataset))
        # f.writelines("num_val_samples: %d\n" % len(test_dataset))
    tf.logging.info("Write dataset infomation into %s" %
                    FLAGS.output_info_file)

    # Create anchor labeller
    labeller = AnchorLabeller()

    _process_dataset("train", train_dataset, labeller, FLAGS.train_shards)
    _process_dataset("val", val_dataset, labeller, FLAGS.val_shards)
    # _process_dataset("test", test_dataset, labeller, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
