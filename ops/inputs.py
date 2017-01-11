"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

def parse_sequence_example(serialized):
    """Parses a tensorflow.SequenceExample into an image and caption.

    Args:
        serialized: A scalar string Tensor; a single serialized SequenceExample.

    Returns:
        image: A scalar string Tensor containing a JPEG encoded image.
        img_w: A Tensor containing the width size of image.
        img_h: A Tensor containing the height size of image.
        targets: A dictionary containing labels of image.
    """
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string),
        },
        sequence_features={
            'image/anchors/anchors': tf.FixedLenSequenceFeature([4], dtype=tf.float32),
            'image/anchors/classes': tf.FixedLenSequenceFeature([1], dtype=tf.int64),
            'image/coords/ids': tf.FixedLenSequenceFeature([1], dtype=tf.int64),
            'image/coords/coords': tf.FixedLenSequenceFeature([2], dtype=tf.float32),
            'image/sides/side_classes': tf.FixedLenSequenceFeature([1], dtype=tf.int64),
            'image/sides/ids': tf.FixedLenSequenceFeature([1], dtype=tf.int64),
            'image/sides/offsets': tf.FixedLenSequenceFeature([1], dtype=tf.float32),
        })

    image = context['image/encoded']
    img_file = context['image/filename']

    targets = dict()
    targets['anchors'] = sequence['image/anchors/anchors']
    targets['classes'] = sequence['image/anchors/classes']
    targets['coord_ids'] = sequence['image/coords/ids']
    targets['coords'] = sequence['image/coords/coords']
    targets['side_classes'] = sequence['image/sides/side_classes']
    targets['side_ids'] = sequence['image/sides/ids']
    targets['offsets'] = sequence['image/sides/offsets']

    return image, targets, img_file


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
    """Prefetches string values from disk into an input queue.

    In training the capacity of the queue is important because a larger queue
    means better mixing of training examples between shards. The minimum number of
    values kept in the queue is values_per_shard * input_queue_capacity_factor,
    where input_queue_memory factor should be chosen to trade-off better mixing
    with memory usage.

    Args:
      reader: Instance of tf.ReaderBase.
      file_pattern: Comma-separated list of file patterns (e.g.
          /tmp/train_data-?????-of-00100).
      is_training: Boolean; whether prefetching for training or eval.
      batch_size: Model batch size used to determine queue capacity.
      values_per_shard: Approximate number of values per shard.
      input_queue_capacity_factor: Minimum number of values to keep in the queue
        in multiples of values_per_shard. See comments above.
      num_reader_threads: Number of reader threads to fill the queue.
      shard_queue_name: Name for the shards filename queue.
      value_queue_name: Name for the values input queue.

    Returns:
      A Queue containing prefetched string values.
    """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name)
    else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))
    tf.summary.scalar(
        "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

    return values_queue


def main(argv):
    _test_dataset(FLAGS.input_file_pattern)

if __name__ == '__main__':
    tf.app.run()
