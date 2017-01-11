"""Evaluate the model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time

import tensorflow as tf

import configuration
import text_detector

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern",
                       "./data/tf_records_synth_10k/val-*",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir",
                       "checkpoints_synth/train_lr_3e-4_wa_1_wc_2_wo_35_ws_25",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir",
                       "checkpoints_synth/val_train_lr_3e-4_wa_1_wc_2_wo_35_ws_25",
                       "Directory to write event logs.")

tf.flags.DEFINE_integer("eval_interval_secs", 10,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 2000,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 10,
                        "Minimum global step to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate_model(sess, model, global_step, summary_writer, summary_op):
    """Computes accuracy and losses over the evaluation dataset.

    Args:
        sess: Session object.
        model: Instance of TextDetector; the model to evaluate.
        global_step: Integer; global step of the model checkpoint.
        summary_writer: Instance of SummaryWriter.
        summary_op: Op for generating model summaries.
    """
    # Log model summaries on a single batch.
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, global_step)

    # Compute accuracy and losses over the entire dataset.
    num_eval_batches = int(
        math.ceil(FLAGS.num_eval_examples / model.config.batch_size))

    start_time = time.time()
    sum_acc = 0.
    sum_side_acc = 0.
    sum_class_loss = 0.
    sum_side_class_loss = 0.
    sum_coords_loss = 0.
    sum_offset_loss = 0.

    for i in xrange(num_eval_batches):
        class_acc, side_class_acc, class_loss, side_class_loss, coords_loss, offset_loss,  = sess.run(
            [model.class_acc, model.side_class_acc, model.class_loss,
             model.side_class_loss, model.coords_loss, model.offset_loss])
        sum_acc += class_acc
        sum_side_acc += side_class_acc
        sum_class_loss += class_loss
        sum_side_class_loss += side_class_loss
        sum_coords_loss += coords_loss
        sum_offset_loss += offset_loss

        if not i % 100:
            tf.logging.info("Computed losses for %d of %d batches.", i + 1,
                            num_eval_batches)
    sum_acc /= num_eval_batches
    sum_side_acc /= num_eval_batches
    sum_class_loss /= num_eval_batches
    sum_side_class_loss /= num_eval_batches
    sum_coords_loss /= num_eval_batches
    sum_offset_loss /= num_eval_batches
    eval_time = time.time() - start_time

    tf.logging.info("Class accuracy = %f, Side accuracy =%f, Class loss = %f, Side loss = %f, Coords loss = %f, Offset loss = %f (%.2g sec)",
                    sum_acc, sum_side_acc, sum_class_loss, sum_side_class_loss,
                    sum_coords_loss, sum_offset_loss, eval_time)

    # Log the accuracy and losses to the SummaryWriter
    summary = tf.Summary()

    value_acc = summary.value.add()
    value_acc.simple_value = sum_acc
    value_acc.tag = "Class_acc"

    value_side_acc = summary.value.add()
    value_side_acc.simple_value = sum_side_acc
    value_side_acc.tag = "Side_Class_acc"

    value_class_loss = summary.value.add()
    value_class_loss.simple_value = sum_class_loss
    value_class_loss.tag = "Class_loss"

    value_side_class_loss = summary.value.add()
    value_side_class_loss.simple_value = sum_side_class_loss
    value_side_class_loss.tag = "Side_Class_loss"

    value_coords_loss = summary.value.add()
    value_coords_loss.simple_value = sum_coords_loss
    value_coords_loss.tag = "Coords_loss"

    value_offset_loss = summary.value.add()
    value_offset_loss.simple_value = sum_offset_loss
    value_offset_loss.tag = "Offset_loss"

    summary_writer.add_summary(summary, global_step)

    # Write the Events file to the eval directory.
    summary_writer.flush()
    tf.logging.info("Finished processing evaluation at global step %d.",
                    global_step)


def run_once(model, saver, summary_writer, summary_op):
    """Evaluates the latest model checkpoint.

    Args:
        model: Instance of TextDetector; the model to evaluate.
        saver: Instance of tf.train.Saver for restoring model Variables.
        summary_writer: Instance of SummaryWriter.
        summary_op: Op for generating model summaries.
    """
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not model_path:
        tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                        FLAGS.checkpoint_dir)
        return

    with tf.Session() as sess:
        # Load model from checkpoint
        tf.logging.info("Loading model from checkpoint: %s", model_path)
        saver.restore(sess, model_path)
        global_step = tf.train.global_step(sess, model.global_step.name)
        tf.logging.info("Successfully loaded %s at global_step = %d.",
                        os.path.basename(model_path), global_step)
        if global_step < FLAGS.min_global_step:
            tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                            FLAGS.min_global_step)
            return

        # Start the queue runners
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Run evaluation on the lasted checkpoint
        try:
            evaluate_model(
                sess=sess,
                model=model,
                global_step=global_step,
                summary_writer=summary_writer,
                summary_op=summary_op)
        except Exception, e:
            tf.logging.error("Evaluation failed.")
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def run():
    """Runs evaluation in a loop, and logs summaries to TensorBoard."""
# Create the evaluation directory if it doesn't exist.
    eval_dir = FLAGS.eval_dir
    if not tf.gfile.IsDirectory(eval_dir):
        tf.logging.info("Creating eval directory: %s", eval_dir)
        tf.gfile.MakeDirs(eval_dir)

    g = tf.Graph()
    with g.as_default():
        # Build the model for evaluation
        model_config = configuration.ModelConfig()
        model_config.input_file_pattern = FLAGS.input_file_pattern
        model = text_detector.TextDetector(model_config, mode='eval')
        model.build()

        # Create the Saver to restore model Variables
        saver = tf.train.Saver()

        # Create the summary operation and the summary writer
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(eval_dir)

        g.finalize()

        # Run a new evaluation run every eval_interval_secs
        while True:
            start = time.time()
            tf.logging.info("Starting evaluation at " + time.strftime(
                "%Y-%m-%d-%H:%M:%S", time.localtime()))
            run_once(model, saver, summary_writer, summary_op)
            time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)

def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
    assert FLAGS.eval_dir, "--eval_dir is required"
    run()

if __name__ == "__main__":
    tf.app.run()
