"""Detect texts in images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import cv2

import tensorflow as tf

import configuration
import inference_wrapper

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path",
                       "checkpoints_svt/fintune_lr_1e-5_finetune_train_side_10_weight_3",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_files", "test_real_images/*.jpg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        infer = inference_wrapper.InferenceWrapper()
        restore_fn = infer.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)

    g.finalize()

    # Load file in the provied directory
    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running text detection on %d files matching %s",
              len(filenames), FLAGS.input_files)


    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        filenames.sort()
        # Predict
        for filename in filenames:
            with tf.gfile.GFile(filename, "r") as f:
                # Read image
                cv_img = cv2.imread(filename)
                image = f.read()

                # Make prediction
                tic = time.time()
                text_bboxes = infer.inference_step(sess, image)
                toc = time.time()
                print("Prediction for image %s in %.3f ms" %
                      (os.path.basename(filename), (toc - tic) * 1000))

                # Show the result
                for i in range(len(text_bboxes)):
                    text = "{}: {:.3f}".format(i, float(text_bboxes[i][4]))

                    cv2.putText(cv_img, text, (int(text_bboxes[i][0]) + 5, int(text_bboxes[i][1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                    cv2.rectangle(cv_img,
                        (int(text_bboxes[i][0]), int(text_bboxes[i][1])),
                        (int(text_bboxes[i][2]), int(text_bboxes[i][3])),
                        (0,0,255), 2)

                cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
                cv2.resizeWindow('image', 1500, 900);
                cv2.imshow('image', cv_img)

                k = cv2.waitKey(0)
                if k == ord('n'):
                    cv2.destroyAllWindows()

if __name__ == "__main__":
    tf.app.run()
