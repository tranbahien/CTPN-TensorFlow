"""Text detection model implemented by TensorFlow"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from ops import image_extractor
from ops import image_processing
from ops import inputs as input_ops
from ops import utils as util_ops

slim = tf.contrib.slim

class TextDetector(object):
    """Text detection model used to detect text location in an image.
        This model is adapted from the paper `Detecting Text in Natural Image
        with Connectionist Text Proposal Network`
        (https://arxiv.org/abs/1609.03605)."""

    def __init__(self, config, mode, train_vgg=True, finetuning=False):
        """Basic setup.

        Args:
            config: Object containing configuration parameters.
            mode: "train", "eval" or "inference".
            train_vgg: Whether the vgg submodel variables are trainable.
            is_finetuning: Whether finetuning model from checkpoint.
        """
        assert mode in ["train", "eval", "inference"]
        self.config     = config
        self.mode       = mode
        self.train_vgg  = train_vgg
        self.finetuning = finetuning

        # Reader for the input data
        self.reader = tf.TFRecordReader()

        # Initialize all variable with  a random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # Int32 scalar Tensors storing the size of the original image and the
        # scaled image used for inference steps
        self.old_img_height = None
        self.old_img_width  = None
        self.new_img_height = None
        self.new_img_width  = None

        # An int32 sparse Tensor with shape [batch_size, seq_length].
        self.targets = None

        # A float32 Tensor with shape [num_anchors, 4] containing the coodinates
        # of each anchor
        self.anchors = None

        # Integer Tensors with shape [batch_of_anchors] storing the indexes
        # of selected anchors for training.
        self.sample_ids      = None
        self.side_sample_ids = None

        # Float32 Tensors with shape [num_anchors, 2] storing the predicted
        # and target classes of anchors
        self.pred_class_scores      = None # logits
        self.pred_classes           = None # normalized by softmax
        self.pred_side_class_scores = None # logits
        self.target_classes         = None
        self.target_side_classes    = None

        # A float32 Tensor with shape [num_anchors, 2] storing the predicted
        # and target relative vertical coords of anchors
        self.pred_coords_deltas   = None
        self.target_coords_deltas = None

        # A float32 Tensor with shape [num_anchors, 1] storing the predicted
        # and target relative offsets of anchors
        self.pred_offsets_deltas   = None
        self.target_offsets_deltas = None

        # A float32 Tensor with shape [height, width, 4] storing the coordiates
        # of region proposals
        self.proposals = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss      = None
        # A float32 scalar Tensor; the class loss.
        self.class_loss      = None
        # A float32 scalar Tensor; the vertical coordinates loss.
        self.coords_loss     = None
        # A float32 scalar Tensor; the horizontal offset loss.
        self.offset_loss     = None
        # A float32 scalar Tensor; the side-class loss.
        self.side_class_loss = None

        # A float32 scalar Tensor; predicted text/non-text
        self.class_acc      = None
        # A float32 scalar Tensor; the side-class loss.
        self.side_class_acc = None

        # Collection of variables from the vgg submodel.
        self.vgg_pretrained_variables   = []
        # Collection of pretrained variables
        self.model_pretrained_variables = []

        # Function to restore the vgg submodel from checkpoint.
        self.init_fn    = None
        # Function to restore the model from checkpoint for training.
        self.restore_fn = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image):
        """Decodes and processes an image string.
        Inputs:
            encoded_image: A scalar string Tensor; the encoded image.

        Outputs:
            A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              mode=self.mode,
                                              width=self.config.image_width,
                                              image_format=self.config.image_format)

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
            self.images
            self.input_seqs
            self.targets (training and eval only)
        """
        if self.mode == "inference":
            with tf.name_scope("image_input"):
                "In inference mode, images are fed via placeholders."
                image_feed = tf.placeholder(
                    dtype=tf.string, shape=[], name="image_feed")

                # Process image and insert batch dimensions
                images, old_h, old_w = self.process_image(image_feed)

                self.old_img_height = old_h
                self.old_img_width  = old_w
                self.new_img_height = tf.shape(images)[1]
                self.new_img_width  = tf.shape(images)[2]

                # No targets in inference mode.
                targets = None

        else:
            with tf.name_scope("load_tf_records"), tf.device("/cpu:0"):
                # Prefetch serialized SequenceExample protos.
                input_queue = input_ops.prefetch_input_data(
                    self.reader,
                    self.config.input_file_pattern,
                    is_training=self.is_training(),
                    batch_size=self.config.batch_size,
                    values_per_shard=self.config.values_per_input_shard,
                    input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                    num_reader_threads=self.config.num_input_reader_threads)

                # Decode data from sequence example.
                serialized_sequence_example = input_queue.dequeue()
                image, targets, img_file = input_ops.parse_sequence_example(
                    serialized_sequence_example)
                images = tf.expand_dims(self.process_image(image), 0)

        self.images = images
        self.targets = targets


    def build_image_features(self):
        """Builds the image model subgraph and generates image features.

        Inputs:
            self.images

        Outputs:
            self.pred_class_scores
            self.pred_coords_deltas
            self.pred_offsets_deltas
            self.proposals
        """
        # Get image features created by CNN model
        vgg_output = image_extractor.vgg16(self.images,
                                           trainable=self.train_vgg,
                                           is_training=self.is_training(),
                                           weight_decay=self.config.vgg_weight_decay)
        self.vgg_pretrained_variables = slim.get_variables_to_restore(
            include=["vgg_16/conv1", "vgg_16/conv2", "vgg_16/conv3",
                     "vgg_16/conv4", "vgg_16/conv5"])

        with tf.variable_scope("im2col") as scope:
            # Extract patches of feature maps, [N, H, W, C]
            patches = util_ops.get_patches(vgg_output, 3)

            # Tranpose patches to [H, W, C, N]
            tranposed_patches = tf.transpose(patches, (1, 2, 3, 0))

            # Reshape patches to [H, W, C*N]
            shape = tf.shape(tranposed_patches)
            N = tranposed_patches.get_shape().as_list()[-1]
            C = tranposed_patches.get_shape().as_list()[-2]
            image_features = tf.reshape(tranposed_patches,
                                        [shape[0], shape[1], N * C])

            self.image_features = image_features

    def build_region_proposal_network(self):
        """Build region proposal network

        Inputs:
            self.image_features
            self.targets (training and eval only)

        Ouputs:
            self.pred_side_class_scores
            self.pred_class_scores
            self.pred_coords_deltas
            self.pred_offsets_deltas
        """
        with tf.name_scope("get_feat_shape") as scope:
            # Find the size of image features
            feat_shape = tf.shape(self.image_features)
            height, width = feat_shape[0], feat_shape[1]

        # Define the cell of RNN model
        lstm_cell = tf.contrib.rnn.LSTMCell(
            num_units=self.config.num_lstm_units, state_is_tuple=True)

        if self.mode == "train":
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                cell=lstm_cell,
                output_keep_prob=self.config.lstm_dropout_keep_prob)

        # Build the RNN model
        with tf.variable_scope("lstm", initializer=self.initializer) as scope:
            # Stack RNN cell if using multi-layer LSTM
            with tf.name_scope("multilayer_rnn_cell"):
                cell = tf.contrib.rnn.MultiRNNCell(
                    cells=[lstm_cell] * self.config.num_lstm_layers,
                    state_is_tuple=True)

            seq_lens = tf.ones(shape=[height], dtype=tf.int32) * width

            # Create the Bidirectional-LSTM
            _lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell,
                cell_bw=cell,
                inputs=self.image_features,
                sequence_length=seq_lens,
                dtype=tf.float32)

            # Merge and reshape the outputs of Bidirectional-LSTM from
            # [H, seq_len, hid_len*2] to [batch_size, H, seq_len, hid_len*2]
            rnn_outputs_fw, rnn_outputs_bw = _lstm_outputs
            lstm_outputs = tf.concat(axis=2, values=[rnn_outputs_fw, rnn_outputs_bw],
                                     name='concat_lstm_outputs')

        # Feed output of LSTM into fully connected layer
        with tf.variable_scope("fc") as scope:
            lstm_outputs_flat = tf.reshape(
                lstm_outputs,
                [-1, self.config.num_lstm_units * 2],
                name="lstm_outputs_flat")

            fc_outputs_flat = slim.fully_connected(
                inputs=lstm_outputs_flat,
                num_outputs=self.config.num_hidden_units,
                activation_fn=tf.nn.relu)

            # Drop units of fully-connected layer
            if self.mode == "train":
                fc_outputs_flat = tf.nn.dropout(
                    fc_outputs_flat, self.config.fc_dropout_keep_prob)

        # Make predictions
        with tf.name_scope("predictions") as pred_scope:
            # Predict class of each anchor
            with tf.variable_scope("rpn_cls") as scope:
                cls_scores_flat = slim.fully_connected(
                    inputs=fc_outputs_flat,
                    num_outputs=20,
                    activation_fn=None)

                cls_pred = tf.reshape(cls_scores_flat, [height, -1, 20])

            # Predict vertical coordinates of each anchor
            with tf.variable_scope("rpn_bbox_delta") as scope:
                bbox_delta_pred_flat = slim.fully_connected(
                    inputs=fc_outputs_flat,
                    num_outputs=20,
                    activation_fn=None)

                bbox_delta_pred = tf.reshape(bbox_delta_pred_flat,
                                             [height, -1, 20])


            # Predict class of each side-anchor
            with tf.variable_scope("rpn_side_cls") as scope:
                side_cls_scores_flat = slim.fully_connected(
                    inputs=fc_outputs_flat,
                    num_outputs=20,
                    activation_fn=None)

                side_cls_pred = tf.reshape(side_cls_scores_flat,
                                           [height, -1, 20])

            # Predict offset of side-anchor
            with tf.variable_scope("rpn_offset_delta") as scope:
                offset_delta_pred_flat = slim.fully_connected(
                    inputs=fc_outputs_flat,
                    num_outputs=10,
                    activation_fn=None)

                offset_delta_pred = tf.reshape(offset_delta_pred_flat,
                                               [height, -1, 10])

            # Compute the text proposals
            with tf.variable_scope("rpn_proposals") as scope:
                # Generate anchors on feature maps
                anchors = util_ops.generate_anchors(
                    tf.stack([height, width]),
                    tf.constant(16, dtype=tf.int32),
                    tf.constant(10, dtype=tf.int32))

            self.anchors = anchors
            self.pred_side_class_scores = side_cls_pred
            self.pred_class_scores = cls_pred
            self.pred_coords_deltas = bbox_delta_pred
            self.pred_offsets_deltas = offset_delta_pred

    def build_inference(self):
        """Build prediction operation used for inference steps.

        The image is passed through the network to make proposal and offset
        predictions. Next, we connect these small text proposals to build the
        final text bounding boxes.

        Inputs:
            self.anchors
            self.pred_coords_deltas
            self.pred_class_scores
            self.pred_offsets_deltas
            self.pred_side_class_scores

        Outputs:
            text_bboxes: A float32 Tensor having shape [num_bboxes, 5] contains
                coordianates (left, top, right, bottom) and the corresponding
                predicted confidence of each text bounding box. Note that we
                only make `text_bboxes` operation in graph for getting the
                result when doing inference.

        """
        with tf.variable_scope("inference", reuse=True):
            # Calculate the text proposals based on anchors and corresponding
            # relative vertical coordianates
            proposals = util_ops.apply_vertical_deltas_to_anchors(
                tf.reshape(self.pred_coords_deltas, [-1, 2]),
                tf.reshape(self.anchors, [-1, 4]))

            # Calculate the score of each predicted text proposals normalized by
            # softmax function
            normalized_pred_scores = tf.nn.softmax(
                tf.reshape(self.pred_class_scores, [-1, 2]))
            scores = normalized_pred_scores[:, 1]

            # Calculate the offset of each text proposals based on defined
            # anchors and relative horizontal coordinates
            offsets = tf.reshape(
                util_ops.apply_horizontal_deltas_to_anchors(
                    tf.reshape(self.pred_offsets_deltas, [-1, 1]),
                    tf.reshape(self.anchors, [-1, 4])), [-1])

            # Calculate the score of each predicted offsets normalized by
            # softmax function
            normalized_pred_side_scores = tf.nn.softmax(
                tf.reshape(self.pred_side_class_scores, [-1, 2]))
            offset_scores = normalized_pred_side_scores[:, 1]

            # Connect text proposals to build the final text bounding boxes
            with tf.device("/cpu:0"):
                text_bboxes = util_ops.connect_proposals(
                    tf.reshape(proposals, [-1, 4]),
                    tf.reshape(scores, [-1, 1]),
                    tf.reshape(offsets, [-1, 1]),
                    tf.reshape(offset_scores, [-1, 1]),
                    tf.to_int32(tf.stack(
                        [self.new_img_height, self.new_img_width]))
                )

            # Rescale the predicted bounding boxes based on the original size of
            # the input image
            text_bboxes = util_ops.rescale_bboxes(
                text_bboxes,
                self.old_img_height, self.old_img_width,
                self.new_img_height, self.new_img_width)

            # Define operation in graph for getting the predicted bounding boxes
            # at inference steps
            text_bboxes = tf.identity(text_bboxes, name="text_bboxes")


    def build_batch_samples(self):
        """Select batch of anchors for training.

        We create a batch of anchors with 1:1 ratio for positive and negative
        anchors for training to determine the vertical coordinates
        (coords_deltas) and the class (text or non-text) of each anchors. We
        also build another batch of anchors with 1:1 ratio for positive and
        negative anchors for optimizing the predicted class (containing-offset
        or non-containing-offset) and the relative horizontal coordinates
        (offsets_deltas).

        Inputs:
            self.targets

        Ouputs:
            self.sample_ids
        """
        # Select the batch of anchors for detecting text or non-text
        with tf.variable_scope("batching") as scope, tf.device("/cpu:0"):
            # Get targets
            classes = tf.to_int32(self.targets["classes"])

            # Select only positive anchors and negative anchors
            pos_indices = tf.to_int32(tf.where(
                tf.equal(classes, tf.ones_like(classes, tf.int32))))[:, 0]
            neg_indices = tf.to_int32(tf.where(
                tf.equal(classes, tf.zeros_like(classes, tf.int32))))[:, 0]

            # Calculate the number of positive samples and negative samples
            # with 1:1 ratio
            num_samples_anchors = tf.constant(
                int(self.config.batch_anchor_size / 2), dtype=tf.int64)
            num_pos_anchors = tf.cond(
                tf.to_int64(tf.shape(pos_indices)[0]) <= num_samples_anchors,
                lambda: tf.to_int64(tf.shape(pos_indices)[0]),
                lambda: num_samples_anchors)
            num_neg_anchors = tf.cond(
                num_pos_anchors < num_samples_anchors,
                lambda: num_samples_anchors * 2 - num_pos_anchors,
                lambda: num_samples_anchors)

            # Get randomly (with uniform distribution) positive samples and
            # negative samples
            pos_random_range = tf.random_shuffle(
                tf.range(tf.shape(pos_indices)[0]))
            neg_random_range = tf.random_shuffle(
                tf.range(tf.shape(neg_indices)[0]))

            pos_sample_ids = tf.gather(pos_indices,
                                       pos_random_range[:tf.to_int32(num_pos_anchors)])
            neg_sample_ids = tf.gather(neg_indices,
                                       neg_random_range[:tf.to_int32(num_neg_anchors)])

            sample_ids = tf.random_shuffle(tf.concat(axis=0, values=[pos_sample_ids, neg_sample_ids]))

        # Select the batch of anchors for detecting containing-offset or
        # not-containing-offset
        with tf.variable_scope("side_batching") as scope, tf.device("/cpu:0"):
            # Get side targets
            side_classes = tf.to_int32(self.targets["side_classes"])

            # Select only positive anchors and negative side-anchors
            pos_side_indices = tf.to_int32(tf.where(
                tf.equal(side_classes,
                         tf.ones_like(side_classes, tf.int32))))[:, 0]
            neg_side_indices = tf.to_int32(tf.where(
                tf.equal(side_classes,
                         tf.zeros_like(side_classes, tf.int32))))[:, 0]

            # Calculate the number of positive samples and negative samples
            # with 1:1 ratio
            num_samples_side_anchors = tf.constant(
                int(self.config.batch_side_anchor_size / 2), dtype=tf.int64)
            num_pos_side_anchors = tf.cond(
                tf.to_int64(tf.shape(pos_side_indices)[0]) <= num_samples_side_anchors,
                lambda: tf.to_int64(tf.shape(pos_side_indices)[0]),
                lambda: num_samples_side_anchors)
            num_neg_side_anchors = tf.cond(
                num_pos_side_anchors < num_samples_side_anchors,
                lambda: num_samples_side_anchors * 2 - num_pos_side_anchors,
                lambda: num_samples_side_anchors)

            # Get randomly (with uniform distribution) positive side-samples
            # and negative side-samples
            pos_side_random_range = tf.random_shuffle(
                tf.range(tf.shape(pos_side_indices)[0]))
            neg_side_random_range = tf.random_shuffle(
                tf.range(tf.shape(neg_side_indices)[0]))

            pos_side_sample_ids = tf.gather(pos_side_indices,
                                       pos_side_random_range[:tf.to_int32(num_pos_side_anchors)])
            neg_side_sample_ids = tf.gather(neg_side_indices,
                                       neg_side_random_range[:tf.to_int32(num_neg_side_anchors)])

            side_sample_ids = tf.random_shuffle(tf.concat(axis=0, values=[pos_side_sample_ids, neg_side_sample_ids]))

        self.sample_ids = sample_ids
        self.side_sample_ids = side_sample_ids

    def build_targets(self):
        """Build the target labels for training.

        Inputs:
            self.targets
            self.pred_class_scores

        Ouputs:
            self.target_classes
            self.pred_class_scores
            self.target_coords_deltas
            self.pred_coords_deltas
            self.target_offsets_deltas
            self.pred_offsets_deltas
        """
        with tf.name_scope("targets") as target_scope, tf.device("/cpu:0"):
            sample_ids = self.sample_ids
            side_sample_ids = self.side_sample_ids
            anchors = self.targets["anchors"]
            classes = self.targets["classes"]
            side_classes = self.targets["side_classes"]
            coords = self.targets["coords"]
            offsets = self.targets["offsets"]
            coord_ids = tf.to_int32(tf.reshape(self.targets["coord_ids"],
                                   [tf.shape(self.targets["coord_ids"])[0]]))
            offset_ids = tf.reshape(self.targets["side_ids"],
                                    [tf.shape(self.targets["side_ids"])[0]])

            # Get the classes targets
            with tf.variable_scope("target_classes") as scope:
                self.target_classes = tf.gather_nd(
                    classes, tf.reshape(sample_ids, [-1, 1]))
                self.target_classes = tf.reshape(self.target_classes, [-1])

                self.pred_class_scores = tf.reshape(
                    self.pred_class_scores, [-1, 2])
                self.pred_class_scores = tf.gather_nd(
                    self.pred_class_scores, tf.reshape(sample_ids, [-1, 1]))

            # Calculate the vertical coordinates targets
            with tf.variable_scope("target_coord_deltas") as scope:
                target_coords_masks, target_coords_ids = util_ops.get_intersect(
                    coord_ids, sample_ids)
                target_anchors = tf.gather_nd(
                    anchors, tf.reshape(target_coords_ids, [-1, 1]))
                target_coords = tf.gather_nd(
                    coords, tf.reshape(target_coords_masks, [-1, 1]))
                self.target_coords_deltas = util_ops.convert_to_vertical_deltas(
                    target_coords, target_anchors)

                self.pred_coords_deltas = tf.reshape(self.pred_coords_deltas,
                                                     [-1, 2])
                self.pred_coords_deltas = tf.gather_nd(
                    self.pred_coords_deltas, tf.reshape(target_coords_ids,
                                                        [-1, 1]))


            # Get the classes side-targets
            with tf.variable_scope("target_side_classes") as scope:
                self.target_side_classes = tf.gather_nd(
                    side_classes, tf.reshape(side_sample_ids, [-1, 1]))
                self.target_side_classes = tf.reshape(self.target_side_classes,
                                                      [-1])

                self.pred_side_class_scores = tf.reshape(
                    self.pred_side_class_scores, [-1, 2])
                self.pred_side_class_scores = tf.gather_nd(
                    self.pred_side_class_scores,
                    tf.reshape(side_sample_ids, [-1, 1]))

            # Calculate the offsets targets
            with tf.variable_scope("target_offset_deltas") as scope:
                target_offsets_masks, target_offsets_ids = util_ops.get_intersect(
                    offset_ids, sample_ids)
                target_anchors = tf.gather_nd(
                    anchors, tf.reshape(target_offsets_ids, [-1, 1]))
                target_offsets = tf.gather_nd(
                    offsets, tf.reshape(target_offsets_masks, [-1, 1]))
                self.target_offsets_deltas = util_ops.convert_to_horizontal_deltas(
                    target_offsets, target_anchors)

                self.pred_offsets_deltas = tf.reshape(self.pred_offsets_deltas,
                                                      [-1, 1])
                self.pred_offsets_deltas = tf.gather_nd(
                    self.pred_offsets_deltas, tf.reshape(target_offsets_ids,
                                                         [-1, 1]))

    def build_losses(self):
        """Build the loss functions for otimization.

        There are 4 main losses contributing to the total loss.
        Classification losses (`class_loss`, `side_class_loss`) for optimizing
        the predicted classes of each anchors (text or non-text,
        containing-offset or not-containing-offset). We also have some
        regression losses for optimizing the predicted horizontal and vertical
        coordinates.

        Inputs:
            self.target_classes
            self.pred_class_scores
            self.target_coords_deltas
            self.pred_coords_deltas
            self.target_offsets_deltas
            self.pred_offsets_deltas
            self.config.coords_loss_weight
            self.config.offset_loss_weight

        Ouputs:
            self.class_loss
            self.coords_loss
            self.side_class_loss
            self.offset_loss
            self.total_loss
            self.class_acc
            self.side_class_acc
        """
        with tf.name_scope("losses") as loss_scope:
            # Build the classification loss for detecting text or non-text
            # anchors
            with tf.variable_scope("class_loss") as scope:
                class_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pred_class_scores, labels=self.target_classes)
                self.class_loss = tf.reduce_mean(class_losses)
                self.class_acc = tf.reduce_mean(
                    tf.to_float(tf.equal(tf.argmax(self.pred_class_scores, 1),
                                self.target_classes)))

            # Build the regression loss for predicting vertical coordinates
            # of each anchors
            with tf.variable_scope("coords_loss") as scope:
                self.coords_loss = tf.reduce_mean(tf.reduce_sum(
                    util_ops.smooth_l1(self.pred_coords_deltas,
                                       self.target_coords_deltas), axis=1))

            # Build the classification loss for detecting containing-offet or
            # not-containing-offset anchors
            with tf.variable_scope("side_class_loss") as scope:
                side_class_losses =\
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.pred_side_class_scores, labels=self.target_side_classes)
                self.side_class_loss = tf.reduce_mean(side_class_losses)
                self.side_class_acc = tf.reduce_mean(
                    tf.to_float(tf.equal(
                        tf.argmax(self.pred_side_class_scores, 1),
                        self.target_side_classes)))

            # Build the regression loss for predicting horizontal coordinate
            # of each anchors
            with tf.variable_scope("offset_loss") as scope:
                self.offset_loss = tf.reduce_mean(
                    util_ops.smooth_l1(self.pred_offsets_deltas,
                                       self.target_offsets_deltas))

            # Build the total loss by weighted sum of above losses
            with tf.variable_scope("total_loss") as scope:
                self.total_loss = self.class_loss + \
                    self.config.coords_loss_weight * self.coords_loss + \
                    self.config.offset_loss_weight * self.offset_loss + \
                    self.config.side_class_loss_weight * self.side_class_loss

    def add_summaries(self):
        """Add loss summaries and histograms of variables to Tensorboard."""
        # Add summaries
        if self.mode != "inference":
            tf.summary.scalar("class_loss", self.class_loss)
            tf.summary.scalar("coords_loss", self.coords_loss)
            tf.summary.scalar("offset_loss", self.offset_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("class_acc", self.class_acc)
            tf.summary.scalar("side_class_acc", self.side_class_acc)
            tf.summary.scalar("side_class_loss", self.side_class_loss)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)


    def setup_vgg_initializer(self):
        """Sets up the function to restore vgg variables from checkpoint."""
        if self.mode != "inference":
            # Restore vgg variables only.
            saver = tf.train.Saver(self.vgg_pretrained_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring VGG variables from checkpoint file %s",
                                self.config.vgg_checkpoint_file)
                saver.restore(sess, self.config.vgg_checkpoint_file)

            self.init_fn = restore_fn

    def setup_checkpoint_loader(self, finetuning=False):
        """Sets up the function to restore Text detector from checkpoint."""
        self.model_pretrained_variables = slim.get_variables_to_restore(
            include=["vgg_16", "lstm", "fc", "rpn_cls", "rpn_bbox_delta",
                     "rpn_offset_delta", "rpn_side_cls"])

        if self.mode == "train":
            if finetuning:
                saver = tf.train.Saver(self.model_pretrained_variables)
            else:
                saver = tf.train.Saver()

            def restore_fn(sess):
                tf.logging.info(
                    "Loading model from checkpoint: %s",
                    self.config.model_checkpoint_file)
                saver.restore(sess, self.config.model_checkpoint_file)
                tf.logging.info("Successfully loadded checkpoint: %s",
                                os.path.basename(self.config.model_checkpoint_file))

            self.restore_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            dtype=tf.int32,
            collections=[tf.GraphKeys.GLOBAL_STEP,
                         tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_features()
        self.build_region_proposal_network()

        if self.mode == "inference":
            self.build_inference()
        else:
            self.build_batch_samples()
            self.build_targets()
            self.build_losses()
            self.setup_global_step()
            self.add_summaries()

        if self.config.vgg_checkpoint_file:
            self.setup_vgg_initializer()
        if self.config.model_checkpoint_file:
            self.setup_checkpoint_loader(self.finetuning)
