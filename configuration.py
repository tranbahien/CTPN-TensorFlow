"""Text detection model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None

        # Image format ("jpeg" or "png").
        self.image_format = "jpeg"

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard      = 2
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads    = 1
        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads      = 4

        # Batch size must be equal 1
        self.batch_size             = 1
        # Size of batch of anchors
        self.batch_anchor_size      = 128
        self.batch_side_anchor_size = 32

        # The weight of losses
        self.coords_loss_weight     = 2.
        self.offset_loss_weight     = 3.
        self.side_class_loss_weight = 2.

        # Dimensions of VGG input images (only use in inference mode)
        self.image_width = 1200

        # File containing an VGG checkpoint to initialize the variables
        # of the VGG model. Must be provided when starting training for the
        # first time.
        self.vgg_checkpoint_file = None

        # File containing an trained text detector checkpoint to initialize the
        # variables of model.
        self.model_checkpoint_file = ""

        # Scale used to initialize model variables.
        self.initializer_stddev = 0.01
        self.initializer_scale  = 0.008

        # Number of layers of LSTM
        self.num_lstm_layers  = 1
        # Number of units of LSTM cell
        self.num_lstm_units   = 256
        # Number of units fully connected layer
        self.num_hidden_units = 1024

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.5
        # If < 1.0, the dropout keep probability applied to FC variables.
        self.fc_dropout_keep_prob   = 0.5

        # Coefficient for L2 weight regularization.
        self.vgg_weight_decay = 0.0004


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 183

        # Optimizer for training the model.
        self.optimizer = "Adam"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate      = 3e-5
        self.learning_rate_decay_factor = 0.7
        self.num_epochs_per_decay       = 10

        # Learning rate when finetuning the VGG 16 parameters.
        self.train_vgg_learning_rate = 8e-6

        # If not None, clip gradients to this value.
        self.clip_gradients = 1.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5

class InferenceConfig:
    """Configuration for doing inference"""
    LINE_MIN_SCORE            = 0.8
    TEXT_PROPOSALS_MIN_SCORE  = 0.7
    OFFSET_MIN_SCORE          = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MAX_HORIZONTAL_GAP        = 50
    OFFSET_REDUCE_GAP         = 20.
    TEXT_LINE_NMS_THRESH      = 0.3
    MIN_NUM_PROPOSALS         = 2
    MIN_RATIO                 = 0.7
    MIN_V_OVERLAPS            = 0.7
    MIN_SIZE_SIM              = 0.7
    TEXT_PROPOSALS_WIDTH      = 16
