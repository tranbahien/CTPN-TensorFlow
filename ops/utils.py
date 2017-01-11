"""Utility ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from tools.anchor import locate_anchors as py_locate_anchors
from proposals_connector.connect_proposals import \
    connect_proposals as py_connect_proposals


def get_patches(images, patch_dim):
    """Extract patches from images and put them in "depth" output dimension.

    Args:
        images: 4-D Tensor with shape [batch, in_rows, in_cols, depth].
        patch_dim: An integer number representing the dimension of patch.

    Returns:
        patches: 3-D Tensor with shape
            [batch, out_rows, in_cols, patch_dim**2 * depth].
    """
    in_filters = images.get_shape().as_list()[3]
    out_filters = patch_dim**2 * in_filters
    kernel = tf.constant(np.eye(out_filters).
                         reshape(patch_dim, patch_dim,
                                 in_filters, out_filters),
                         tf.float32)

    patches = tf.nn.conv2d(images, kernel,
                           [1, 1, 1, 1], "SAME")

    return patches


def generate_anchors(feat_map_size, feat_stride, anchor_num):
    """Generate anchors on feature map of an image.

    Args:
        feat_map_size: Int32 1-D Tensor containing the sizes of feature map
            [feat_map_height, feat_map_width].
        feat_stride: Int32 Constant Tensor containing the stride of kernel on
            feature maps.
        anchor_num: Int32 Constant Tensor containing the number of generated
            anchors.

    Returns:
        anchors: 3-D Tensor with shape [height, width, 4] containing the
            generated anchors.
    """
    anchors = tf.py_func(py_locate_anchors,
                         [feat_map_size, feat_stride, anchor_num],
                         tf.int32, name="generate_anchors")

    anchors = tf.to_float(anchors)

    return anchors


def apply_vertical_deltas_to_anchors(boxes_delta, anchors):
    """Compute coordinates of boxes based on its predicted relative coordianates
        (boxes_deltas) and anchors.

    Args:
        boxes_delta: Float 2-D Tensor, [num_anchors, 2], containing
            the boxes deltas.
        anchors: Float 2-D Tensor, [num_anchors, 4] containing the
            coordinates of anchors.

    Returns:
        boxes: Float 2-D Tensor, [num_anchors, 4], containing the real
            coordinates of boxes.
    """
    anchor_y_ctr = (anchors[:, 1] + anchors[:, 3]) / 2.
    anchor_h = anchors[:, 3] - anchors[:, 1] + 1
    vertical_heights = tf.exp(boxes_delta[:, 1]) * anchor_h
    vertical_centers = boxes_delta[:, 0] * anchor_h + anchor_y_ctr

    vertical_tops = vertical_centers - vertical_heights / 2.
    vertical_bottoms = vertical_centers + vertical_heights / 2.

    boxes = tf.stack([anchors[:, 0], vertical_tops,
                     anchors[:, 2], vertical_bottoms ], axis=1)

    return boxes


def apply_horizontal_deltas_to_anchors(offset_delta, anchors):
    """Compute horizontal coordinates of boxes based on its predicted
        relative coordianates (boxes_deltas) and anchors.

    Args:
        offset_delta: Float 2-D Tensor, [num_anchors, 1], containing
            the boxes deltas.
        anchors: Float 2-D Tensor, [num_anchors, 4] containing the
            coordinates of anchors.

    Returns:
        offset: Float 2-D Tensor, [num_anchors, 1], containing the real
            offset of boxes.
    """
    anchor_x_ctr = (anchors[:, 0] + anchors[:, 2]) / 2.
    anchor_w = anchors[:, 2] - anchors[:, 0] + 1
    offsets = offset_delta[:, 0] * anchor_w + anchor_x_ctr

    return offsets


def convert_to_vertical_deltas(coords, anchors):
    """Compute relative coordianates of boxes (deltas) based on anchors.

    Args:
        coords: Float 2-D Tensor, [num_boxes, 2], containing the real
            coordinates of boxes.
        anchors: Float 2-D Tensor, [num_boxes, 4] containing the
            coordinates of anchors.

    Returns:
        boxes_deltas: Float 2-D Tensor, [num_boxes, 2], containing
            the boxes deltas.
    """
    coords_y_ctr = coords[:, 1]
    coords_h = coords[:, 0]
    anchor_y_ctr = (anchors[:, 1] + anchors[:, 3]) / 2.
    anchor_h = anchors[:, 3] - anchors[:, 1] + 1

    vertical_ctr = (coords_y_ctr - anchor_y_ctr) / anchor_h
    vertical_h = tf.log(coords_h / anchor_h)

    boxes_deltas = tf.stack([vertical_ctr, vertical_h], axis=1)

    return boxes_deltas


def convert_to_horizontal_deltas(offsets, anchors):
    """Compute relative coordianates of boxes (deltas) based on anchors.

    Args:
        offsets: Float 2-D Tensor, [num_offsets, 1], containing offsets.
        anchors: Float 2-D Tensor, [num_offsets, 4] containing the
            coordinates of anchors.

    Returns:
        horizontal_ctr: Float 2-D Tensor, [num_offsets, 1], containing
            the offsets deltas.
    """
    x_sides = tf.reshape(offsets, [-1])
    anchor_x_ctr = (anchors[:, 0] + anchors[:, 2]) / 2.
    anchor_w = anchors[:, 2] - anchors[:, 0] + 1
    horizontal_ctr = (x_sides - anchor_x_ctr) / anchor_w
    horizontal_ctr = tf.reshape(horizontal_ctr, [-1, 1])

    return horizontal_ctr


def convert_box_to_veritical_coords(boxes):
    """Convert the coordinates of proposal boxes to vertical coordinates
        including the height and y-axis center of the proposal boxes.

    Args:
        boxes: Float 3-D Tensor, [height, width, 4], containing the coordinates
            of the proposal boxes.

    Returns:
        vertical_coords: Float 3-D Tensor, [height, width, 2] containing the
            vertical coordinates of the proposal boxes.
    """
    y_ctr = (boxes[:, 1] + boxes[:, 3]) / 2.
    h = boxes[:, 3] - boxes[:, 1] + 1

    vertical_coords = tf.stack([y_ctr, h], axis=1)

    return vertical_coords

def rescale_bboxes(bboxes, old_h, old_w, new_h, new_w):
    """Rescale bouding boxes being suitable for the original image.

    Args:
        bboxes: A Float32 Tensor with shape [num_bboxes, 5] contains the
            bounding boxes corresponding to the scaled image.
        old_h: A Int32 scalar Tensor contains the height of the original image.
        old_w: A Int32 scalar Tensor contains the width of the original image.
        new_h: A Int32 scalar Tensor contains the height of the scaled image.
        new_w: A Int32 scalar Tensor contains the width of the scaled image.

    Returns:
        scaled_bboxes: A Float32 Tensor with shape [num_bboxes, 5] contains the
            bounding boxes corresponding to the original image.
    """
    # Scale coordinates (left, top, right, bottom) of bounding boxes
    l = bboxes[:, 0] * (tf.to_float(old_w) / tf.to_float(new_w))
    t = bboxes[:, 1] * (tf.to_float(old_h) / tf.to_float(new_h))
    r = bboxes[:, 2] * (tf.to_float(old_w) / tf.to_float(new_w))
    b = bboxes[:, 3] * (tf.to_float(old_h) / tf.to_float(new_h))
    scores = bboxes[:, 4]

    # Build the bounding boxes corresponding to the original image
    scaled_bboxes = tf.stack([l, t, r, b, scores], axis=1)

    return scaled_bboxes

def connect_proposals(proposals, scores, offsets,
                      offset_scores, img_shape):
    """Connect text proposals returned by neural nets to build the final text
        bounding boxes. This function is a TensorFlow operation built from
        a python function.

    Args:
        proposals: A Float32 Tensor with shape [num_proposals, 4] contains the
            coordinates of each text proposal.
        scores: A Float32 Tensor with shape [num_proposals, 1] contains the
            predicted confidence of each text proposal.
        offsets: A Float32 Tensor with shape [num_proposals, 1] contains the
            horizontal offset of each text proposal.
        offset_scores: A Float32 Tensor with shape [num_proposals, 1] contains
            the predicted confidence of each text proposal.
    """
    text_lines = tf.py_func(py_connect_proposals,
                        [proposals, scores, offsets, offset_scores, img_shape],
                        tf.float32)

    return text_lines


def get_intersect(x, y):
    """Find the intersection of 2 1-D arrays.

    Args:
        x: The first Int 1-D Tensor.
        y: The second Int 1-D Tensor.

    Returns:
        inter_ids: The Int 1_D Tensor containing intersection ids.
        inter_values: The Int 1_D Tensor containing intersection elements.
    """
    def py_get_intersect(x, y):
        inter_mask = np.in1d(x, y)
        inter_values = x[inter_mask]
        inter_ids = np.where(inter_mask == True)[0]

        return inter_ids, inter_values

    x = tf.to_int64(x)
    y = tf.to_int64(y)
    inter_ids, inter_values = tf.py_func(py_get_intersect,
                                         [x, y], [tf.int64, tf.int64],
                                         name="get_intersect")
    inter_ids = tf.to_int32(inter_ids)
    inter_values = tf.to_int32(inter_values)

    return inter_ids, inter_values


def smooth_l1(pred, targets):
    """Calculate the smooth L1 value for loss of regression models.
        smooth_l1(x) = 0.5 * x^2, if |x| < 1
                       |x| - 0.5, otherwise
    Args:
        pred: A float32 tensor contains the predicted values.
        targets: A float32 tensor contains the target values.
    """
    term = pred - targets
    sign = tf.cast(tf.less(tf.abs(term), 1.0), tf.float32)
    pos_result = 0.5 * (term * term)
    neg_result = tf.abs(term) - 0.5
    result = pos_result * sign + neg_result * tf.abs(1 - sign)

    return result
