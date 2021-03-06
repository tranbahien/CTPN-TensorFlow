"""Connect text proposals generated by model into text bounding boxes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np

from libs.cpu_nms import cpu_nms as nms
from proposals_connector.connector import TextProposalConnector
from proposals_connector.inference_utils import normalize
from configuration import InferenceConfig as cfg


def filter_boxes(boxes):
    """Filter bounding boxes having low confidence and small width.

    Args:
        boxes: Numpy array with shape [num_boxes, 4] contains the coordinates
            of each bounding box.

    Returns: the indexs of chosen bounding boxes.
    """
    heights = boxes[:, 3] - boxes[:, 1] + 1
    widths  = boxes[:, 2] - boxes[:, 0] + 1
    scores  = boxes[:, -1]

    return np.where(
        (widths / heights > cfg.MIN_RATIO) &\
        (scores > cfg.LINE_MIN_SCORE) &\
        (widths > (cfg.TEXT_PROPOSALS_WIDTH * cfg.MIN_NUM_PROPOSALS)))[0]


def connect_proposals(text_proposals, scores, offsets,
                      offset_scores, img_shape):
    """Connect all text proposals into text bounding boxes.

    Args:
        text_proposals: Numpy array with shape [num_proposals, 4] contains
            the coodinates of each text proposal.
        scores: Numpy array with shape [num_proposals, 1] contains the
            predicted confidence of each text proposal.
        offsets_scores: Numpy array with shape [num_proposals, 1] contains
            the offset of each text proposal.
        offsets_scores: Numpy array with shape [num_proposals, 1] contains
            the predicted confidence of each offset.
        im_size: Numpy array with shape [2] contains the the size of image.

    Returns:
        text_lines: Numpy array with shape [num_bboxes, 5] contains the
            coordinates and score of each bounding box.
    """
    # Keep only proposals having high confidence
    keep_inds = np.where(scores > cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
    text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]
    offsets, offset_scores = offsets[keep_inds], offset_scores[keep_inds]

    # Sort text proposals
    sorted_indices = np.argsort(scores.ravel())[::-1]
    text_proposals, scores = text_proposals[sorted_indices],\
        scores[sorted_indices],
    offsets, offset_scores = offsets[sorted_indices],\
        offset_scores[sorted_indices]
    text_proposals, scores, offsets, offset_scores = text_proposals[sorted_indices], scores[
        sorted_indices], offsets[sorted_indices], offset_scores[sorted_indices]

    # Apply non-maximum supression for filtering the overlap text proposals
    keep_inds = nms(np.hstack((text_proposals, scores)),
                    cfg.TEXT_PROPOSALS_NMS_THRESH)
    text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]
    offsets, offset_scores = offsets[keep_inds], offset_scores[keep_inds]

    # Normalize scores
    scores = normalize(scores)

    # Connect text proposals into text boxes
    connector = TextProposalConnector()
    text_lines, _ = connector.get_text_lines(
        text_proposals, scores, offsets, offset_scores, img_shape)

    # Filter text bounding boxes having low confidence or small width
    keep_inds = filter_boxes(text_lines)
    text_lines = text_lines[keep_inds]

    # Use non-maximum supression for filter overlap bounding boxes
    if text_lines.shape[0] != 0:
        keep_inds = nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
        text_lines = text_lines[keep_inds]

    return text_lines
