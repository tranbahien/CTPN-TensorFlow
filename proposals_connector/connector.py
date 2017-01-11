"""Connect all text proposals into text bounding boxes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from proposals_connector.inference_utils import clip_boxes, reduce_offsets
from proposals_connector.graph_builder import TextProposalGraphBuilder
from configuration import InferenceConfig as cfg


class TextProposalConnector(object):
    """Connect text proposals into text bouding boxes.
    """

    def __init__(self):
        """Basic setup"""
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        """Group text proposals into groups. Each group contains the text
        proposals belong into the same line of text.

        Args:
            text_proposals: Numpy array with shape [num_proposals, 4] contains
                the coodinates of each text proposal.
            scores: Numpy array with shape [num_proposals, 1] contains the
                predicted confidence of each text proposal.
            im_size: Numpy array with shape [2] contains the the size of image.
        """
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)

        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        """Interpolate the vertical coordinates based on data and 2 given
        horizontal coordinates.

        Args:
            X: A numpy array contains the horizontal coodinates
            Y: A numpy array contains the vertical coodinates
            x1: The horizontal coordinate of point 1
            x2: The horizontal coordinate of point 2
        """
        len(X) != 0

        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]

        p = np.poly1d(np.polyfit(X, Y, 1))

        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores,
                       offsets, offsets_scores, im_size):
        """Combine text proposals into text bounding boxes.

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
            tp_groups: List of list of index of text proposals belong to the
                same group.
        """
        # Group text proposals
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)

        # Initialize the list of text bounding boxes
        text_lines = []

        # Now, connect text proposals in each group
        for tp_indices in tp_groups:
            # Get the coordinates, offset, and scores of each proposal in
            # group
            text_line_boxes = text_proposals[list(tp_indices)]
            line_offsets, line_offsets_scores = offsets[list(
                tp_indices)], offsets_scores[list(tp_indices)]

            # Select offsets having highest score
            keep_inds = np.where(line_offsets_scores > cfg.OFFSET_MIN_SCORE)[0]
            line_offsets, line_offsets_scores = line_offsets[keep_inds], line_offsets_scores[keep_inds]

            # Get the predicted left and right coordinate of text line
            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            # Append x0, x1 into list of offsets and then reduce this list
            line_offsets = np.append(line_offsets, [[x0], [x1]], axis=0)
            line_offsets = reduce_offsets(line_offsets, cfg.OFFSET_REDUCE_GAP)

            # Find vertical coordiates of test lines
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5
            lt_y, rt_y = self.fit_y(
                text_line_boxes[:, 0],
                text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(
                text_line_boxes[:, 0],
                text_line_boxes[:, 3], x0 + offset, x1 - offset)
            y0 = min(lt_y, rt_y)
            y1 = max(lb_y, rb_y)

            # The score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            # Now, we split text line into text bonding boxe of words based on
            # offsets
            line_offsets = line_offsets.reshape(-1).tolist()
            for i, offset in enumerate(line_offsets):
                if i < len(line_offsets) - 1:
                    text_line = [line_offsets[i], y0,
                                 line_offsets[i + 1], y1, score]
                    text_lines.append(text_line)

        text_lines = np.array(text_lines, dtype=np.float32).reshape(-1, 5)

        # Refine the bouding boxes
        text_lines = clip_boxes(text_lines, im_size)

        return text_lines, tp_groups
