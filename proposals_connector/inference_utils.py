"""Utility function for inference step."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Graph(object):
    """Object represents the graph containing the connected text proposals.
    """
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        """Refine the original graph having num_proposals x num_proposals
            vertices into a list of group of connected text proposals.

        Args:
            self.graph

        Returns:
            sub_graphs: List of list of indexs of connected text proposals
        """
        sub_graphs = []
        for index in xrange(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs


def threshold(coords, min_, max_):
    """Get the suitable value of each coordinates based on the provided minimum
        value and maximum value.

    Args:
        coords: A numpy array contains the clipped coordinates.
        min_: Integer, the minimum threshold of a coodinate.
        max_: Integer, the minimum threshold of a coodinate.

    Returns: Clipped coordinates based on the thresholds.
    """
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries.

    Args:
        boxes: A numpy array with shape [num_bboxes, 4] contains the
            coordinates of each boxes.
        im_shape: A numpy array with shape [2] contains the size of the image.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0] - 1)

    return boxes


def normalize(data):
    """Normalize predicted scores into range of [min, max] of each dimension.

    Args:
        data: numpy array contains the data need to be normalized.

    Returns: Normalized data.
    """
    if data.shape[0] == 0:
        return data

    max_ = data.max()
    min_ = data.min()

    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_


def reduce_offsets(offsets, min_dist=20.):
    """Reduce the neighbor offsets having distance less than min_dist into a
        offsets by mean calculation.

    Args:
        offsets: A numpy array contains the offsets.
        min_dist: Float, the minimum distance for reducing.

    Returns:
        chosen_offsets: A numpy array contains the reduced offsets.
    """
    offset_list = list(offsets)
    offset_list.sort()

    chosen_offsets = []
    for i, e in enumerate(offset_list):
        if i == 0:
            chosen_offsets.append(e)
        elif abs(e - chosen_offsets[-1] <= min_dist):
            chosen_offsets[-1] = abs(e + chosen_offsets[-1]) / 2.
        else:
            chosen_offsets.append(e)

    chosen_offsets = np.array(chosen_offsets, dtype=np.float32).reshape(-1, 1)

    return chosen_offsets
