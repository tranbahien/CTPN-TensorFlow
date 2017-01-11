"""Build graph of text proposals for connecting them."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np

from configuration import InferenceConfig as cfg
from proposals_connector.inference_utils import Graph


class TextProposalGraphBuilder(object):
    """Build Text proposals into a graph for connecting.
    """
    def __init__(self):
        """Basic setup.
        """
        self.text_proposals = None
        self.scores         = None
        self.im_size        = None
        self.heights        = None
        self.boxes_table    = None

    def get_successions(self, index):
        """ Get succession vertice of a vertice. This means find text proposals
        belong to same group of the current text proposal.

        Args:
            index: Integer, the id of current vertice.

        Returns:
            results: List of integer contains the index of suitable text
            proposals.
        """
        # Get coodinates of the current text proposal
        box = self.text_proposals[index]
        results = []

        # Now, find the suitable proposals from left to right consequently.
        for left in range(int(box[0]) + 1,
                          min(int(box[0]) + cfg.MAX_HORIZONTAL_GAP + 1,
                              self.im_size[1])):
            # Get all the index of text proposals having the same left coodinate
            adj_box_indices = self.boxes_table[left]

            # Check each text proposals
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)

            if len(results) != 0:
                return results

        return results

    def get_precursors(self, index):
        """Get the previous (right-side) text proposals belonging to the same
        group of the current text proposals.

        Args:
            index: Integer, the id of current vertice.

        Returns:
            results: List of integer contains the index of suitable text
            proposals.
        """
        # Get coodinates of current text proposals
        box = self.text_proposals[index]
        results = []

        # Now, find the suitable proposals from right to left consequently.
        for left in range(int(box[0]) - 1,
                          max(int(box[0] - cfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            # Get all the index of text proposals having the same left coodinate
            adj_box_indices = self.boxes_table[left]

            # Check each text proposals
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)

            if len(results) != 0:
                return results

        return results

    def is_succession_node(self, index, succession_index):
        """Check if a provided text proposal is connected to the current text
        proposal.

        Args:
            index: Integer, the id of current text proposal.
            succession_index: Integer, the id of checking text proposal.

        Return: Boolean, the checked result.
        """
        # Get all right-side text proposals belonging to same group
        precursors = self.get_precursors(succession_index)

        # If checking text proposal having higher or equal score than
        # right-side text proposals, return True
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True

        return False

    def meet_v_iou(self, index1, index2):
        """Check 2 text_proposals whether they belong into same group. Fist,
            we check the vertical overlap and then check the size similarity.

        Args:
            index1: Integer, the index of the first text proposal.
            index2: Integer, the index of the second text proposal.

        Return: Boolean, the checked result.
        """
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1],
                     self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3],
                     self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= cfg.MIN_V_OVERLAPS and \
            size_similarity(index1, index2) >= cfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        """Build graph of text_propsals. This graph has
            num_proposals x num_proposals vertices, and vertices is connected
            if corresponding text proposals is also connected (belong in to
            a same text boxes).

        Args:
            text_proposals: A numpy array with shape [num_proposals, 4] contains
                the coordiates of each text proposal.
            scores: A numpy array with shape [num_proposals, 1] contains the
                predicted scores of each text propsals.
            im_size: A nunmpy array with shape [2] contains the size of image.

        Returns:
            graph: A Graph object
        """
        # Get text proposal and image information
        self.text_proposals = text_proposals
        self.scores         = scores
        self.im_size        = im_size
        self.heights        = text_proposals[:, 3] - text_proposals[:, 1] + 1

        # Construct a list of text proposals group. Specifically, text proposals
        # having the same left coodinates will be belong to a same group.
        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        # Initialize a graph of text proposals.
        graph = np.zeros(
            (text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        # Find the connections of each vertices in graph.
        for index, box in enumerate(text_proposals):
            # Get succession vertices (connected proposals) of current vertice
            successions = self.get_successions(index)

            if len(successions) == 0:
                continue

            # Choose the succession having highest score
            succession_index = successions[np.argmax(scores[successions])]

            # Update graph
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if
                # multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True

        return Graph(graph)
