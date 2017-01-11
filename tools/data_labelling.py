"""Tool for generate labels for text detection model."""

import math
import numpy as np

from bbox import bbox_overlaps
from anchor import locate_anchors


class AnchorLabeller(object):
    """Tool used to generate anchors and corresponding labels for text detection
        model."""

    def __init__(self, num_anchors=10, shift=16.0, offset_dist_threshold=10.,
                 positive_threshold=0.7, negative_threshold=0.5,
                 coord_overlap_threshold=0.5, ver_overlap_threshold=0.5):
        """Basic setup.

        Args:
            num_anchors: the number of generated anchor at each point on
                feature map.
            shift: the width of each anchor
            offset_dist_threshold: the threshold of distance to text offset for
                determining side anchors.
            positive_threshold: the IoU threhsold for determining positive
                anchors.
            negative_threshold: the IoU threhsold for determining negative
                anchors.
            ver_overlap_threshold:the vertical IoU threshold for determining
                side anchors.
        """
        self.num_anchors             = num_anchors
        self.shift                   = shift
        self.offset_dist_threshold   = offset_dist_threshold
        self.positive_threshold      = positive_threshold
        self.negative_threshold      = negative_threshold
        self.coord_overlap_threshold = coord_overlap_threshold
        self.ver_overlap_threshold   = ver_overlap_threshold

    def generate_anchors(self, img_width, img_height, gt_bboxes):
        """Generate anchors corresponding an image and create corresponding
            labels for each anchor. Specifically, we first generate anchors
            and determine whether an anchor is either positive or negative
            anchors (has text or doesn't have text) and the vertical coordinates
            (center and height) of ground-truth anchors. After that, we
            determine whether an anchor is either side anchor or not (contains
            the offset bounding box or not) and calculate the offset for each
            side anchor.

        Args:
            img_width: the width of image.
            img_height: the height of image.
            gt_bboxes: a numpy array with shape [num_text_boxes, 4] contains
                the bounding box of texts.

        Returns:
            data: A dictionary contains the anchors and labels of data.
        """
        # Estimate the size of feature map created by Convolutional neural
        # network (VGG-16)
        feat_map_size = [int(math.ceil(img_width / self.shift)),
                         int(math.ceil(img_height / self.shift))]

        # Generate anchors
        anchors = locate_anchors(feat_map_size, 16)

        # Determine the coordinates of ground-truth anchors based on anchors
        # and the ground-truth bounding boxes
        gt_anchors = self.divide_gt_bboxes(gt_bboxes)

        # Determine whether an anchors is positive or negative (contains text
        # or non)
        cls_anchors, _ = self.label_anchors(anchors, gt_anchors,
                                            pos_threshold=0.7, neg_threshold=0.5)

        # Calculate the ground-truth vertical coordinates of positive anchors.
        coord_ids, coords = self.calculate_gt_coordinates(anchors, gt_anchors)

        # Determine whether anchor contains offset of text bounding box or non
        side_classes, side_ids, offsets = self.calculate_offsets(
            anchors, cls_anchors, gt_bboxes, self.ver_overlap_threshold,
            self.offset_dist_threshold)

        data = {
            "anchors": anchors,
            "classes": cls_anchors,
            "coord_ids": coord_ids,
            "coords": coords,
            "side_classes": side_classes,
            "side_ids": side_ids,
            "offsets": offsets
        }

        return data

    def divide_gt_bboxes(self, gt_bboxes, shift=16.):
        """Calculate ground-truth bboxes based on ground-truth text bounding
        boxes. The width of these anchors is equal to shift distance.

        Args:
            gt_bboxes: a numpy array with shape [num_bboxes, 4] contains the
                coodinates of each text bounding boxes.
            shift: the width size of each anchor.

        Returns:
            gt_anchors: A numpy array with shape [num_gt_anchors, 4] contains
                the coordinates of each ground-truth anchor.
        """
        anchors_list = []
        for left, top, right, bottom in gt_bboxes:
            anchor_ids = np.arange(int(math.floor(1. * left / shift)),
                                   int(math.ceil(1. * right / shift)))
            anchors = np.zeros((len(anchor_ids), 4))
            anchors[:, [1, 3]] = (top, bottom)
            anchors[:, 0] = anchor_ids * shift
            anchors[:, 2] = (anchor_ids + 1) * shift
            anchors_list.append(anchors)

        gt_anchors = np.concatenate(anchors_list, axis=0)

        return gt_anchors

    def label_anchors(self, anchors, gt_anchors,
                      pos_threshold=0.7, neg_threshold=0.5):
        """Label each anchor (text or non-text).

        Args:
            anchors: A numpy array with shape [num_anchors, 4] contains the
                coordinates of each anchor.
            gt_anchors: A numpy array with shape [num_gt_anchors, 4] contains
                the coordinates of each ground-truth anchor.
            pos_threshold: A IoU threshold for determining an anchor is
                positive.
            neg_threshold: A IoU threshold for determining an anchor is
                negative.

        Returns:
            cls_anchors: A numpy array with shape [num_anchors] contains
                the class of each anchor.
            pos_anchors: A numpy array with shape [num_pos_anchors, 4] contains
                the coordinates of each positive anchor.
        """
        # Array containing the label for each anchor
        cls_anchors = np.ones((anchors.shape[0]), dtype=np.int) * (-1)

        # Calculate the IoU between the anchors and the ground truth anchors
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_anchors, dtype=np.float))

        # Labeling anchors
        # i. Negative anchors (< 0.5 IoU overlap with all GT boxes)
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(anchors.shape[0]), argmax_overlaps]
        cls_anchors[max_overlaps < neg_threshold] = 0

        # ii. The anchors with the highest IoU overlap with GT boxes.
        highest_argmax_overlaps = overlaps.argmax(axis=0)
        cls_anchors[highest_argmax_overlaps] = 1
        highest_argmax_overlaps = np.array([highest_argmax_overlaps,
                                            np.arange(len(highest_argmax_overlaps))])

        # iii. Anchors that have > threhsold IoU overlap with any GT box
        valid_argmax_overlaps = np.where(overlaps > pos_threshold)
        cls_anchors[valid_argmax_overlaps[0]] = 1

        mask = np.in1d(highest_argmax_overlaps[0], valid_argmax_overlaps[0])
        new_anchors_id = np.where(~mask)[0]
        if len(np.where(~mask)[0]) > 0:
            pos_anchors = (np.append(valid_argmax_overlaps[0],
                                     highest_argmax_overlaps[0][new_anchors_id]),
                           np.append(valid_argmax_overlaps[1],
                                     highest_argmax_overlaps[1][new_anchors_id]))
        else:
            pos_anchors = valid_argmax_overlaps

        return cls_anchors, pos_anchors

    def calculate_gt_coordinates(self, anchors, gt_anchors):
        """Calculate the ground-truth vertical coodinates of each coordinating
            anchors. Coordinating anchors are positive anchors and anchors
            having IoU > 0.5 with any ground-truth anchor.

        Args:
            anchors: A numpy array with shape [num_anchors, 4] contains the
                coordinates of each anchor.
            gt_anchors: A numpy array with shape [num_gt_anchors, 4] contains
                the coordinates of each ground-truth anchor.

        Returns:
            valid_anchors: A numpy array contains the index of coordinating
                anchors.
            valid_coords: A numpy array with shape [num_valid_anchors, 2]
                contains the vertical coordinates of valid anchors.
        """
        # Get the positive anchors
        cls_anchors_1, pos_anchors_1 = self.label_anchors(anchors, gt_anchors,
                                                          pos_threshold=0.7)

        # Get the anchors having IoU > 0.5 overlap with a ground truth proposal
        cls_anchors_2, pos_anchors_2 = self.label_anchors(anchors, gt_anchors,
                                                          pos_threshold=0.5)

        # Select the coordinate anchors (valid anchors)
        mask = np.in1d(pos_anchors_1[0], pos_anchors_2[0])
        new_anchors_id = np.where(~mask)[0]
        if len(np.where(~mask)[0]) > 0:
            pos_anchors = (np.append(pos_anchors_1[0],
                                     pos_anchors_2[0][new_anchors_id]),
                           np.append(pos_anchors_1[1],
                                     pos_anchors_2[1][new_anchors_id]))
        else:
            pos_anchors = pos_anchors_1

        num_pos_anchors = len(pos_anchors[0])
        valid_coords = np.zeros((num_pos_anchors, 2))

        # Compuute the vertical coodinates (height and center) of each valid
        # anchors
        gt_valid_anchors = gt_anchors[pos_anchors[1]]
        height = gt_valid_anchors[:, 3] - gt_valid_anchors[:, 1] + 1
        center = (gt_valid_anchors[:, 1] + gt_valid_anchors[:, 3]) / 2.
        valid_coords = np.stack((height, center), axis=-1)

        return pos_anchors[0], valid_coords

    def compute_vertical_overlap(self, anchors, gt_anchors):
        """Compute the vertical overlap between anchors and ground-truth
            anchors.

        Args:
            anchors: A numpy array with shape [num_anchors, 4] contains the
                coordinates of each anchor.
            gt_anchors: A numpy array with shape [num_gt_anchors, 4] contains
                the coordinates of each ground-truth anchor.

        Returns:
            overlaps: A numpy array with shape [num_anchors, num_gt_anchors]
                contains the vertical overlap value between each anchor and
                each ground-truth anchor.
        """
        N = anchors.shape[0]
        K = gt_anchors.shape[0]
        overlaps = np.zeros((N, K))

        for n in range(N):
            overlap_heights = np.maximum(
                (np.minimum(anchors[n, 3], gt_anchors[:, 3]) -
                 np.maximum(anchors[n, 1], gt_anchors[:, 1])),
                0)
            total_heights = np.maximum(anchors[n, 3], gt_anchors[:, 3]) - \
                np.minimum(anchors[n, 1], gt_anchors[:, 1]) + 1
            overlaps[n, :] = overlap_heights / total_heights

        return overlaps

    def calculate_offsets(self, anchors, cls_anchors, gt_bboxes,
                          ver_threshold=0.5, hor_threshold=10):
        """Label side anchors containing offset of bounding box.

        Args:
            anchors: A numpy array with shape [num_anchors, 4] contains the
                coordinates of each anchor.
            cls_anchors: A numpy array with shape [num_anchors, 1] constains the
                class (text or non-text) of each anchor.
            gt_bboxes: A numpy array with shape [num_bboxes, 4] contains the
                coodinates of ground-truth bounding box.
            ver_threshold: Float, threshold of vertical overlapping between
                anchors and ground-truth bounding box.
            hor_threshold: Float, threshold of vertical distance between
                ground-truth bounding box and the side anchors.

        Returns:
            cls_side_anchors: A numpy array with shape [num_anchors, 1] contains
                the side class (containing-offset or non-containing-offset)
                of each anchor.
            side_anchors_ids: A numpy array with shape [num_side_anchors]
                contains the index of side anchors.
            offsets: A numpy array with shape [num_side_anchors] contains
                the offset of each side anchors.
        """
        # Array containing the side-classes of each anchor
        cls_side_anchors = np.ones((anchors.shape[0]), dtype=np.int) * (-1)

        # Assign positive anchors into 0-class
        cls_side_anchors[np.where(cls_anchors == 1)] = 0

        # Find vertical anchors defined as anchors having vertical overlap
        # with any text bounding box higher than a threshold
        vertical_overlaps = self.compute_vertical_overlap(anchors, gt_bboxes)
        ver_anchors_ids, bbox_ids = np.where(vertical_overlaps > ver_threshold)
        ver_anchors = anchors[ver_anchors_ids, :]

        # Find side anchors
        side_anchors_ids = []
        for i, gt_bbox in enumerate(gt_bboxes):
            pos_mask = np.where(bbox_ids == i)[0]
            centers = ver_anchors[:, 0] + (ver_anchors[:, 2] -
                                           ver_anchors[:, 0]) / 2.
            left_ids = np.where(np.absolute(centers - gt_bbox[0]) <
                                hor_threshold)[0]
            right_ids = np.where(np.absolute(centers - gt_bbox[2]) <
                                 hor_threshold)[0]
            left_ids = np.intersect1d(pos_mask, left_ids).tolist()
            right_ids = np.intersect1d(pos_mask, right_ids).tolist()
            side_anchors_ids.extend(left_ids)
            side_anchors_ids.extend(right_ids)

        side_anchors_ids = list(set(side_anchors_ids))

        # Calculate offset for each side anchors
        offsets = []
        for i in side_anchors_ids:
            center = ver_anchors[i, 0] + (ver_anchors[i, 2] -
                                          ver_anchors[i, 0]) / 2.
            off_id = np.argmin(np.minimum(np.absolute(center - gt_bboxes[:, 0]),
                                          np.absolute(center - gt_bboxes[:, 2])))
            if (abs(center - gt_bboxes[off_id, 0]) <
                    abs(center - gt_bboxes[off_id, 2])):
                offset = gt_bboxes[off_id, 0]
            else:
                offset = gt_bboxes[off_id, 2]

            offsets.append(offset)

        # Label side anchors
        side_anchors_ids = ver_anchors_ids[side_anchors_ids]
        cls_side_anchors[side_anchors_ids] = 1
        side_anchors_ids = np.array(side_anchors_ids, dtype=np.int32)

        return cls_side_anchors, side_anchors_ids, offsets
