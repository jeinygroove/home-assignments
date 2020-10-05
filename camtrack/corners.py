# ! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class CornersHelper:
    """
    Class that has methods for detecting and tracking corners
    """

    def __init__(self, h, w):
        self.corners = []
        self.sizes = []
        self.ids = []
        self.max_id = 0
        self.frame_sizes = (h, w)

        self.WIN_SIZE = 20
        self.MAX_LEVELS = 2
        self.MAX_CORNERS = 5000
        self.MIN_DIST = 10
        self.QUALITY_LEVEL = 0.01
        self.BLOCK_SIZE = 7

    def create_mask(self):
        """
        Creates a mask on which we mark existing corners with their size
        to avoid finding too close corners.
        :return np array of the size of frame filled with 0 or/and 255
        """
        mask = np.full(self.frame_sizes, 255, dtype=np.uint8)
        for (x, y), radius in zip(self.corners, self.sizes):
            mask = cv2.circle(mask, (np.round(x).astype(int), np.round(y).astype(int)), radius, thickness=-1, color=0)
        return mask

    @staticmethod
    def count_condition(st1, st2):
        """
        Count condition using status that can be used to remove untracked corners
        :param st1: array with 0/1 for untracked/tracked corners
        :param st2: array with 0/1 for untracked/tracked corners in backtracking
        :return array of the same size as statuses
        """
        statuses1 = st1.ravel()
        statuses2 = st2.ravel()
        conditions = (statuses1 == 1) & (statuses2 == 1)
        return conditions

    def detect_and_track_corners(self, prev_frame, curr_frame):
        """
        For prev_frame and curr_frame track self.corners
        and find new ones.
        :param prev_frame: previous image
        :param curr_frame: current image
        :return FrameCorners with tracked and found corners
        """
        if len(self.corners) > 0:
            # find new corners with LK using several levels of pyramids
            new_corners, st1, _ = \
                cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame,
                                         np.asarray(self.corners, dtype=np.float32),
                                         None,
                                         maxLevel=self.MAX_LEVELS,
                                         winSize=(self.WIN_SIZE, self.WIN_SIZE),
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001),
                                         minEigThreshold=1e-4)
            backtracking_corners, st2, _ = \
                cv2.calcOpticalFlowPyrLK(curr_frame, prev_frame,
                                         np.asarray(new_corners, dtype=np.float32),
                                         None,
                                         maxLevel=self.MAX_LEVELS,
                                         winSize=(self.WIN_SIZE, self.WIN_SIZE),
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001),
                                         minEigThreshold=1e-4)

            dist = abs(self.corners - backtracking_corners).reshape(-1, 2).max(-1)
            good_tracked = dist < 0.3

            # remove untracked corners and badly tracked corners
            condition = self.count_condition(st1, st2) & good_tracked
            self.corners = new_corners[condition].tolist()
            self.sizes = np.asarray(self.sizes)[condition].tolist()
            self.ids = np.asarray(self.ids)[condition].tolist()

        # find pyramid for current frame
        levels, frame_pyramid = cv2.buildOpticalFlowPyramid(curr_frame,
                                                            (self.WIN_SIZE, self.WIN_SIZE),
                                                            self.MAX_LEVELS, None, False)
        mask = self.create_mask()
        for level, frame_level in enumerate(frame_pyramid):
            if len(self.corners) >= self.MAX_CORNERS:
                break
            # find new corners, since we bounded by MAX_CORNERS,
            # then there's no need to search for more than
            # self.MAX_CORNERS - len(self.corners)
            new_corners = cv2.goodFeaturesToTrack(
                frame_level,
                maxCorners=self.MAX_CORNERS - len(self.corners),
                qualityLevel=0.001 if prev_frame is None else self.QUALITY_LEVEL,
                minDistance=self.MIN_DIST * (np.round(2 ** level).astype(int)),
                blockSize=self.BLOCK_SIZE,
                mask=mask
            )

            # add new corners if there are less than self.MAX_CORNERS in total
            # and if they're not close to already added corners
            if new_corners is not None:
                # reshape corners for FrameCorners
                new_corners = new_corners.reshape(-1, 2).astype(np.float32)
                cur_size = self.BLOCK_SIZE
                for (x, y) in new_corners:
                    if len(self.corners) >= self.MAX_CORNERS:
                        break

                    real_x = x * 2 ** level
                    real_y = y * 2 ** level
                    x = np.round(x).astype(int)
                    y = np.round(y).astype(int)
                    if mask[y, x] != 0:
                        self.corners.append((real_x, real_y))
                        self.sizes.append(cur_size * 2 ** level)
                        self.ids.append(self.max_id)
                        self.max_id += 1

                    mask = cv2.circle(mask, (x, y), cur_size, thickness=-1, color=0)
            # update mask for the next level of pyramid
            mask = cv2.pyrDown(mask).astype(np.uint8)

        return FrameCorners(np.array(self.ids), np.array(self.corners), np.array(self.sizes))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    h, w = frame_sequence.frame_shape[:2]
    corners_helper = CornersHelper(h, w)

    prev_frame = None
    for index, frame in enumerate(frame_sequence):
        curr_frame = (frame * 255).astype(np.uint8)
        frame_corners = corners_helper.detect_and_track_corners(prev_frame, curr_frame)
        builder.set_corners_at_frame(index, frame_corners)
        prev_frame = curr_frame


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
