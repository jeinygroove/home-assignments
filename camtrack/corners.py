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

        self.WIN_SIZE = 15
        self.MAX_LEVELS = 3
        self.MAX_CORNERS = 10000
        self.MIN_DIST = 7
        self.QUALITY_LEVEL = 0.05
        self.BLOCK_SIZE = 3

    def create_mask(self):
        mask = np.full(self.frame_sizes, 255, dtype=np.uint8)
        for (x, y), radius in zip(self.corners, self.sizes):
            mask = cv2.circle(mask, (np.round(x).astype(int), np.round(y).astype(int)), radius, thickness=-1, color=0)
        return mask

    @staticmethod
    def count_condition(statuses):
        statuses = statuses.ravel()
        conditions = (statuses == 1)
        return conditions

    def detect_and_track_corners(self, prev_frame, curr_frame):
        prev_frame_pyramid = None
        if prev_frame is not None:
            _, prev_frame_pyramid = cv2.buildOpticalFlowPyramid(prev_frame,
                                                                (self.WIN_SIZE, self.WIN_SIZE),
                                                                self.MAX_LEVELS, None, False)
        levels, frame_pyramid = cv2.buildOpticalFlowPyramid(curr_frame,
                                                            (self.WIN_SIZE, self.WIN_SIZE),
                                                            self.MAX_LEVELS, None, False)

        if len(self.corners) > 0:
            new_corners, statuses, err = None, None, None
            for level in range(levels, -1, -1):
                new_corners, statuses, _ = \
                    cv2.calcOpticalFlowPyrLK(prev_frame_pyramid[level], frame_pyramid[level],
                                             np.asarray(self.corners, dtype=np.float32) / 2 ** level,
                                             None if new_corners is None else (new_corners * 2),
                                             flags=0 if new_corners is None else cv2.OPTFLOW_USE_INITIAL_FLOW,
                                             winSize=(self.WIN_SIZE, self.WIN_SIZE))

            condition = self.count_condition(statuses)
            self.corners = new_corners[condition].tolist()
            self.sizes = np.asarray(self.sizes)[condition].tolist()
            self.ids = np.asarray(self.ids)[condition].tolist()

        mask = self.create_mask()
        for level, frame_level in enumerate(frame_pyramid):
            new_corners = cv2.goodFeaturesToTrack(
                frame_level,
                maxCorners=self.MAX_CORNERS - len(self.corners),
                qualityLevel=self.QUALITY_LEVEL,
                minDistance=self.MIN_DIST,
                blockSize=self.BLOCK_SIZE,
                mask=mask
            )

            if new_corners is not None:
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

                    mask = cv2.circle(mask, (x, y), cur_size + self.MIN_DIST, thickness=-1, color=0)

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
