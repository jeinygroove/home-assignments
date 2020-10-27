#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    TriangulationParameters,
    build_correspondences,
    pose_to_view_mat3x4,
    calc_inlier_indices,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4,
    check_baseline,
    get_baseline)


class CameraTrackerError(Exception):
    pass


class CameraTracker:
    MAX_REPROJ_ERR = 4.0

    def __init__(self, intrinsic_mat, corner_storage, known_view_1, known_view_2):
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage
        self.pc_builder = PointCloudBuilder()

        view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
        view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
        self.tracked_poses = [eye3x4()] * len(self.corner_storage)
        self.tracked_poses[known_view_1[0]] = view_mat_1
        self.tracked_poses[known_view_2[0]] = view_mat_2
        self.initial_baseline = get_baseline(view_mat_1, view_mat_2)

        self._add_points_from_frame(known_view_1[0], known_view_2[0], initial_triangulation=True)
        self.min_init_frame = min(known_view_1[0], known_view_2[0])
        self.max_init_frame = max(known_view_1[0], known_view_2[0])
        self.outliers = set()

    def _get_pos(self, frame_number) -> np.ndarray:
        corners = self.corner_storage[frame_number]
        # find corners ids that in point cloud
        _, pts_ids, corners_ids = np.intersect1d(self.pc_builder.ids,
                                                 corners.ids,
                                                 assume_unique=True,
                                                 return_indices=True)

        common_pts_3d = self.pc_builder.points[pts_ids]
        common_corner_ids = self.pc_builder.ids[pts_ids]
        common_corners = corners.points[corners_ids]
        inlier_mask = np.zeros_like(common_pts_3d, dtype=np.bool)
        inlier_counter = 0
        for common_id, mask_elem in zip(common_corner_ids[:, 0], inlier_mask):
            if common_id not in self.outliers:
                mask_elem[:] = True
                inlier_counter += 1
        if inlier_counter > 7:
            common_pts_3d = common_pts_3d[inlier_mask].reshape(-1, 3)
            common_corner_ids = common_corner_ids[inlier_mask[:, :1]].reshape(-1, 1)
            common_corners = common_corners[inlier_mask[:, :2]].reshape(-1, 2)
        if len(common_pts_3d) < 4:
            raise CameraTrackerError('Not enough points to solve RANSAC on frame ', str(frame_number))
        _, r_vec, t_vec, inliers = cv2.solvePnPRansac(common_pts_3d,
                                                      common_corners,
                                                      self.intrinsic_mat,
                                                      None,
                                                      reprojectionError=self.MAX_REPROJ_ERR,
                                                      flags=cv2.SOLVEPNP_EPNP)
        extrinsic_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        proj_matr = self.intrinsic_mat @ extrinsic_mat
        if inliers is None:
            raise CameraTrackerError('Couldn\'t solve PnP on frame', str(frame_number))

        mult = 1
        while len(inliers) < 5:
            mult *= 1.2
            inliers = calc_inlier_indices(common_pts_3d, common_corners, proj_matr, self.MAX_REPROJ_ERR * mult)
        inlier_pts = common_pts_3d[inliers]
        inlier_corners = common_corners[inliers]
        outlier_ids = np.setdiff1d(common_corner_ids, common_corner_ids[inliers], assume_unique=True)
        self.outliers.update(outlier_ids)

        print('Number of points in cloud: ', len(pts_ids))
        print('Number of inliers: ', len(inlier_corners))

        _, r_vec, t_vec, inliers = cv2.solvePnPRansac(inlier_pts, inlier_corners, self.intrinsic_mat, None,
                                                      r_vec, t_vec, useExtrinsicGuess=True)

        return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)

    def _add_points_from_frame(self,
                               frame_num_1: int,
                               frame_num_2: int,
                               initial_triangulation: bool = False):
        corners_1 = self.corner_storage[frame_num_1]
        corners_2 = self.corner_storage[frame_num_2]
        corresps = build_correspondences(corners_1, corners_2, ids_to_remove=self.pc_builder.ids)
        if len(corresps.ids) > 0:
            # I don't use here self.MAX_REPROJ_ERR because it gives worse result here.
            max_reproj_err = 1.0
            min_angle = 1.1
            view_1 = self.tracked_poses[frame_num_1]
            view_2 = self.tracked_poses[frame_num_2]
            triangulation_params = TriangulationParameters(max_reproj_err, min_angle, 0)
            pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                            view_1,
                                                                            view_2,
                                                                            self.intrinsic_mat,
                                                                            triangulation_params)

            if initial_triangulation:
                while len(pts_3d) < 8:
                    triangulation_params = TriangulationParameters(max_reproj_err, min_angle, 0)
                    pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                                    view_1,
                                                                                    view_2,
                                                                                    self.intrinsic_mat,
                                                                                    triangulation_params)
                    max_reproj_err *= 1.2
                    min_angle *= 0.8

            print('Added', len(pts_3d), 'points')
            self.pc_builder.add_points(triangulated_ids, pts_3d)

    def track(self):
        curr_frame = self.min_init_frame + 1
        num_of_frames = len(self.corner_storage)
        for _ in range(2, num_of_frames):
            if curr_frame == self.max_init_frame:
                curr_frame += 1
            if curr_frame >= num_of_frames:
                curr_frame = self.min_init_frame - 1
                self.outliers = set()
            print('Current frame: ', curr_frame)
            try:
                self.tracked_poses[curr_frame] = self._get_pos(curr_frame)
            except CameraTrackerError as error:
                print(error)
                print('Stopping tracking')
                break
            frame_num_diff = 5
            num_pairs = 0
            while num_pairs < 5:
                prev_frame = \
                    curr_frame - frame_num_diff if curr_frame > self.min_init_frame else curr_frame + frame_num_diff
                frame_num_diff += 1
                if prev_frame < num_of_frames and (
                        self.min_init_frame <= prev_frame or curr_frame < self.min_init_frame):
                    if check_baseline(self.tracked_poses[prev_frame], self.tracked_poses[curr_frame],
                                      self.initial_baseline * 0.15):
                        self._add_points_from_frame(prev_frame, curr_frame)
                        num_pairs += 1
                else:
                    break
            if curr_frame > self.min_init_frame:
                curr_frame += 1
            else:
                curr_frame -= 1

        return self.tracked_poses, self.pc_builder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_cloud_builder = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2).track()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
