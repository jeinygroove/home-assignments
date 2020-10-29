#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    TriangulationParameters,
    Correspondences,
    compute_reprojection_errors
)
import cv2


class CameraTrackerError(Exception):
    pass


class CloudPointInfo:
    def __init__(self, pos, inliers):
        self.pos = pos
        self.inliers = inliers


class TrackedPoseInfo:
    def __init__(self, pos, inliers):
        self.pos = pos
        self.inliers = inliers


class CameraTracker:
    MAX_REPROJ_ERR = 1.5

    def __init__(self, intrinsic_mat, corner_storage, known_view_1, known_view_2, num_of_frames):
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage
        self.num_of_frames = num_of_frames
        self.point_cloud = {}
        # precalculate for each corner frames and indices where it's visible to make retriangulation faster
        self.corner_pos_in_frames = {}
        for frame in range(self.num_of_frames):
            for index, corner_id in enumerate(self.corner_storage[frame].ids.flatten()):
                if corner_id not in self.corner_pos_in_frames.keys():
                    self.corner_pos_in_frames[corner_id] = []
                self.corner_pos_in_frames[corner_id].append((frame, index))

        view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
        view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
        self.tracked_poses = [None] * self.num_of_frames
        self.tracked_poses[known_view_1[0]] = TrackedPoseInfo(view_mat_1, float('inf'))
        self.tracked_poses[known_view_2[0]] = TrackedPoseInfo(view_mat_2, float('inf'))

        init_cloud_pts, init_ids = self._triangulate(known_view_1[0], known_view_2[0], True)
        print(f'Init point cloud: added {len(init_cloud_pts)} points.')
        self._update_point_cloud(init_cloud_pts, init_ids, 2 * np.ones_like(init_ids))

        self.retriangulations = {}

    def _update_point_cloud(self, cloud_pts, ids, inliers):
        num_of_updated_pts = 0
        for pt_id, pt, inl in zip(ids, cloud_pts, inliers):
            if pt_id not in self.point_cloud.keys() or inl >= self.point_cloud[pt_id].inliers:
                num_of_updated_pts += 1
                self.point_cloud[pt_id] = CloudPointInfo(pt, inl)
        return num_of_updated_pts

    def _get_pos(self, frame_number):
        corners = self.corner_storage[frame_number]
        common_cloud_pts = []
        common_corners = []
        for i, corner in zip(corners.ids.flatten(), corners.points):
            if i in self.point_cloud.keys():
                common_cloud_pts.append(self.point_cloud[i].pos)
                common_corners.append(corner)
        common_cloud_pts, common_corners = np.array(common_cloud_pts), np.array(common_corners)
        if len(common_cloud_pts) < 4:
            return None  # Not enough points for ransac
        is_success, r_vec, t_vec, inliers = cv2.solvePnPRansac(common_cloud_pts, common_corners, self.intrinsic_mat,
                                                               None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=1.6)
        if not is_success:
            return None

        _, r_vec, t_vec, _ = cv2.solvePnPRansac(common_cloud_pts[inliers], common_corners[inliers], self.intrinsic_mat,
                                                None, r_vec, t_vec, useExtrinsicGuess=True)

        return r_vec, t_vec, len(np.array(inliers).flatten())

    def _triangulate(self, frame_num_1: int, frame_num_2: int, initial_triangulation: bool = False):
        corners_1 = self.corner_storage[frame_num_1]
        corners_2 = self.corner_storage[frame_num_2]
        corresps = build_correspondences(corners_1, corners_2,
                                         ids_to_remove=np.array(list(map(int, self.point_cloud.keys())), dtype=int))
        if len(corresps.ids) > 0:
            # I don't use here self.MAX_REPROJ_ERR because it gives worse result here.
            max_reproj_err = 1.0
            min_angle = 1.0
            view_1 = self.tracked_poses[frame_num_1].pos
            view_2 = self.tracked_poses[frame_num_2].pos
            triangulation_params = TriangulationParameters(max_reproj_err, min_angle, 0)
            pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                            view_1,
                                                                            view_2,
                                                                            self.intrinsic_mat,
                                                                            triangulation_params)

            if initial_triangulation:
                while len(pts_3d) < 20:
                    triangulation_params = TriangulationParameters(max_reproj_err, min_angle, 0)
                    pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                                    view_1,
                                                                                    view_2,
                                                                                    self.intrinsic_mat,
                                                                                    triangulation_params)
                    max_reproj_err *= 1.2
                    min_angle *= 0.8

            return pts_3d, triangulated_ids

    def _retriangulate(self, corner_id):
        frames = []
        corners = []
        poses = []
        for frame, index_on_frame in self.corner_pos_in_frames[corner_id]:
            if self.tracked_poses[frame] is not None:
                frames.append(frame)
                corners.append(self.corner_storage[frame].points[index_on_frame])
                poses.append(self.tracked_poses[frame].pos)

        if len(frames) < 2:
            return None  # not enough frames for retriangulation
        if len(frames) == 2:
            cloud_pts, _ = self._triangulate(frames[0], frames[1])
            if len(cloud_pts) == 0:
                return None
            return cloud_pts[0], 2

        indices = np.arange(len(frames))
        np.random.shuffle(indices)
        indices = indices[:20]
        frames, corners, poses = np.array(frames)[indices], np.array(corners)[indices], np.array(poses)[indices]

        best_pos = None
        best_inliers = None
        for _ in range(5):
            frame_1, frame_2 = np.random.choice(len(frames), 2, replace=False)
            corner_pos_1, corner_pos_2 = corners[frame_1], corners[frame_2]
            cloud_pts, _, _ = triangulate_correspondences(
                Correspondences(np.zeros(1), np.array([corner_pos_1]), np.array([corner_pos_2])),
                poses[frame_1], poses[frame_2], self.intrinsic_mat,
                TriangulationParameters(self.MAX_REPROJ_ERR, 2.5, 0.0))
            if len(cloud_pts) == 0:
                continue
            repr_errors = [
                compute_reprojection_errors(cloud_pts, np.array([corner]),
                                            self.intrinsic_mat @ self.tracked_poses[frame].pos).flatten()[0]
                for frame, corner in zip(frames, corners)
            ]
            inliers = np.sum(np.array(repr_errors) <= self.MAX_REPROJ_ERR)
            if best_pos is None or best_inliers < inliers:
                best_pos = cloud_pts[0]
                best_inliers = inliers

        if best_pos is None:
            return None
        return best_pos, best_inliers

    def _update_point_cloud_with_retriangulation(self, frame, step_num, retriangulation_interval=5,
                                                 retriangulation_limit=1000):
        points = [i for i in self.corner_storage[frame].ids.flatten()
                  if i not in self.retriangulations.keys() or self.retriangulations[
                      i] < step_num - retriangulation_interval]
        np.random.shuffle(points)
        points = points[:retriangulation_limit]

        retr_cloud_pts, retr_ids, retr_inliers = [], [], []
        for i, retr_result in zip(points, map(self._retriangulate, points)):
            if retr_result is not None:
                cloud_pt, inliers = retr_result
                retr_cloud_pts.append(cloud_pt)
                retr_ids.append(i)
                retr_inliers.append(inliers)
            self.retriangulations[i] = step_num

        print(f'Updated points in the cloud: ', self._update_point_cloud(retr_cloud_pts, retr_ids, retr_inliers))

    def _update_camera_poses(self):
        unsolved_frames = [i for i in range(self.num_of_frames) if self.tracked_poses[i] is not None]
        updated_poses = 0
        for i, pos_info in zip(unsolved_frames, map(self._get_pos, unsolved_frames)):
            if pos_info is not None:
                r_vec, t_vec, num_of_inliers = pos_info
                if num_of_inliers >= self.tracked_poses[i].inliers:
                    updated_poses += 1
                    self.tracked_poses[i] = TrackedPoseInfo(
                        rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec),
                        num_of_inliers)
        print('Updates camera position: ', updated_poses)

    def track(self):
        step_num = 1
        num_of_defined_poses = np.sum([track_pos_info is not None for track_pos_info in self.tracked_poses])
        while num_of_defined_poses != self.num_of_frames:
            unsolved_frames = [i for i in range(self.num_of_frames) if self.tracked_poses[i] is None]

            new_poses_info = []
            for frame, found_pos_info in zip(unsolved_frames, map(self._get_pos, unsolved_frames)):
                if found_pos_info is not None:
                    new_poses_info.append((frame, found_pos_info))

            if len(new_poses_info) == 0:
                raise CameraTrackerError(
                    f'Can not get more camera positions, '
                    f'{self.num_of_frames - num_of_defined_poses}'
                    f' frames left without defined camera position')

            best_frame = None
            best_new_pos_info = None

            for frame, pos_info in new_poses_info:
                if best_new_pos_info is None or best_new_pos_info[2] < pos_info[2]:
                    best_new_pos_info = pos_info
                    best_frame = frame

            print('Added camera position for frame ', best_frame)
            print('Number of inliers: ', best_new_pos_info[2])

            self.tracked_poses[best_frame] = TrackedPoseInfo(
                rodrigues_and_translation_to_view_mat3x4(
                    best_new_pos_info[0], best_new_pos_info[1]), best_new_pos_info[2])

            self._update_point_cloud_with_retriangulation(best_frame, step_num)

            if step_num % 5 == 0:
                self._update_camera_poses()
            step_num += 1
            num_of_defined_poses = np.sum([tracked_pos_info is not None for tracked_pos_info in self.tracked_poses])
            print(
                f'{num_of_defined_poses}/{self.num_of_frames} camera positions found, {len(self.point_cloud)} points in cloud')

        self._update_camera_poses()

        ids, cloud_points = [], []
        for pt_id, clout_pt_info in self.point_cloud.items():
            ids.append(pt_id)
            cloud_points.append(clout_pt_info.pos)
        return list(map(lambda tracked_pos_info: tracked_pos_info.pos, self.tracked_poses)), \
               PointCloudBuilder(np.array(ids), np.array(cloud_points))


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

    # if we're unlucky, Camera Tracker can not to find poses for a few frames
    try:
        view_mats, point_cloud_builder = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2,
                                                       len(rgb_sequence)).track()
    except CameraTrackerError:
        view_mats, point_cloud_builder = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2,
                                                       len(rgb_sequence)).track()

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
