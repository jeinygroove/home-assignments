#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import datetime
from typing import List, Optional, Tuple

import numpy as np
import itertools

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
    compute_reprojection_errors,
    _remove_correspondences_with_ids, eye3x4,
    project_points,
)
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


class CameraTrackerError(Exception):
    pass


class CloudPointInfo:
    """
    Class for storing info about point from the cloud.
    It's position and number of inliers which can show
    how good this position is.
    """

    def __init__(self, pos, inliers):
        self.pos = pos
        self.inliers = inliers


class TrackedPoseInfo:
    """
    Class for storing info about found camera position.
    It's position and number of inliers which can show
    how good this position is.
    """

    def __init__(self, pos, inliers):
        self.pos = pos
        self.inliers = inliers


class CameraTracker:
    """
    Class that contains public track() method and additional private methods for it.
    """

    def __init__(self, intrinsic_mat, corner_storage, known_view_1, known_view_2, num_of_frames):
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage
        self.num_of_frames = num_of_frames
        # create dictionary instead of PointCloud object, because I want to store additional data for each point
        self.point_cloud = {}
        # precalculate for each corner frames and indices where it's visible to make retriangulation faster
        self.corner_pos_in_frames = {}
        for frame in range(self.num_of_frames):
            for index, corner_id in enumerate(self.corner_storage[frame].ids.flatten()):
                if corner_id not in self.corner_pos_in_frames.keys():
                    self.corner_pos_in_frames[corner_id] = []
                self.corner_pos_in_frames[corner_id].append((frame, index))

        self.tracked_poses = [None] * self.num_of_frames
        # compute proj matrices instead of multiplying intrinsic and view mat all the time
        self.proj_matrices = [None] * self.num_of_frames
        init_frame1, init_frame2 = None, None
        if known_view_1 is None or known_view_2 is None:
            init_frame1, init_frame2 = self._initialize_camera_tracker()
        else:
            init_frame1, init_frame2 = known_view_1[0], known_view_2[0]
            view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
            view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
            self.tracked_poses[init_frame1] = TrackedPoseInfo(view_mat_1, float('inf'))
            self.tracked_poses[init_frame2] = TrackedPoseInfo(view_mat_2, float('inf'))
            self.proj_matrices[init_frame1] = self.intrinsic_mat @ view_mat_1
            self.proj_matrices[init_frame2] = self.intrinsic_mat @ view_mat_2

        init_cloud_pts, init_ids = self._triangulate(init_frame1, init_frame2, True)
        print(f'Init point cloud: added {len(init_cloud_pts)} points.')

        self._update_point_cloud(init_cloud_pts, init_ids, 2 * np.ones_like(init_ids))

        # for each corner save the last time it was retriangulated.
        self.retriangulations = {}

    """
    Initialize camera positions for a pair of frames
    """

    def _initialize_camera_tracker(self,
                                   max_reproj_error=1.0,
                                   max_reproj_error_coef=1.1,
                                   min_angle=2.0,
                                   min_angle_coef=0.9,
                                   fst_frame_step_divider=7,
                                   snd_frame_step_size=5,
                                   seed=13,
                                   min_inliers=20):
        np.random.seed(seed)
        print("Init start: ", datetime.datetime.now())
        best_frame1, best_frame2 = None, None
        best_pair_num_of_points3d = 0
        best_essential_mat = None
        m1 = eye3x4()
        for frame1 in range(0, len(self.corner_storage), int(self.num_of_frames / fst_frame_step_divider)):
            frame1_corners = self.corner_storage[frame1]
            for frame2 in range(frame1 + snd_frame_step_size, len(self.corner_storage)):
                frame2_corners = self.corner_storage[frame2]
                corresp = build_correspondences(frame1_corners, frame2_corners)
                essential_mat, inliers_mask = cv2.findEssentialMat(corresp.points_1, corresp.points_2,
                                                                   self.intrinsic_mat, method=cv2.RANSAC)
                if essential_mat is None or inliers_mask is None:
                    continue
                _, homography_inliers = cv2.findHomography(corresp.points_1, corresp.points_2,
                                                           method=cv2.RANSAC, confidence=0.999,
                                                           ransacReprojThreshold=max_reproj_error)

                if np.count_nonzero(inliers_mask * homography_inliers) < min_inliers:
                    continue

                inliers_mask = inliers_mask.flatten()
                corr = _remove_correspondences_with_ids(corresp,
                                                        np.arange(len(inliers_mask))[inliers_mask == 0])
                R1, R2, t = cv2.decomposeEssentialMat(essential_mat)
                for rot, tvec in itertools.product([R1, R2], [t, -t]):
                    m2 = np.hstack((rot, tvec))
                    points3d, _, _ = triangulate_correspondences(corr, m1, m2, self.intrinsic_mat,
                                                                 TriangulationParameters(max_reproj_error, min_angle,
                                                                                         0))

                    if (best_frame1 is None and best_frame2 is None) or len(points3d) > best_pair_num_of_points3d:
                        best_frame1 = frame1
                        best_frame2 = frame2
                        best_pair_num_of_points3d = len(points3d)
                        best_essential_mat = m2.copy()
        if best_frame1 is None or best_frame2 is None:
            self._initialize_camera_tracker(max_reproj_error * max_reproj_error_coef, min_angle * min_angle_coef)
        else:
            print(f'Initialized with: frame1={best_frame1}, frame2={best_frame2}')
            self.tracked_poses[best_frame1] = TrackedPoseInfo(m1, float('inf'))
            self.tracked_poses[best_frame2] = TrackedPoseInfo(best_essential_mat, float('inf'))
            self.proj_matrices[best_frame1] = self.intrinsic_mat @ m1
            self.proj_matrices[best_frame2] = self.intrinsic_mat @ best_essential_mat
        print("Init finished", datetime.datetime.now())
        return best_frame1, best_frame2

    """
    Add/replace points in cloud with the new ones, 
    update if new position for the point is better.

    Attributes:
        cloud_pts: points positions
        ids: ids
        inliers: number of inliers

    Returns:
        int: number of updated points
    """

    def _update_point_cloud(self, cloud_pts, ids, inliers):
        num_of_updated_pts = 0
        for pt_id, pt, inl in zip(ids, cloud_pts, inliers):
            if pt_id not in self.point_cloud.keys() or inl >= self.point_cloud[pt_id].inliers:
                num_of_updated_pts += 1
                self.point_cloud[pt_id] = CloudPointInfo(pt, inl)
        return num_of_updated_pts

    """
    Try to find position of the camera on the frame.

    Attributes:
        frame_number: frame number for which we want to find camera position

    Returns:
        R, t and number of inliers or None if smth went wrong.    
    """

    def _get_pos(self,
                 frame_number,
                 max_reproj_error=1.6):
        corners = self.corner_storage[frame_number]
        common_corners, common_cloud_pts = [], []
        # find cloud points and corners that we know and are 'visible' on the given frame.
        for i, corner in zip(corners.ids.flatten(), corners.points):
            if i in self.point_cloud.keys():
                common_corners.append(corner)
                common_cloud_pts.append(self.point_cloud[i].pos)
        common_corners, common_cloud_pts = np.array(common_corners), np.array(common_cloud_pts)
        if len(common_cloud_pts) < 4:
            return None  # Not enough points for ransac
        # find inliers and initial position of the camera
        is_success, r_vec, t_vec, inliers = cv2.solvePnPRansac(common_cloud_pts, common_corners, self.intrinsic_mat,
                                                               None, flags=cv2.SOLVEPNP_EPNP,
                                                               reprojectionError=max_reproj_error)
        if not is_success:
            return None

        # specify PnP solution with iterative minimization of reprojection error using inliers
        _, r_vec, t_vec, _ = cv2.solvePnPRansac(common_cloud_pts[inliers], common_corners[inliers], self.intrinsic_mat,
                                                None, r_vec, t_vec, useExtrinsicGuess=True)
        return r_vec, t_vec, len(np.array(inliers).flatten())

    """
    Triangulate corners from two given frames.
    """

    def _triangulate(self,
                     frame_num_1: int,
                     frame_num_2: int,
                     initial_triangulation: bool = False,
                     max_reproj_error=2.0,
                     max_reproj_error_coef=1.2,
                     min_angle=1.0,
                     min_angle_coef=0.8,
                     min_pts3d_for_init=20):
        corners_1 = self.corner_storage[frame_num_1]
        corners_2 = self.corner_storage[frame_num_2]
        corresps = build_correspondences(corners_1, corners_2,
                                         ids_to_remove=np.array(list(map(int, self.point_cloud.keys())), dtype=int))
        if len(corresps.ids) > 0:
            # I don't use here self.MAX_REPROJ_ERR because it gives worse result here.
            view_1 = self.tracked_poses[frame_num_1].pos
            view_2 = self.tracked_poses[frame_num_2].pos
            triangulation_params = TriangulationParameters(max_reproj_error, min_angle, 0)
            pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                            view_1,
                                                                            view_2,
                                                                            self.intrinsic_mat,
                                                                            triangulation_params)

            # if it's initial triangulation, I want to find enough points because in other case
            # some tests (especially ironman) may fail.
            if initial_triangulation:
                while len(pts_3d) < min_pts3d_for_init:
                    triangulation_params = TriangulationParameters(max_reproj_error, min_angle, 0)
                    pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                                    view_1,
                                                                                    view_2,
                                                                                    self.intrinsic_mat,
                                                                                    triangulation_params)
                    max_reproj_error *= max_reproj_error_coef
                    min_angle *= min_angle_coef
            return pts_3d, triangulated_ids

    """
    Retriangulate corner.
    """

    def _retriangulate(self,
                       corner_id,
                       triangulation_parameters=TriangulationParameters(1.5, 2.5, 0.0),
                       min_num_of_pairs=10,
                       max_frames_for_retriang=5):
        frames, ids, corners, poses = [], [], [], []
        # find frames and position in each frame for this corner.
        for frame, index_on_frame in self.corner_pos_in_frames[corner_id]:
            if self.tracked_poses[frame] is not None:
                frames.append(frame)
                ids.append(index_on_frame)
                corners.append(self.corner_storage[frame].points[index_on_frame])
                poses.append(self.tracked_poses[frame].pos)

        if len(frames) < 2:
            return None  # not enough frames for retriangulation
        if len(frames) == 2:
            cloud_pts, _, _ = triangulate_correspondences(
                Correspondences(np.array([corner_id]), np.array([corners[0]]), np.array([corners[1]])),
                poses[0], poses[1], self.intrinsic_mat, triangulation_parameters)
            if len(cloud_pts) == 0:
                return None
            return cloud_pts[0], 2

        best_pos, best_frames_inliers, best_num_of_frames_inliers = None, None, None
        # triangulation for each pair of frames can take a lot of time, so do only num_of_pairs pairs
        num_of_pairs = np.minimum(len(frames) ** 2, min_num_of_pairs)
        for i in range(num_of_pairs):
            frame_1, frame_2 = np.random.choice(len(frames), 2, replace=False)
            corner_pos_1, corner_pos_2 = corners[frame_1], corners[frame_2]
            cloud_pts, _, _ = triangulate_correspondences(
                Correspondences(np.zeros(1), np.array([corner_pos_1]), np.array([corner_pos_2])),
                poses[frame_1], poses[frame_2], self.intrinsic_mat,
                triangulation_parameters)
            if len(cloud_pts) == 0:
                continue

            frames_inliers = np.full(len(frames), 0)
            for i, frame, corner in zip(range(len(frames)), frames, corners):
                frames_inliers[i] = np.sum(np.array(compute_reprojection_errors(
                    cloud_pts,
                    np.array([corner]),
                    self.proj_matrices[frame]
                ).flatten()) <= triangulation_parameters.max_reprojection_error)
            num_of_inliers = np.count_nonzero(frames_inliers)
            if best_frames_inliers is None or best_num_of_frames_inliers < num_of_inliers:
                best_frames_inliers = frames_inliers
                best_num_of_frames_inliers = num_of_inliers

        if best_frames_inliers is None or best_num_of_frames_inliers < 4:
            return None

        frames_idx = np.array([i for i, is_inlier in enumerate(best_frames_inliers) if is_inlier > 0])
        np.random.shuffle(frames_idx)
        corresp = [(corners[i], self.proj_matrices[frames[i]]) for i in frames_idx[:max_frames_for_retriang]]
        eqs = [view_mat_proj[2] * point[h] - view_mat_proj[h]
               for h in [0, 1] for point, view_mat_proj in corresp]
        u, s, vh = np.linalg.svd(eqs)
        cloud_pts = vh[-1][:3] / vh[-1][-1]

        return cloud_pts, best_num_of_frames_inliers

    """
    Retriangulate corners from frame and update point cloud.
    """

    def _update_point_cloud_with_retriangulation(self, frame, step_num,
                                                 retriangulation_interval=5,
                                                 retriangulation_limit=200,
                                                 log_time = (True, True)):
        if log_time[0]:
            print("Update with retriangulation started: ", datetime.datetime.now())
        # choose corners from frame that weren't retriangulated before of
        # the last retriangulation was more than retriangulation_interval ago.
        points = [i for i in self.corner_storage[frame].ids.flatten()
                  if i not in self.retriangulations.keys()
                  or self.retriangulations[i] < step_num - retriangulation_interval]
        np.random.shuffle(points)
        # choose not all points for retriangulation to make it faster
        points = points[:retriangulation_limit]

        retr_cloud_pts, retr_ids, retr_inliers = [], [], []
        for i, retr_result in zip(points, map(self._retriangulate, points)):
            if retr_result is not None:
                cloud_pt, inliers = retr_result
                retr_cloud_pts.append(cloud_pt)
                retr_ids.append(i)
                retr_inliers.append(inliers)
            self.retriangulations[i] = step_num

        if log_time[1]:
            print("Update with retriangulation ended: ", datetime.datetime.now())
        print(f'Updated points in the cloud: ', self._update_point_cloud(retr_cloud_pts, retr_ids, retr_inliers))

    """
    Calculate again already found camera positions and log number of updates (if new position is better.)
    """

    def _update_camera_poses(self, frames_for_update):
        solved_frames = [i for i in frames_for_update if self.tracked_poses[i] is not None]
        updated_poses = 0
        for i, pos_info in zip(solved_frames, map(self._get_pos, solved_frames)):
            if pos_info is not None:
                r_vec, t_vec, num_of_inliers = pos_info
                if num_of_inliers >= self.tracked_poses[i].inliers:
                    updated_poses += 1
                    view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
                    self.tracked_poses[i] = TrackedPoseInfo(
                        view_mat,
                        num_of_inliers)
                    self.proj_matrices[i] = self.intrinsic_mat @ view_mat
        print('Updated camera positions: ', updated_poses)

    def track(self, max_frames_for_brute_force=300, max_unsolved_frames=50, step_size=4, seed=42):
        np.random.seed(seed)
        print("Tracking started: ", datetime.datetime.now())
        step_num = 1
        frames_for_update = []
        num_of_defined_poses = np.sum([track_pos_info is not None for track_pos_info in self.tracked_poses])
        while num_of_defined_poses != self.num_of_frames:
            unsolved_frames = [i for i in range(self.num_of_frames) if self.tracked_poses[i] is None]

            leaved_frames = []
            if self.num_of_frames >= max_frames_for_brute_force:
                np.random.shuffle(np.array(unsolved_frames))
                leaved_frames = unsolved_frames[max_unsolved_frames:]
                unsolved_frames = unsolved_frames[:max_unsolved_frames]

            # find positions for unknown frames.
            new_poses_info = []
            for frame, found_pos_info in zip(unsolved_frames, map(self._get_pos, unsolved_frames)):
                if found_pos_info is not None:
                    new_poses_info.append((frame, found_pos_info))

            if len(new_poses_info) == 0 and len(leaved_frames) != 0:
                new_poses_info = []
                for frame, found_pos_info in zip(unsolved_frames, map(self._get_pos, leaved_frames)):
                    if found_pos_info is not None:
                        new_poses_info.append((frame, found_pos_info))
            if len(new_poses_info) == 0:
                raise CameraTrackerError(
                    f'Can not get more camera positions, '
                    f'{self.num_of_frames - num_of_defined_poses}'
                    f' frames left without defined camera position')

            best_frame = None
            best_new_pos_info = None

            # chose the best position info by comparing number of inliers.
            for frame, pos_info in new_poses_info:
                if best_new_pos_info is None or best_new_pos_info[2] < pos_info[2]:
                    best_new_pos_info = pos_info
                    best_frame = frame

            print('\nAdded camera position for frame ', best_frame)
            print('Number of inliers: ', best_new_pos_info[2])

            view_mat = rodrigues_and_translation_to_view_mat3x4(best_new_pos_info[0], best_new_pos_info[1])
            self.tracked_poses[best_frame] = TrackedPoseInfo(
                view_mat,
                best_new_pos_info[2]
            )
            self.proj_matrices[best_frame] = self.intrinsic_mat @ view_mat
            frames_for_update.append(best_frame)

            # update camera positions each 4 steps.
            if step_num % step_size == 0:
                self._update_point_cloud_with_retriangulation(best_frame, step_num)
                self._update_camera_poses(frames_for_update)
                frames_for_update.clear()
            step_num += 1
            num_of_defined_poses = np.sum([tracked_pos_info is not None for tracked_pos_info in self.tracked_poses])
            print(
                f'{num_of_defined_poses}/{self.num_of_frames} camera positions found, {len(self.point_cloud)} points in cloud')

        for frame in range(self.num_of_frames):
            log_time = (frame == 0, frame == self.num_of_frames-1)
            self._update_point_cloud_with_retriangulation(frame, step_num, log_time=log_time)
        self._update_camera_poses(np.arange(0, self.num_of_frames))

        ids, cloud_points = [], []
        for pt_id, clout_pt_info in self.point_cloud.items():
            ids.append(pt_id)
            cloud_points.append(clout_pt_info.pos)
        print("Tracking ended: ", datetime.datetime.now())
        return list(map(lambda tracked_pos_info: tracked_pos_info.pos, self.tracked_poses)), \
               PointCloudBuilder(np.array(ids), np.array(cloud_points))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
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
