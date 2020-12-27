#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import datetime
from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import itertools

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
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
    compute_reprojection_errors_without_norm,
    _get_bundle_adjustment_matrix,
    _view_mat4x4_to_rodrigues_and_translation,
    compute_reprojection_errors_from_params,
    FullInfoPointCloudBuilder,
)
import cv2
from scipy.optimize import least_squares


class CameraTrackerError(Exception):
    pass


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

    def __init__(self, intrinsic_mat, corner_storage, known_view_1, known_view_2, num_of_frames, seed=42):
        self.seed = seed
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage
        self.num_of_frames = num_of_frames
        self.point_cloud = FullInfoPointCloudBuilder()
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
        self.init_frame1, self.init_frame2 = None, None
        if known_view_1 is None or known_view_2 is None:
            self.init_frame1, self.init_frame2 = self._initialize_camera_tracker()
        else:
            self.init_frame1, self.init_frame2 = known_view_1[0], known_view_2[0]
            view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
            view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
            self.tracked_poses[self.init_frame1] = TrackedPoseInfo(view_mat_1, float('inf'))
            self.tracked_poses[self.init_frame2] = TrackedPoseInfo(view_mat_2, float('inf'))
            self.proj_matrices[self.init_frame1] = self.intrinsic_mat @ view_mat_1
            self.proj_matrices[self.init_frame2] = self.intrinsic_mat @ view_mat_2

        init_cloud_pts, init_ids = self._triangulate(self.init_frame1, self.init_frame2, True)
        print(f'Init point cloud: added {len(init_cloud_pts)} points.')
        self.point_cloud.add_points_with_inliers(init_ids, init_cloud_pts, np.array([2]))
        self.point_cloud.add_corners_on_frame(self.corner_storage[self.init_frame1], self.init_frame1)
        self.point_cloud.add_corners_on_frame(self.corner_storage[self.init_frame2], self.init_frame2)

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
                                   min_depth=10,
                                   fst_frame_step_divider=7,
                                   snd_frame_step_size=5,
                                   max_homography_part=0.3,
                                   rec_depth=0,
                                   max_rec_depth=10):
        if self.seed is not None:
            np.random.seed(self.seed)
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

                if essential_mat is None or inliers_mask is None or np.count_nonzero(inliers_mask) == 0:
                    continue
                inliers_mask = inliers_mask.astype(np.bool).flatten()
                corresp = Correspondences(corresp.ids[inliers_mask], corresp.points_1[inliers_mask],
                                          corresp.points_2[inliers_mask])

                _, homography_inliers = cv2.findHomography(corresp.points_1, corresp.points_2,
                                                           method=cv2.RANSAC, confidence=0.999,
                                                           ransacReprojThreshold=max_reproj_error)

                if np.count_nonzero(homography_inliers) / np.count_nonzero(inliers_mask) > max_homography_part:
                    continue

                inliers_mask = inliers_mask.flatten()
                corr = _remove_correspondences_with_ids(corresp,
                                                        np.arange(len(inliers_mask))[inliers_mask == 0])
                R1, R2, t = cv2.decomposeEssentialMat(essential_mat)
                for rot, tvec in itertools.product([R1, R2], [t, -t]):
                    m2 = np.hstack((rot, tvec))
                    points3d, _, _ = triangulate_correspondences(corr, m1, m2, self.intrinsic_mat,
                                                                 TriangulationParameters(max_reproj_error, min_angle,
                                                                                         min_depth))

                    if (best_frame1 is None and best_frame2 is None) or len(points3d) > best_pair_num_of_points3d:
                        best_frame1 = frame1
                        best_frame2 = frame2
                        best_pair_num_of_points3d = len(points3d)
                        best_essential_mat = m2.copy()
        if best_frame1 is None or best_frame2 is None:
            if rec_depth >= max_rec_depth:
                raise CameraTrackerError("Can't initialize camera tracker!")
            self._initialize_camera_tracker(max_reproj_error * max_reproj_error_coef,
                                            min_angle * min_angle_coef,
                                            rec_depth=rec_depth + 1)
        else:
            print(f'Initialized with: frame1={best_frame1}, frame2={best_frame2}')
            self.tracked_poses[best_frame1] = TrackedPoseInfo(m1,
                                                              self.find_view_mat_inliers(
                                                                  best_frame1,
                                                                  m1,
                                                                  max_reproj_error)
                                                              )
            self.tracked_poses[best_frame2] = TrackedPoseInfo(best_essential_mat,
                                                              self.find_view_mat_inliers(
                                                                  best_frame2,
                                                                  best_essential_mat,
                                                                  max_reproj_error)
                                                              )
            self.proj_matrices[best_frame1] = self.intrinsic_mat @ m1
            self.proj_matrices[best_frame2] = self.intrinsic_mat @ best_essential_mat
        print("Init finished", datetime.datetime.now())
        return best_frame1, best_frame2

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
        # find cloud points and corners that we know and are 'visible' on the given frame.
        intersection, (ids_3d, ids_2d) = snp.intersect(self.point_cloud.ids.flatten(),
                                                       corners.ids.flatten(),
                                                       indices=True)
        common_corners, common_cloud_pts = np.array(corners.points[ids_2d]), np.array(self.point_cloud.points[ids_3d])
        if len(common_cloud_pts) < 4:
            return None  # Not enough points for ransac
        # find inliers and initial position of the camera
        is_success, r_vec, t_vec, inliers = self.solvePnPWithMEstimator(common_cloud_pts, common_corners,
                                                                        max_reproj_error)

        return r_vec, t_vec, len(np.array(inliers).flatten())

    def find_view_mat_inliers(self, frame, view_mat, max_reproj_error):
        corners = self.corner_storage[frame]
        # find cloud points and corners that we know and are 'visible' on the given frame.
        intersection, (ids_3d, ids_2d) = snp.intersect(self.point_cloud.ids.flatten(),
                                                       corners.ids.flatten(),
                                                       indices=True)
        common_corners, common_cloud_pts = np.array(corners.points[ids_2d]), np.array(self.point_cloud.points[ids_3d])
        reproj_errors = compute_reprojection_errors(common_cloud_pts, common_corners, self.intrinsic_mat @ view_mat)
        inliers = np.count_nonzero(reproj_errors < max_reproj_error)
        return inliers

    def find_points_inliers(self, cloud_points, ids, proj_mats, max_reproj_error):
        inliers = []
        for i, cloud_pt in enumerate(cloud_points):
            point = self.point_cloud.points_with_pose3d[ids[i]]
            frames = point.frames
            points2d = np.array(point.corners)
            cloud_pt = np.append(cloud_pt, 1)
            proj_corners = np.dot(proj_mats[frames], cloud_pt)
            proj_corners /= proj_corners[:, [2]]
            proj_corners = proj_corners[:, :2]
            reproj_errors = np.linalg.norm(points2d - proj_corners, axis=1)
            inliers.append(np.count_nonzero(reproj_errors < max_reproj_error))
        return np.array(inliers)

    def apply_bundle_adjustment(self, view_mats_mask, max_reproj_error):
        print("Bundle adjustment algorithm start: ", datetime.datetime.now())
        frames = np.argwhere(view_mats_mask).flatten()
        ids = self.point_cloud.ids.flatten()
        view_mats = []

        for frame in range(self.num_of_frames):
            if self.tracked_poses[frame] is None:
                # set null matrix instead of None for easier implementation of other methods
                view_mats.append(np.zeros((3, 4)))
            else:
                view_mats.append(self.tracked_poses[frame].pos)
        view_mats = np.array(view_mats)
        view_mats[view_mats_mask], points3d = self.bundle_adjustment_algorithm_with_retriang(
            self.point_cloud.points,
            self.point_cloud.ids,
            view_mats[view_mats_mask],
            frames)
        self.proj_matrices = self.intrinsic_mat @ view_mats
        inliers = self.find_points_inliers(points3d, ids, self.proj_matrices, max_reproj_error)
        self.point_cloud.add_points_with_inliers(self.point_cloud.ids,
                                                 points3d,
                                                 inliers)
        for i, vm in enumerate(view_mats):
            if vm.any():
                inliers = self.find_view_mat_inliers(i, vm, max_reproj_error)
                self.tracked_poses[i] = TrackedPoseInfo(vm, inliers)
        print("Bundle adjustment algorithm finished: ", datetime.datetime.now())

    def bundle_adjustment_algorithm_with_retriang(self, points3d, ids, view_mats, frames):
        num_of_view_mats = len(view_mats)
        num_of_cloud_pts = len(points3d)

        corners = np.zeros((0, 2))
        corners_ids = np.zeros((0,), dtype=np.int)
        camera_inds = np.zeros((0,), dtype=np.int)
        point_ids = np.zeros((0,), dtype=np.int)
        for i, frame in enumerate(frames):
            _, (points3d_ids, points2d_ids) = snp.intersect(ids.flatten(), self.corner_storage[frame].ids.flatten(),
                                                            indices=True)
            corners = np.concatenate((corners, self.corner_storage[frame].points[points2d_ids]))
            corners_ids = np.concatenate((corners_ids, self.corner_storage[frame].ids.flatten()[points2d_ids]))
            camera_inds = np.concatenate((camera_inds, np.full(len(points2d_ids), i)))
            point_ids = np.concatenate((point_ids, points3d_ids))

        print("Retriangulation before bundle adjustment start: ", datetime.datetime.now())
        for i in np.unique(corners_ids):
            retr_result = self._retriangulate(i)
            if retr_result is not None:
                cloud_pt, inliers = retr_result
                self.point_cloud.add_points_with_inliers(np.array([i]), np.array([cloud_pt]), np.array([inliers]))
        print("Retriangulation before bundle adjustment finished: ", datetime.datetime.now())

        x0 = np.zeros(6 * num_of_view_mats + 3 * num_of_cloud_pts)
        for i, view_mat in enumerate(view_mats):
            r_vec, t_vec = _view_mat4x4_to_rodrigues_and_translation(view_mat)
            x0[6 * i:6 * (i + 1)] = np.concatenate((r_vec, t_vec))

        for i, pt3d in enumerate(points3d):
            x0[6 * num_of_view_mats + 3 * i:6 * num_of_view_mats + 3 * i + 3] = pt3d

        ba_mat = _get_bundle_adjustment_matrix(num_of_cloud_pts, num_of_view_mats, camera_inds, point_ids, corners)
        result = least_squares(fun=compute_reprojection_errors_from_params,
                               x0=x0, jac_sparsity=ba_mat, x_scale='jac',
                               method='trf', ftol=1e-4,
                               args=(num_of_view_mats, num_of_cloud_pts, camera_inds,
                                     point_ids, corners, self.intrinsic_mat)).x

        for i in range(num_of_view_mats):
            r_vec, t_vec = result[6 * i:6 * i + 3, np.newaxis], result[6 * i + 3:6 * i + 6, np.newaxis]
            view_mats[i] = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        for i in range(num_of_cloud_pts):
            points3d[i] = result[6 * num_of_view_mats + 3 * i:6 * num_of_view_mats + 3 * i + 3]
        return view_mats, points3d

    def solvePnPWithMEstimator(self, cloud_pts, corners, max_reprojection_error=1.5):
        # calculate the first approximation with EPNP
        is_success, r_vec, t_vec, inliers = cv2.solvePnPRansac(cloud_pts, corners, self.intrinsic_mat, None,
                                                               max_reprojection_error,
                                                               confidence=0.999,
                                                               flags=cv2.SOLVEPNP_EPNP
                                                               )
        if is_success:
            pos_vec_init = np.concatenate((r_vec, t_vec)).flatten()
            pos_vec = least_squares(
                fun=lambda v, *args:
                compute_reprojection_errors_without_norm(*args,
                                                         self.intrinsic_mat @ rodrigues_and_translation_to_view_mat3x4(
                                                             v[:3, np.newaxis], v[3:, np.newaxis])),
                args=(cloud_pts[inliers.flatten()], corners[inliers.flatten()]),
                x0=pos_vec_init,
                loss='huber',
                method='trf',
            ).x

            r_vec, t_vec = pos_vec[:3, np.newaxis], pos_vec[3:, np.newaxis]
        return is_success, r_vec, t_vec, inliers

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
                     min_depth=10,
                     min_pts3d_for_init=20):
        corners_1 = self.corner_storage[frame_num_1]
        corners_2 = self.corner_storage[frame_num_2]
        corresps = build_correspondences(corners_1, corners_2,
                                         ids_to_remove=np.array(list(map(int, self.point_cloud.ids)), dtype=int))
        if len(corresps.ids) > 0:
            # I don't use here self.MAX_REPROJ_ERR because it gives worse result here.
            view_1 = self.tracked_poses[frame_num_1].pos
            view_2 = self.tracked_poses[frame_num_2].pos
            triangulation_params = TriangulationParameters(max_reproj_error, min_angle, min_depth)
            pts_3d, triangulated_ids, med_cos = triangulate_correspondences(corresps,
                                                                            view_1,
                                                                            view_2,
                                                                            self.intrinsic_mat,
                                                                            triangulation_params)

            # if it's initial triangulation, I want to find enough points because in other case
            # some tests (especially ironman) may fail.
            if initial_triangulation:
                while len(pts_3d) < min_pts3d_for_init:
                    triangulation_params = TriangulationParameters(max_reproj_error, min_angle, min_depth)
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
                       triangulation_parameters=TriangulationParameters(1.5, 2.5, 0.1),
                       min_num_of_pairs=20,
                       max_frames_for_retriang=7):
        frames, ids, corners, poses = [], [], [], []
        # find frames and position in each frame for this corner.
        for frame, index_on_frame in self.corner_pos_in_frames[corner_id]:
            if self.tracked_poses[frame] is not None:
                frames.append(frame)
                ids.append(index_on_frame)
                corners.append(self.corner_storage[frame].points[index_on_frame])
                poses.append(self.tracked_poses[frame].pos)
        if len(frames) < 2 or len(corners) < 2:
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
                                                 max_frames_for_limitation=200,
                                                 retriangulation_limits=(300, 700),
                                                 log_time=(True, True)):
        if log_time[0]:
            print("Update with retriangulation started: ", datetime.datetime.now())
        # choose corners from frame and sort them by the last retriangulation step in asc order
        points = sorted([i for i in self.corner_storage[frame].ids.flatten()],
                        key=lambda i: self.retriangulations[i] if i in self.retriangulations.keys() else -1)
        # choose not all points for retriangulation to make it faster
        retriangulation_limit = retriangulation_limits[0]
        if self.num_of_frames <= max_frames_for_limitation:
            retriangulation_limit = min(max(retriangulation_limit, int(len(points))), retriangulation_limits[1])
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
        if len(retr_ids) != 0:
            self.point_cloud.add_points_with_inliers(np.array(retr_ids), np.array(retr_cloud_pts),
                                                     np.array(retr_inliers))
        print(f'Updated points in the cloud: ', len(retr_ids))

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

    def track(self,
              max_frames_for_brute_force=300,
              max_unsolved_frames=50,
              step_size=4,
              max_reproj_error_ba=1.3,
              ba_frames_threshold=70):
        print("Tracking started: ", datetime.datetime.now())
        step_num = 1
        frames_for_update = []
        view_mats_for_ba_mask = np.full(self.num_of_frames, False)
        view_mats_for_ba_mask[self.init_frame1] = True
        view_mats_for_ba_mask[self.init_frame2] = True
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
            self.point_cloud.add_corners_on_frame(self.corner_storage[best_frame], best_frame)
            frames_for_update.append(best_frame)
            view_mats_for_ba_mask[best_frame] = True

            # update camera positions each 4 steps.
            if step_num % step_size == 0:
                self._update_point_cloud_with_retriangulation(best_frame, step_num)
                self._update_camera_poses(frames_for_update)
                frames_for_update.clear()
            if self.num_of_frames < ba_frames_threshold and \
                    (step_num % (self.num_of_frames // 3) == 0 or view_mats_for_ba_mask.all()):
                self.apply_bundle_adjustment(view_mats_for_ba_mask, max_reproj_error_ba)
            step_num += 1
            num_of_defined_poses = np.sum([tracked_pos_info is not None for tracked_pos_info in self.tracked_poses])
            print(
                f'{num_of_defined_poses}/{self.num_of_frames} camera positions found, {len(self.point_cloud.ids)} points in cloud')

        self._update_camera_poses(np.arange(0, self.num_of_frames))

        print("Tracking ended: ", datetime.datetime.now())
        return list(map(lambda tracked_pos_info: tracked_pos_info.pos, self.tracked_poses)), self.point_cloud


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
    num_of_frames = len(rgb_sequence)
    step_size = 2 if (num_of_frames <= 50) else (
        4 if (num_of_frames <= 100) else (5 if (len(rgb_sequence) <= 350) else 7))
    try:
        view_mats, point_cloud_builder = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2,
                                                       len(rgb_sequence), seed=40).track(step_size=step_size)
    except CameraTrackerError:
        view_mats, point_cloud_builder = CameraTracker(intrinsic_mat, corner_storage, known_view_1, known_view_2,
                                                       len(rgb_sequence), seed=50).track(step_size=step_size)

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
