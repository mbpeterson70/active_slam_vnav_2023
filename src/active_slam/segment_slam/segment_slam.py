#!/usr/bin/python3
#!/usr/bin/env python3

import gtsam
import numpy as np

class SegmentSLAM():
    
    def __init__(self, K, distortion_params):
        """
        SegmentSLAM constructor. Sets camera parameters.

        Args:
            K (np.array, shape(3,3)): Camera intrinsic calibration matrix
            distortion_params (array, shape(4,)): Distortion parameters (k1, k2, p1, p2)
        """
        # internal variables
        self.pose_idx = -1
        self.pose_chain = []
        self.object_id_mapping = {}
        self.object_ids = []
        
        # Set up gtsam factor graph variables
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_guess = gtsam.Values()
        
        self.cal3ds2 = gtsam.Cal3DS2(K[0,0], K[1,1], K[0,1], K[0,2], K[1,2], 
                                     distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3])

        self.o = gtsam.symbol_shorthand.O
        self.x = gtsam.symbol_shorthand.X
        
        
    def set_initial_pose(self, T_init=np.eye(4)):
        assert self.pose_idx == -1
        self.pose_idx = 0
        factor = gtsam.NonlinearEqualityPose3(self.x(0), gtsam.Pose3(T_init))
        self.graph.push_back(factor)
        self.initial_guess.insert(self.x(0), gtsam.Pose3(T_init))
        self.pose_chain.append(T_init) # initial pose at origin
        
    def add_relative_pose(self, T_relative, covariance, pre_idx=None):
        """
        Adds a new factor between two poses

        Args:
            T_relative (np.array, shape(4,4)): Rigid body transform matrix between the pose
                at pre_idx and the new pose variable.
            covariance (np.array, shape(6,6)): Rotation and position (in this order) covariance
            pre_idx (int, optional): Pose variable that the new pose is relative to. Defaults to 
                None if the last added pose should be used. If pre_idx == -1 (if this is the first
                pose entered, then only a prior is added attaching the first pose to the origin).
        """
        if pre_idx is None:
            pre_idx = self.pose_idx
        
        if pre_idx == -1:
            self.set_initial_pose()
        else:
            self.pose_idx += 1
            noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
            factor = gtsam.BetweenFactorPose3(self.x(pre_idx), self.x(self.pose_idx), 
                                              gtsam.Pose3(T_relative), noise)
            self.graph.push_back(factor)
            self.pose_chain.append(T_relative @ self.pose_chain[pre_idx])
            # self.pose_chain.append(self.pose_chain[pre_idx] @ T_relative)
            self.initial_guess.insert(self.x(self.pose_idx), gtsam.Pose3(self.pose_chain[-1]))
    
    def add_segment_measurement(self, object_id, center_pixel, pixel_std_dev, initial_guess=None, pose_idx=None):
        assert object_id in self.object_id_mapping
        if pose_idx is None:
            pose_idx = self.pose_idx
        
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_std_dev)
        factor = gtsam.GenericProjectionFactorCal3DS2(
            gtsam.Point2(center_pixel), measurement_noise, self.x(pose_idx), self.o(object_id), self.cal3ds2)
        self.graph.push_back(factor)
        
        if initial_guess is not None:
            try:
                self.initial_guess.insert(self.o(object_id), gtsam.Point3(initial_guess))
            except:
                print("SegmentSLAM Warning: Initial guess for object may already exist")

    def triangulate_object_init_guess(self, pixels: list, pixel_std_dev: float, pose_idxs: list):
        camera_poses = []
        for idx in pose_idxs:
            camera_poses.append(gtsam.PinholeCameraCal3DS2(gtsam.Pose3(self.pose_chain[idx]), self.cal3ds2))
        camera_set = gtsam.gtsam.CameraSetCal3DS2(camera_poses)
        pixels_point_vec = gtsam.Point2Vector([gtsam.Point2(pix) for pix in pixels])
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_std_dev)
        init_guess = gtsam.triangulatePoint3(camera_set, pixels_point_vec, rank_tol=1e-9, optimize=True, model=measurement_noise)
        return init_guess
    
    def new_objects_data_association(self, object_ids, init_guesses, last_pose_idxs):
        # Step 1: Fix the init_guesses so they are not used by only chaining together odometry
        # instead, use optimized poses to calculate this

        # Step 2: Perform data association.
        
        # Step 3: Add any segments that have been associated to existing objects to the internal
        # mapping.
        # TODO: implement above. For now, just assume it's a new object
        for obj_id in object_ids:
            self.object_id_mapping[obj_id] = obj_id
            self.object_ids.append(obj_id)
            
    def solve(self, reset_init_guess=True):
        optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_guess)
        result = optimizer.optimize()
        # if reset_init_guess:
        #     self.use_solution_as_init_guess(result)
        return result
    
    def remove_object(self, object_id):
        # for something in range(2):
        #     print(gtsam.VariableIndex(self.graph))
        i = 0
        num_factors = 0
        to_remove = []
        while num_factors < self.graph.nrFactors():
            if self.graph.exists(i):
                num_factors += 1
            else:
                i += 1
                continue
            factor = self.graph.at(i)
            if len(factor.keys()) != 2:
                i += 1
                continue
            for k in factor.keys():
                if gtsam.Symbol(k).chr() == ord('o') and gtsam.Symbol(k).index() == object_id:
                    to_remove.append(i)
                    break
            i += 1
        to_remove.reverse()
        # print(f"Found {len(to_remove)} factors to remove")
        for el in to_remove:
            self.graph.remove(el)

        # remove from object id mapping
        self.object_ids.remove(object_id)
        to_remove = []
        for x, y in self.object_id_mapping.items():
            if y == object_id:
                to_remove.append(x)
        for el in to_remove:
            del self.object_id_mapping[el]

        return to_remove
    
    # def use_solution_as_init_guess(self, solution):
    #     for i in range(len(self.pose_chain)):
    #         position = solution.atPose3(self.x(i)).matrix()