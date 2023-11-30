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
        
        # Set up gtsam factor graph variables
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_guess = gtsam.Values()
        
        self.cal3ds2 = gtsam.Cal3DS2(K[0,0], K[1,1], K[0,1], K[0,2], K[1,2], 
                                     distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3])

        self.o = gtsam.symbol_shorthand.L
        self.x = gtsam.symbol_shorthand.X
        
        
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
        
        self.pose_idx += 1
        if pre_idx == -1:
            # prior_noise = gtsam.noiseModel.Gaussian.Covariance(
            #     np.diag([.01, .01, .01, .0001, .0001, .0001]))
            factor = gtsam.NonlinearEqualityPose3(self.x(1), gtsam.Pose3())
            self.graph.push_back(factor)
            self.initial_guess.insert(self.x(1), gtsam.Pose3())
            self.pose_chain.append(np.eye(4)) # initial pose at origin

        else:
            noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
            factor = gtsam.BetweenFactorPose3(self.x(pre_idx+1), self.x(self.pose_idx+1), 
                                              gtsam.Pose3(T_relative), noise)
            self.graph.push_back(factor)
            self.pose_chain.append(T_relative @ self.pose_chain[pre_idx])
            self.initial_guess.insert(self.x(self.pose_idx+1), gtsam.Pose3(self.pose_chain[-1]))
    
    def add_segment_measurement(self, object_id, measurement, pixel_std_dev, initial_guess=None, pose_idx=None):
        if pose_idx is None:
            pose_idx = self.pose_idx
        
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.pixelMsmtNoiseStd)
        factor = gtsam.GenericProjectionFactorCal3DS2(
            gtsam.Point2(measurement), measurement_noise, self.x(pose_idx+1), self.o(object_id+1), self.cal3ds2)
        self.graph.push_back(factor)
        
        if initial_guess is not None:
            self.initial_guess.insert(self.l(object_id+1), gtsam.Point3(initial_guess))
            
    def solve(self):
        optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_guess)
        result = optimizer.optimize()
        return result
        
    # def reconstruct(self, poses, poseNoises, tracks):

    #     # L = symbol_shorthand.L
    #     # X = symbol_shorthand.X

    #     # #cameraMeasurementNoise_noHuber = gtsam.noiseModel.Isotropic.Sigma(2, self.pixelMsmtNoiseStd)
    #     # #cameraMeasurementNoise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(self.huberParam ), cameraMeasurementNoise_noHuber)
    #     # cameraMeasurementNoise = gtsam.noiseModel.Isotropic.Sigma(2, self.pixelMsmtNoiseStd)

    #     self.tracksIncludedInSolution = []
    #     self.poses = poses

    #     self.landmarkMAPmeans = dict()
    #     self.landmarkMAPcovs = dict()
    #     self.landmarkSizes = dict()

    #     # Triangulate point positions using linear triangulation, to provide an initial guess for MAP
    #     for track in tracks:

    #         trackId = track.getTrackId()

    #         graph = NonlinearFactorGraph()
    #         initialValues = Values()

    #         framesWhereSeen, _, _ = track.getPxCoordsAndDescriptorsForAllFrames()
            
    #         gtsam_cams = []
    #         gtsam_observations = []
    #         projection_factors = []

    #         camera_centerpoints = []

    #         if (len(framesWhereSeen) >= self.minNumObservations):

    #             for frame in framesWhereSeen:
    #                 T = poses[frame]
    #                 T_gtsam = Pose3(T)

    #                 factor = gtsam.NonlinearEqualityPose3(X(frame), T_gtsam)
    #                 graph.push_back(factor)
    #                 initialValues.insert(X(frame), T_gtsam)

    #                 cam = gtsam.PinholeCameraCal3DS2(T_gtsam, self.K_gtsam)

    #                 pxCoords, _ = track.getPxcoordsAndDescriptorsForFrame(frame)                   

    #                 x_px = pxCoords[0]
    #                 y_px = pxCoords[1]

    #                 observation = Point2(x_px, y_px)

    #                 gtsam_cams.append(cam)
    #                 gtsam_observations.append(observation)

    #                 # Besides triangulating an initial guess, assemble a set of projection factors in case it turns out
    #                 # that triangulation succeeds for this track
    #                 proj_factor = GenericProjectionFactorCal3DS2(observation, cameraMeasurementNoise, X(frame), L(track.getTrackId()), self.K_gtsam)#, True, True)
    #                 projection_factors.append(proj_factor)

    #                 camera_centerpoints.append(T[0:3,3])

    #             measurements = gtsam.Point2Vector(gtsam_observations)
    #             cameras = gtsam.gtsam.CameraSetCal3DS2(gtsam_cams)

    #             try:
    #                 dlt_estimate = gtsam.triangulatePoint3(cameras, measurements, rank_tol=1e-9, optimize=True, model=cameraMeasurementNoise)
    #                 #print(f"dlt estimate for {track.getTrackId()}={dlt_estimate}")

    #                 # Add triangulation result as initial guess
    #                 initialValues.insert(L(trackId), dlt_estimate)

    #                 # Add projection factors corresponding to this track
    #                 for proj_factor in projection_factors:
    #                     graph.push_back(proj_factor)

    #                 params = gtsam.LevenbergMarquardtParams()
    #                 params.setVerbosityLM("TERMINATION")
    #                 optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialValues, params)
    #                 result = optimizer.optimize()

    #                 initialError = graph.error(initialValues)
    #                 finalError = graph.error(result)

    #                 marginals = gtsam.Marginals(graph, result)

    #                 lmMapMean = result.atPoint3(L(trackId))
    #                 lmMapCov = marginals.marginalCovariance(L(trackId))
                
    #                 self.landmarkMAPmeans[trackId] = lmMapMean
    #                 self.landmarkMAPcovs[trackId] = lmMapCov

    #                 # compute object size in meters, assuming it is a round disk
    #                 # whose normal is towards the line of observation
    #                 sizes_m = []
    #                 for camera_centerpoint, frameWhereSeen in zip(camera_centerpoints, framesWhereSeen):
    #                     dist = np.linalg.norm(lmMapMean - camera_centerpoint)
    #                     size_px = track.getSizeForFrame(frameWhereSeen)
    #                     size_m = 2*dist*size_px/(self.fx+self.fy)
    #                     sizes_m.append(size_m)

    #                 # Store size of track as meters
    #                 self.landmarkSizes[trackId] = np.mean(sizes_m)

    #                 # Make a note that this track is included in solution
    #                 self.tracksIncludedInSolution.append(track)

    #             except RuntimeError:
    #                 print(f"Track {track.getTrackId()} failed")
    #                 # If triangulation fails, runtime error is raised. Silently omit this.
    #                 pass

    #     return (None, None)

    # def getReconstructionResults(self):
    #     return self.landmarkMAPmeans, self.landmarkMAPcovs, self.poses, None, self.landmarkSizes
        
