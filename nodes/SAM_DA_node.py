#!/usr/bin/env python
import rospy
from active_slam.blob_SAM_node import BlobSAMNode
import numpy as np

# ROS imports
import rospy
import cv_bridge
import message_filters
from geometry_msgs.msg import Pose2D, PoseWithCovariance

# ROS msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import active_slam.msg as active_slam_msgs
import std_msgs.msg as std_msgs

# import rotation for pose stuff
from scipy.spatial.transform import Rotation as Rot

from active_slam.BlobTrackerRT import BlobTracker
from active_slam.SamDetectorDescriptorAndSizeComparer import SamDetectorDescriptorAndSizeComparer
from active_slam.SamFeatDdc import SamFeatDdc
from active_slam.FastSamWrapper import FastSamWrapper
from active_slam.utils import readConfig, getLogger, plotErrorEllipse

from utils import T_2_pose_msg, pose_msg_2_T

class SAM_DA_node:

    def __init__(self):
        
        # internal variables
        self.bridge = cv_bridge.CvBridge()
        self.counter = 0

        # ros params
        self.num_meas_new_obj = rospy.get_param("~num_meas_new_obj", 3) # number of objects needed 
                                                                        # to create new object, default: 3
        
        # ros subscribers
        subs = [
            message_filters.Subscriber("airsim_node/Multirotor/odom_local_ned", nav_msgs.Odometry),
            message_filters.Subscriber("airsim_node/Multirotor/front_center_custom/Scene",
                                       sensor_msgs.Image),
            # message_filters.Subscriber("/airsim_node/Multirotor/front_center_custom/Scene/camera_info", 
            #                            sensor_msgs.CameraInfo),
        ]

        # SAM params      
        matchingScoreLowerLimit = 0
        fTestLimit = 2.0
        numFramesToSearchOver = 3 
        pixelMsmtNoiseStd = 3.0
        numObservationsRequiredForTriang = 3
        huberParam = 0.5

        # FastSAM params
        similaritymethod = 'feat'
        pathToCheckpoint = "./FastSAM/Models/FastSAM-x.pt"
        device = "cpu"
        conf = 0.5
        iou = 0.9
        imgsz = 256
        samModel = FastSamWrapper(pathToCheckpoint, device, conf, iou, imgsz)

        logger = getLogger()

        maxSizeDiff = 0.5
        ddc = SamFeatDdc(samModel, maxSizeDiff)

        # Instantiate blob tracker
        self.blobTracker = BlobTracker(ddc, fTestLimit, matchingScoreLowerLimit, numFramesToSearchOver, logger)
        print("BlobTracker instantiated")

        # Instantiate BlobSAMNode
        self.blob_sam_node = BlobSAMNode(image=None, T=None, filename=None, blobTracker=self.blobTracker)  # Instantiate BlobSAMNode class
        print("BlobSAMNode instantiated")

        # subscribe to /first_goal_reached topic
        self.first_goal_reached_sub = rospy.Subscriber("/first_goal_reached", std_msgs.Bool, self.first_goal_reached_cb)

        # Approximate Time Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=1, slop=.1)

        # ros publishers
        self.meas_pub = rospy.Publisher("measurement_packet", active_slam_msgs.MeasurementPacket, queue_size=5)

        # initialize last image and pose
        self.last_image = None
        self.last_pose = None

    # Callback for first goal reached
    def first_goal_reached_cb(self, msg):
        """
        This function gets called when the first goal is reached.
        """
        self.ts.registerCallback(self.cb) # registers incoming messages to callback

    def cb(self, *msgs):
        """
        This function gets called every time synchronized odometry, image message, and camera info 
        message are available.
        """
        odom_msg, img_msg, = msgs #cam_info_msg = msgs

        counter = self.counter

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # extract camera intrinsics
        # K = np.array([cam_info_msg.K]).reshape((3,3))

        # extract pose from odom msg using position and orientation
        R = Rot.from_quat([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, \
        odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
        t = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        T = np.eye(4); T[:3,:3] = R.as_matrix(); T[:3,3] = t

        # get latest keyframe index
        keyframe = self.blob_sam_node.blobTracker.latestKeyframeIndex
        if keyframe == None:
            keyframe = 0

        # create a noise transformation matrix
        pose_pos_cov = 0.05 # meters
        pose_rot_cov = 0.05 # degrees
        rpy_cov = np.deg2rad(pose_rot_cov)**2
        xyz_cov = pose_pos_cov**2
        noise_matrix = np.eye(4)
        noise_matrix[:3, :3] = Rot.from_euler('xyz', np.random.multivariate_normal([0,0,0], np.diag([rpy_cov, rpy_cov, rpy_cov]))).as_matrix()
        noise_matrix[:3, 3] = np.random.multivariate_normal([0,0,0], np.diag([xyz_cov, xyz_cov, xyz_cov])) 

        # image and pose in BlobSAMNode
        self.blob_sam_node.image = img
        self.blob_sam_node.T = T
        self.blob_sam_node.filename = self.blob_sam_node.blobTracker.latestKeyframeIndex
        self.blob_sam_node.blobTracker = self.blobTracker

        print(f"latest keyframe index: {keyframe}")

        # Process image and get tracks
        tracks = self.blob_sam_node.process_image()

        # Create measurement packet
        packet = active_slam_msgs.MeasurementPacket()
        packet.header = img_msg.header
        packet.sequence = np.int32(counter)

        # Add relative pose measurement
        if self.last_pose is None: # for initial pose
            packet.incremental_pose = PoseWithCovariance()
            # TODO: right now sending the absolute pose as the first message, a bit hacky
            packet.incremental_pose.pose = T_2_pose_msg(T) # not adding noise to initial pose for now
            packet.incremental_pose.covariance = np.diag([xyz_cov, xyz_cov, xyz_cov, rpy_cov, rpy_cov, rpy_cov]).reshape(-1)
        else:
            packet.incremental_pose = PoseWithCovariance()
            incremental_pose = np.linalg.inv(self.last_pose) @ T
            packet.incremental_pose.pose = T_2_pose_msg(noise_matrix @ incremental_pose) # noise needs to be applied after finding the true incremental pose
            packet.incremental_pose.covariance = np.diag([xyz_cov, xyz_cov, xyz_cov, rpy_cov, rpy_cov, rpy_cov]).reshape(-1)

        for track in self.blobTracker.tracks:

            # print("Track ID: ", track.trackId)
            print("Counter: ", counter)

            frames_where_seen = track.framesWhereSeen
            # num track id is length of frames_where_seen
            num_track_id = len(frames_where_seen)
            print(f"Track ID: {track.trackId}, Number of times seen: {num_track_id}")

            print("Frames wheres seen: ", frames_where_seen)

            # only run if seen in latest frame:
            if counter not in frames_where_seen:
                continue

            current_px_coords, current_desc = track.getPxcoordsAndDescriptorsForFrame(counter)
            current_sift = current_desc[1]
            #print(f"current sift: {current_sift}")

            # If the track has been seen in 3 frames and pixel coordinates is not none, add all three to the measurement packet
            if num_track_id == 3 and current_px_coords is not None:
                # make a segment measurement for each counter in frames where seen
                for frame in frames_where_seen:
                    print(f"Adding one of three segment measurements to packet at sequence {frame} for track {track.trackId}")

                    #print(f"Frame: {frame}")

                    px_coords, desc = track.getPxcoordsAndDescriptorsForFrame(frame)
                    sift = desc[1]
                    #print(f"Pixel Coordinates: {px_coords}")
                    
                    segmentMeasurement = active_slam_msgs.SegmentMeasurement()
                    segmentMeasurement.id = track.trackId
                    segmentMeasurement.center = Pose2D(x=px_coords[0], y=px_coords[1], theta=0)
                    segmentMeasurement.sequence = np.int32(frame) 
                    # TODO: make rosparam pixel covariance
                    segmentMeasurement.covariance = np.diag([3., 3.]).reshape(-1)
                    # segmentMeasurement.latest_sift = sift
                    packet.segments.append(segmentMeasurement)

            if num_track_id > 3 and current_px_coords is not None:
                print(f"Adding one new segment measurement to packet at sequence {counter} for track {track.trackId}")

                #print(f"Frame: {counter}")

                #print(f"Pixel Coordinates: {current_px_coords}")

                segmentMeasurement = active_slam_msgs.SegmentMeasurement()
                segmentMeasurement.id = track.trackId
                segmentMeasurement.center = Pose2D(x=current_px_coords[0], y=current_px_coords[1], theta=0)
                segmentMeasurement.sequence = np.int32(counter) 
                # TODO: make rosparam pixel covariance
                segmentMeasurement.covariance = np.diag([3., 3.]).reshape(-1)
                # segmentMeasurement.latest_sift = current_sift
                packet.segments.append(segmentMeasurement)

        # print(packet)
        self.meas_pub.publish(packet)

        self.last_pose = T
        self.counter = self.counter + 1

        return
    
def main():

    rospy.init_node('SAM_DA_node')
    node = SAM_DA_node()
    rospy.spin()

if __name__ == "__main__":
    main()