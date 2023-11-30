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


class SAM_DA_node:

    def __init__(self):
        
        # internal variables
        self.bridge = cv_bridge.CvBridge()
        counter = 0

        # ros params
        self.num_meas_new_obj = rospy.get_param("~num_meas_new_obj", 3) # number of objects needed 
                                                                        # to create new object, default: 3
        
        # ros subscribers
        subs = [
            message_filters.Subscriber("/airsim_node/Multirotor/odom_local_ned", nav_msgs.Odometry),
            message_filters.Subscriber("/airsim_node/Multirotor/front_center_custom/Scene",
                                       sensor_msgs.Image),
            message_filters.Subscriber("/airsim_node/Multirotor/front_center_custom/Scene/camera_info", 
                                       sensor_msgs.CameraInfo),
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=1, slop=.1)
        self.ts.registerCallback(self.cb) # registers incoming messages to callback

        # ros publishers
        self.meas_pub = rospy.Publisher("measurement_packet", active_slam_msgs.MeasurementPacket, queue_size=5)

        self.blob_sam_node = BlobSAMNode()  # Instantiate BlobSAMNode class

        self.last_image = None
        self.last_pose = None

    def cb(self, *msgs):
        """
        This function gets called every time synchronized odometry, image message, and camera info 
        message are available.
        """
        odom_msg, img_msg, cam_info_msg = msgs

        counter = counter + 1

        print(f"Callback at counter {counter}")

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # extract camera intrinsics
        K = np.array([cam_info_msg.K]).reshape((3,3))

        # extract pose from odom msg
        T = np.eye(4)
        T[:3,:3] = np.array(odom_msg.pose.pose.orientation).reshape((3,3))
        T[:3,3] = np.array(odom_msg.pose.pose.position)

        # image and pose in BlobSAMNode
        self.blob_sam_node.image = img
        self.blob_sam_node.T = T
        self.blob_sam_node.filename = counter

        # Process image and get tracks
        tracks = self.blob_sam_node.process_image()

        for track_id, pixel_coords in tracks.items():
            print(f"Track ID: {track_id}, Pixel Coordinates: {pixel_coords}")
        
        # Create and publish measurement packet
        packet = active_slam_msgs.MeasurementPacket()
        packet.sequence = np.int32(counter)

        # Add relative pose measurement
        packet.relative_pose = PoseWithCovariance(np.linalg.inv(self.last_pose) @ T, np.zeros((6,6)))

        for track, track_id, pixel_coords in tracks.items():
            
            # Get the frames where the track was seen
            framesWhereSeen, _, _ = track.getPxCoordsAndDescriptorsForAllFrames()

            # If the track has been seen in more than 2 frames, add it to the measurement packet
            if len(framesWhereSeen) > 2:
                segmentMeasurement = active_slam_msgs.SegmentMeasurement()
                segmentMeasurement.id = track_id
                segmentMeasurement.center = Pose2D(x=pixel_coords[0], y=pixel_coords[1], theta=0)
                segmentMeasurement.sequence = np.int32(counter)
                segmentMeasurement.covariance = np.zeros((4,4))
                packet.segments.append(segmentMeasurement)

        print(packet)
        self.meas_pub.publish(packet)

        self.last_pose = T

        return
    
def main():

    rospy.init_node('SAM_DA_node')
    node = SAM_DA_node()
    rospy.spin()

if __name__ == "__main__":
    main()